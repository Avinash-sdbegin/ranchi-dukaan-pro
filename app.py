import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_SALES = BASE_DIR / "sample_data" / "sales_history.csv"
SAMPLE_INVENTORY = BASE_DIR / "sample_data" / "current_inventory.csv"

INVENTORY_COLUMNS = [
    "Product", "Current Stock", "Incoming Stock",
    "Lead Time (days)", "Safety Stock", "Unit Cost (Rs)"
]
QUANTITY_COLUMNS = [
    "Current Stock", "Incoming Stock",
    "Lead Time (days)", "Safety Stock"
]


@lru_cache(maxsize=1)
def get_pipeline():
    token = os.getenv("TABPFN_TOKEN")
    if not token:
        raise RuntimeError(
            "TABPFN_TOKEN is missing. Set it as an environment variable "
            "or app secret. Never put the token in source code."
        )

    os.environ["TABPFN_TOKEN"] = token

    return TabPFNTSPipeline(
        tabpfn_mode=TabPFNMode.CLIENT,
        max_context_length=4096
    )


def load_sales(upload):
    path = upload if isinstance(upload, str) else (
        upload.name if upload is not None else None
    )
    df = pd.read_csv(path) if path else pd.read_csv(SAMPLE_SALES)

    required = {"item_id", "timestamp", "target"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Sales CSV must have columns: item_id, timestamp, target."
        )

    df = df[["item_id", "timestamp", "target"]].copy()
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )
    df["target"] = pd.to_numeric(df["target"], errors="coerce")

    if df.empty or df["item_id"].eq("").any():
        raise ValueError("Sales data is empty or has blank product names.")
    if df["timestamp"].isna().any() or df["target"].isna().any():
        raise ValueError("Sales CSV contains invalid dates or quantities.")
    if (df["target"] < 0).any():
        raise ValueError("Sales quantities cannot be negative.")

    # Combine duplicate product/date rows.
    df = (
        df.groupby(["item_id", "timestamp"], as_index=False)["target"]
        .sum()
        .sort_values(["item_id", "timestamp"])
    )

    if df["item_id"].nunique() > 100:
        raise ValueError("MVP limit: at most 100 products per run.")

    # Require daily history rather than silently interpreting missing
    # calendar dates as zero sales.
    for product, group in df.groupby("item_id"):
        dates = group["timestamp"].sort_values()
        if len(dates) < 14:
            raise ValueError(
                f"{product} has fewer than 14 daily observations. "
                "Add more history before forecasting."
            )
        if (dates.diff().dropna() != pd.Timedelta(days=1)).any():
            raise ValueError(
                f"{product} has missing calendar dates. "
                "Check the sales records; do not fill unknown days with zero."
            )

    return df


def forecast_and_recommend(upload, inventory_input):
    empty_forecast = pd.DataFrame()
    empty_recommendations = pd.DataFrame()

    try:
        sales_df = load_sales(upload)

        stock_df = (
            pd.read_csv(SAMPLE_INVENTORY)
            if inventory_input is None
            else pd.DataFrame(inventory_input).copy()
        )

        if not set(INVENTORY_COLUMNS).issubset(stock_df.columns):
            raise ValueError("Inventory table is missing required columns.")

        stock_df = stock_df[INVENTORY_COLUMNS].copy()
        stock_df["Product"] = stock_df["Product"].astype(str).str.strip()

        for col in INVENTORY_COLUMNS[1:]:
            stock_df[col] = pd.to_numeric(stock_df[col], errors="coerce")

        if stock_df[INVENTORY_COLUMNS[1:]].isna().any().any():
            raise ValueError("Enter valid numbers in all inventory fields.")
        if (stock_df[INVENTORY_COLUMNS[1:]] < 0).any().any():
            raise ValueError("Inventory values and unit costs cannot be negative.")
        if stock_df["Product"].eq("").any():
            raise ValueError("Product names cannot be blank.")
        if stock_df["Product"].duplicated().any():
            raise ValueError("Each product must appear once in inventory.")

        for col in QUANTITY_COLUMNS:
            if (stock_df[col] % 1 != 0).any():
                raise ValueError(f"{col} must use whole numbers.")

        stock_df[QUANTITY_COLUMNS] = (
            stock_df[QUANTITY_COLUMNS].astype(int)
        )
        if (stock_df["Lead Time (days)"] < 1).any():
            raise ValueError("Lead time must be at least one day.")

        pipeline = get_pipeline()

        raw = pipeline.predict_df(
            context_df=sales_df,
            prediction_length=7,
            quantiles=[0.1, 0.5, 0.9]
        ).reset_index()

        needed = {"item_id", "timestamp", "target"}
        if not needed.issubset(raw.columns):
            raise ValueError("Unexpected forecast columns from TabPFN-TS.")

        forecast = raw[["item_id", "timestamp", "target"]].copy()
        forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
        forecast["target"] = pd.to_numeric(
            forecast["target"], errors="coerce"
        )

        if forecast["target"].isna().any():
            raise ValueError("Model returned invalid forecast values.")

        forecast["Estimated Units Sold"] = (
            forecast["target"].clip(lower=0).round().astype(int)
        )

        forecast_out = (
            forecast.rename(columns={
                "item_id": "Product",
                "timestamp": "Forecast Date"
            })[["Product", "Forecast Date", "Estimated Units Sold"]]
            .sort_values(["Product", "Forecast Date"])
        )

        recommendation_rows = []

        for _, item in stock_df.iterrows():
            product = item["Product"]
            product_forecast = (
                forecast_out[forecast_out["Product"] == product]
                .sort_values("Forecast Date")
            )

            if product_forecast.empty:
                recommendation_rows.append({
                    "Product": product,
                    "7-Day Forecast": None,
                    "Lead-Time Demand": None,
                    "Recommended Order": None,
                    "Estimated Order Cost (Rs)": None,
                    "Status": "No matching sales forecast"
                })
                continue

            daily = product_forecast["Estimated Units Sold"].tolist()
            weekly_demand = int(sum(daily))
            lead_days = int(item["Lead Time (days)"])

            # Do not make an incomplete recommendation if lead time exceeds
            # the 7-day prediction horizon.
            if lead_days > len(daily):
                recommendation_rows.append({
                    "Product": product,
                    "7-Day Forecast": weekly_demand,
                    "Lead-Time Demand": None,
                    "Recommended Order": None,
                    "Estimated Order Cost (Rs)": None,
                    "Status": "Lead time exceeds forecast horizon"
                })
                continue

            lead_demand = int(sum(daily[:lead_days]))
            order_qty = max(
                0,
                lead_demand
                + int(item["Safety Stock"])
                - int(item["Current Stock"])
                - int(item["Incoming Stock"])
            )
            order_cost = order_qty * float(item["Unit Cost (Rs)"])

            if order_qty > 0:
                status = "Restock suggested"
            elif int(item["Current Stock"]) > (
                weekly_demand + int(item["Safety Stock"])
            ):
                status = "Check possible overstock"
            else:
                status = "No immediate reorder by this rule"

            recommendation_rows.append({
                "Product": product,
                "7-Day Forecast": weekly_demand,
                "Lead-Time Demand": lead_demand,
                "Recommended Order": order_qty,
                "Estimated Order Cost (Rs)": round(order_cost, 2),
                "Status": status
            })

        recommendations = pd.DataFrame(recommendation_rows)
        estimated_cost = recommendations[
            "Estimated Order Cost (Rs)"
        ].sum(skipna=True)

        chart = px.line(
            forecast_out,
            x="Forecast Date",
            y="Estimated Units Sold",
            color="Product",
            markers=True,
            title="Estimated Demand — Next 7 Days"
        )
        chart.update_layout(
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Estimated units sold"
        )

        message = (
            f"✅ Forecast complete for {forecast['item_id'].nunique()} "
            "products over 7 days.\n\n"
            f"🛒 Estimated order cost from matched inventory: "
            f"₹{estimated_cost:,.2f}\n\n"
            "⚠️ Forecasts are estimates. The bundled sales and inventory "
            "are synthetic examples, not real shop data. Verify stock, "
            "incoming deliveries, expiry and supplier requirements."
        )

        return forecast_out, recommendations, chart, message

    except Exception as exc:
        return (
            empty_forecast,
            empty_recommendations,
            None,
            f"❌ {type(exc).__name__}: {exc}"
        )


default_inventory = pd.read_csv(SAMPLE_INVENTORY)

with gr.Blocks(title="Ranchi Dukaan Pro") as demo:
    gr.Markdown("""
    # 🏪 Ranchi Dukaan Pro
    ### Smart Stock & Demand Predictor

    Forecast estimated demand for the next 7 days and review inventory
    reorder suggestions. This prototype uses TabPFN-TS in client/cloud mode.

    **Demo mode:** Leave sales CSV empty to use synthetic sample data.
    """)

    sales_upload = gr.File(
        label="Sales history CSV (optional)",
        file_types=[".csv"],
        type="filepath"
    )

    gr.Markdown(
        "Sales CSV columns: `item_id`, `timestamp`, `target`. "
        "Use one row per product per calendar day. "
        "Do not upload private customer information."
    )

    inventory_editor = gr.Dataframe(
        value=default_inventory[INVENTORY_COLUMNS].values.tolist(),
        headers=INVENTORY_COLUMNS,
        datatype=["str", "number", "number", "number", "number", "number"],
        label="📦 Edit current inventory",
        interactive=True,
        type="pandas"
    )

    run_button = gr.Button("📈 Forecast & Recommend", variant="primary")
    forecast_table = gr.Dataframe(label="📊 7-Day Demand Forecast")
    recommendation_table = gr.Dataframe(
        label="🛒 Smart Restock Recommendations"
    )
    chart_output = gr.Plot(label="Forecast Chart")
    status_output = gr.Markdown()

    run_button.click(
        fn=forecast_and_recommend,
        inputs=[sales_upload, inventory_editor],
        outputs=[
            forecast_table,
            recommendation_table,
            chart_output,
            status_output
        ]
    )

if __name__ == "__main__":
    demo.launch()
