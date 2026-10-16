# Ranchi Dukaan Pro 🏪

### Smart Stock & Demand Predictor

Ranchi Dukaan Pro is a prototype that uses **TabPFN-TS** to estimate
the next seven days of product sales and combine those estimates with
inventory rules to suggest possible reorder quantities.

> Forecasts are estimates, not guarantees. Bundled sales and inventory
> are synthetic examples. No real shop performance has been established.

## Features

- Optional CSV upload for daily product sales
- Seven-day demand forecast using TabPFN-TS
- Editable current stock, incoming stock, lead time, safety stock and unit cost
- Rule-based reorder recommendations and estimated order cost
- Forecast chart and downloadable evaluation evidence in `evaluation/`

## How it works

1. Read daily sales history (`item_id`, `timestamp`, `target`).
2. Generate a seven-day forecast with TabPFN-TS.
3. Sum estimated demand during the supplier lead time.
4. Calculate a suggested order quantity:

   `max(0, lead-time demand + safety stock - current stock - incoming stock)`

5. Display estimates and warnings in a Gradio dashboard.

The forecasting model and the stock-reorder rules are separate components.
A suggested order is not a purchase instruction.


## 📸 Screenshots

### Dashboard
![Ranchi Dukaan Pro Dashboard](screenshots/dashboard.png)

### Inventory Test
![Inventory Test](screenshots/inventory-test.png)

### Forecast Results
![Seven-Day Forecast Results](screenshots/forecast-results.png)


## Run locally

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Set the Prior Labs API token as an environment variable. Do not hardcode
the token or commit it to GitHub.

**macOS / Linux:**

```bash
export TABPFN_TOKEN="YOUR_TOKEN"
python app.py
```

**Windows PowerShell:**

```powershell
$env:TABPFN_TOKEN = "YOUR_TOKEN"
python app.py
```

Create/manage credentials through the official Prior Labs platform.
The app uses TabPFN-TS client/cloud inference; uploaded sales history may be
sent to the model provider for inference. Do not upload confidential data
without permission and reviewing applicable provider terms.

## Input CSV format

The sales CSV requires these columns:

| Column | Meaning |
|---|---|
| `item_id` | Product name; must match the inventory table for reorder suggestions |
| `timestamp` | Calendar date |
| `target` | Units sold that day |

Use daily records with consecutive dates for each product. If a day is
missing, verify the source records first. Do not assume an unknown value
means zero sales.

The bundled files in `sample_data/` are generated synthetic examples, not
data collected from a real Ranchi business.

## Evaluation: synthetic-data holdout

A single seven-day holdout was run on three synthetic product series.

| Product | TabPFN MAE | Baseline MAE | TabPFN WAPE | Baseline WAPE |
|---|---:|---:|---:|---:|
| All products | 1.29 | 1.62 | 9.18% | 11.56% |
| Bread | 1.43 | 1.29 | 10.00% | 9.00% |
| Milk 500 ml | 1.14 | 2.14 | 5.71% | 10.71% |
| Rice 1 kg | 1.29 | 1.43 | 16.67% | 18.52% |

- Baseline: sales from the same weekday one week earlier.
- Lower MAE/WAPE is better.
- The aggregate result favored TabPFN in this single test window, but
  the simple baseline performed better for Bread.
- Synthetic data and one seven-day holdout do **not** establish real-world
  accuracy. Further backtesting and real-shop validation with permission
  are needed.

Evaluation outputs are saved in `evaluation/`:
- `backtest_metrics.csv`
- `backtest_comparison.csv`
- `backtest_chart.html`

## Limitations

- This is an educational/hackathon prototype, not a production inventory system.
- Demand predictions may be wrong, especially with short or irregular history,
  promotions, festivals, sudden price changes, stock-outs or changing suppliers.
- Current restock logic uses a seven-day forecast and simple safety-stock rules.
- It does not verify live supplier availability, product expiry, opening hours,
  holidays or customer demand.
- No cost savings, sales lift or reduction in wastage has been demonstrated.
- The public Gradio share link used during development is temporary.

## Open innovation

The project uses TabPFN-TS, an open-source time-series forecasting implementation
built around TabPFN. This prototype uses client/cloud inference rather than
claiming fully offline or local inference. The code and stock-planning logic can
be inspected and adapted, while model access and data handling remain subject
to the provider's service and terms.

## Next improvements

- Walk-forward evaluation across multiple time windows
- Baseline/model comparison per product
- Better treatment of stock-outs, closed days and missing sales data
- Expiry-aware ordering and supplier minimum-order quantities
- Feedback and usability testing with a local shopkeeper, with permission

## License

MIT
