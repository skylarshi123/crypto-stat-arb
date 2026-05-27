# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

A Python backtesting system for a **short-term mean-reversion strategy** on crypto spot markets. The core hypothesis: large price drops on high volume create oversold conditions that reverse within 24 hours. The system fetches hourly OHLCV data from Kraken (via CCXT), generates entry signals, simulates trades with realistic costs, and produces performance metrics and charts.

**Universe:** BTC/USD, ETH/USD, SOL/USD, AVAX/USD  
**Signal:** 1h price drop >2% AND volume >1.5× 24h rolling average  
**Holding periods tested:** 4h, 8h, 12h, 24h (24h is the best-performing)  
**Transaction costs:** 40 bps round-trip (20 bps in + 20 bps out)

## Running the Pipeline

Each script must be run sequentially — each stage writes files that the next stage reads:

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Fetch hourly OHLCV data → saves to data/*.csv
python src/data_fetcher.py

# Step 2: Generate entry signals → saves to results/signals.csv
python src/strategy.py

# Step 3: Backtest all holding periods → saves to results/backtest_results.csv + detailed_trades.csv
python src/backtester.py

# Step 4: Calculate performance metrics → saves to results/performance_metrics.json
python src/metrics.py

# Step 5: Generate charts → saves to results/*.png
python src/visualizations.py
```

`data_fetcher.py` skips symbols that already have a cached CSV file — delete files in `data/` to force a re-fetch.

## Architecture

The pipeline is **file-based**: each module is a standalone class with a `main()` that reads from and writes to `data/` or `results/`. Nothing is passed in memory between stages.

| Module | Class | Reads | Writes |
|---|---|---|---|
| `data_fetcher.py` | `CryptoDataFetcher` | Kraken API | `data/*_USD_hourly.csv` |
| `strategy.py` | `MeanReversionStrategy` | `data/*.csv` | `results/signals.csv` |
| `backtester.py` | `StatArbBacktester` | `results/signals.csv` | `results/backtest_results.csv`, `results/detailed_trades.csv` |
| `metrics.py` | `PerformanceMetrics` | `results/detailed_trades.csv`, `results/signals.csv` | `results/performance_metrics.json` |
| `visualizations.py` | `StrategyVisualizer` | `results/detailed_trades.csv` | `results/*.png` |

## Key Parameters

Strategy thresholds live in `MeanReversionStrategy.__init__` (`src/strategy.py`):
- `price_drop_threshold = -0.02` — minimum hourly return to trigger signal
- `volume_ratio_threshold = 1.5` — minimum volume vs 24h average
- `volume_lookback_hours = 24` — rolling window for volume baseline

`PerformanceMetrics.best_holding_period = 24` and `StrategyVisualizer.best_holding_period = 24` must stay in sync if the optimal holding period changes.

`StatArbBacktester` enforces **no overlapping positions** per symbol: a new signal for a symbol is skipped if an active position for that symbol hasn't exited yet.

## Data Format

CSV files in `data/` use the naming convention `{SYMBOL}_USD_hourly.csv` (e.g., `BTC_USD_hourly.csv`) with columns: `datetime, timestamp, open, high, low, close, volume`.

`results/signals.csv` adds `return_1h, volume_24h_avg, volume_ratio, signal, symbol` columns with `datetime` as index.

`results/detailed_trades.csv` has one row per trade per holding period — `holding_hours` column distinguishes periods.

## Live / Paper Trading

`src/live_trader.py` is designed to run as a **one-shot cron script** every hour. It loads state from `state/positions.json`, checks for exits on open positions (24h holding period), scans all symbols for new entry signals, then saves state and exits.

```bash
python src/live_trader.py             # paper mode (default)
python src/live_trader.py --summary   # print open positions + P&L, no trading
python src/live_trader.py --size 250  # paper mode with $250 per position
python src/live_trader.py --live      # real orders (needs env vars below)
```

Live mode requires `KRAKEN_API_KEY` and `KRAKEN_API_SECRET` env vars. Kraken API keys need **Query Funds + Query/Create Orders** permissions only.

**Deployment (free):** `.github/workflows/paper_trader.yml` runs the paper trader every hour via GitHub Actions (free tier). After each run it commits `state/positions.json` and `logs/trader.log` back to the repo so state survives between runs. To switch to live trading, uncomment the `--live` flag and add Kraken keys as GitHub repository secrets (`KRAKEN_API_KEY`, `KRAKEN_API_SECRET`).

**State file schema** (`state/positions.json`):
```json
{
  "paper_cash": 10000.0,
  "paper_pnl": 0.0,
  "positions": {
    "BTC/USD": { "entry_price": 95000, "entry_time": "2026-...", "size_usd": 100 }
  },
  "closed_trades": [ { "symbol": "...", "net_return": 0.03, "pnl_usd": 3.0, ... } ]
}
```

## Dependency Note

`seaborn` is used in `visualizations.py` but is not listed in `requirements.txt`. Install it separately if charts fail:
```bash
pip install seaborn
```
