# Cryptocurrency Statistical Arbitrage Project

## Project Goal
Build a production-quality short-term mean-reversion trading strategy that:
1. Backtests on 2+ years of hourly crypto data
2. Produces publishable performance metrics (Sharpe ratio, returns, drawdown, win rate)
3. Can be clearly explained in 60 seconds to a quant recruiter
4. Demonstrates understanding of realistic trading costs

## Success Criteria
- Sharpe ratio > 1.0 (ideally 1.5+)
- Max drawdown < 20%
- Win rate > 55%
- Annualized return > 15%
- Code runs end-to-end without errors
- Clear visualizations (equity curve, drawdown chart, returns distribution)
- Professional README with results table

## Strategy (Keep It Simple)
**Core hypothesis:** Large price drops on high volume in crypto create short-term oversold conditions that reverse within hours.

**Entry signal:**
- Price drops > 2% in 1 hour AND
- Volume > 1.5x recent average

**Exit signal:**
- Hold for 4-24 hours (test multiple holding periods)
- Exit at fixed time horizon (no complex exit logic)

**Universe:** BTC, ETH, SOL, MATIC, AVAX (5 major coins with good liquidity)

## Technical Constraints
- **Time budget:** 1-2 days maximum
- **Complexity:** Favor simplicity over sophistication - single strategy, clean implementation
- **Transaction costs:** 20 bps per trade (7 bps commission + 13 bps slippage)
- **Data source:** Free APIs only (ccxt library with Binance)
- **Backtesting:** Simple event-driven or vectorized backtest, NO complex frameworks like Zipline

## Tech Stack
- **Language:** Python 3.10+
- **Data:** `ccxt` for fetching historical OHLCV data
- **Analysis:** `pandas`, `numpy` for data manipulation
- **Backtesting:** Custom lightweight framework (100-200 lines max)
- **Visualization:** `matplotlib` or `plotly`
- **Optional:** `streamlit` for interactive dashboard (only if time permits)

## Project Structure
```
crypto-stat-arb/
├── PROJECT_SPEC.md           # This file
├── README.md                 # Results and explanation
├── requirements.txt          # Dependencies
├── src/
│   ├── data_fetcher.py      # Download historical data from Binance
│   ├── strategy.py          # Signal generation logic
│   ├── backtester.py        # Backtesting engine
│   ├── metrics.py           # Performance calculations
│   └── visualizations.py    # Charts and plots
├── data/                     # Cached price data (CSV files)
├── results/                  # Performance metrics and charts
└── notebooks/               # Optional Jupyter notebook for exploration
```

## Implementation Phases

### Phase 1: Data Pipeline (30 min)
- Fetch 2 years of hourly OHLCV data for 5 coins
- Cache to CSV for fast iteration
- Basic data validation (no missing values, reasonable prices)

### Phase 2: Strategy Logic (30 min)
- Calculate returns, volume moving averages
- Implement entry signal (price drop + high volume)
- Generate trade signals (buy/sell timestamps)

### Phase 3: Backtesting Engine (45 min)
- Simple position tracking (long only, no shorts for simplicity)
- Apply 20 bps transaction costs per trade
- Track equity curve over time
- Test holding periods: 4hr, 8hr, 12hr, 24hr

### Phase 4: Performance Metrics (30 min)
- Calculate: total return, annualized return, Sharpe ratio, max drawdown, win rate, # trades
- Compare against buy-and-hold benchmark
- Generate summary statistics table

### Phase 5: Visualizations (30 min)
- Equity curve chart
- Drawdown chart
- Returns distribution histogram
- Optional: holdings over time

### Phase 6: Documentation (15 min)
- README with strategy explanation
- Results table with all metrics
- Brief interpretation of findings

## What to AVOID
- ❌ Complex multi-strategy portfolios (stick to ONE strategy)
- ❌ Machine learning (overkill, hard to explain)
- ❌ Options, futures, derivatives (spot trading only)
- ❌ Short selling (long only is simpler)
- ❌ Dynamic position sizing (fixed size per trade)
- ❌ Optimization over 10+ parameters (max 2-3)
- ❌ Real-time trading integration (backtest only)

## Key Deliverables for Resume
1. **Sharpe ratio:** Target 1.5+
2. **Annualized return:** Target 20%+
3. **Max drawdown:** Target < 15%
4. **Win rate:** Target 60%+
5. **Number of data points:** Should be millions (2 years × 8760 hours × 5 coins = 87,600 rows)
6. **Number of trades:** Target 50-200 trades (enough to be statistically meaningful)

## Resume Bullet Points to Fill
Once complete, update resume with:
- Sharpe ratio (X)
- Annualized returns (Y%)
- Million data points (Z)
- Win rate (W%)
- Max drawdown (V%)

## Notes for Claude Code
- Prioritize working code over perfect code
- Include comments explaining key logic
- Use type hints for clarity
- Print progress/debug info during execution
- Handle edge cases gracefully (missing data, zero volume, etc.)
- Make it easy to re-run with different parameters
```

---

## How to Use This with Claude Code

**First prompt to Claude Code:**
```
I'm building a cryptocurrency statistical arbitrage project. Please read PROJECT_SPEC.md carefully - it contains the full specification, constraints, and success criteria.

Let's start with Phase 1: Data Pipeline. Create data_fetcher.py that:
1. Uses ccxt to fetch 2 years of hourly OHLCV data for BTC, ETH, SOL, MATIC, AVAX from Binance
2. Caches results to CSV files in data/ directory
3. Includes basic validation (no missing timestamps, reasonable price ranges)
4. Prints progress and summary statistics

Keep it simple and robust. Make sure it handles rate limits and saves data incrementally so we don't lose progress if interrupted.