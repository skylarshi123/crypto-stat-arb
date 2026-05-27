# Go-Live Plan

This document is written for a future Claude Code session. It explains the
current state of the project, what needs to be true before going live, and
exactly what to implement when the time comes.

---

## Current State (as of May 2026)

The strategy is **paper trading** via GitHub Actions. Every hour, automatically:

1. `src/live_trader.py` runs on GitHub's free servers
2. It checks all 4 coins (BTC, ETH, SOL, AVAX) for entry signals on Kraken
3. It logs fake trades to `state/positions.json` and `logs/trader.log`
4. No real money is involved yet

The paper trader has been running and accumulating trade history. Check
`state/positions.json` to see current open positions and closed trade history,
and `logs/trader.log` for a full run-by-run diary.

---

## Go-Live Checklist

Before implementing anything real, verify ALL of these are true by reading
`state/positions.json`:

| # | Check | Minimum bar |
|---|-------|-------------|
| 1 | Total closed paper trades | ≥ 30 |
| 2 | Win rate (% of closed trades with net_return > 0) | > 55% |
| 3 | Average net return per trade | > 0.5% |
| 4 | Worst single trade loss | > -8% |
| 5 | Overall paper_pnl trend | Positive |
| 6 | Signal rate (trades per week) | 3–8 per week across all coins |

Calculate win rate from `positions.json` like this:
```bash
python3 - << 'EOF'
import json
state = json.load(open("state/positions.json"))
trades = state["closed_trades"]
wins = sum(1 for t in trades if t["net_return"] > 0)
avg_return = sum(t["net_return"] for t in trades) / len(trades)
print(f"Closed trades : {len(trades)}")
print(f"Win rate      : {wins/len(trades):.1%}")
print(f"Avg return    : {avg_return:.2%}")
print(f"Total P&L     : ${state['paper_pnl']:.2f}")
EOF
```

If any check fails, do not go live. Wait another 2-4 weeks and re-evaluate.

---

## What "Going Live" Means Technically

The strategy logic, signal detection, position tracking, and GitHub Actions
schedule all stay exactly the same. The **only thing that changes** is the
execution layer in `src/live_trader.py` — specifically the two lines that
place orders:

```python
# Currently in check_entries() — paper mode logs this but never runs it:
exchange.create_market_buy_order(symbol, coin_amount)

# Currently in check_exits() — paper mode logs this but never runs it:
exchange.create_market_sell_order(symbol, coin_amount)
```

These two calls need to be replaced with the Robinhood API equivalent.
Everything else stays unchanged.

---

## The Robinhood Integration (implement this when ready)

At the time this document was written, the Robinhood Crypto Trading API docs
had not yet been reviewed. The user will paste the docs into Claude Code at
implementation time.

### What to verify from the docs before writing any code:

1. **Supported coins** — confirm BTC, ETH, SOL, AVAX are all tradeable via API
2. **Fee structure** — our model assumes 0.40% round-trip (20 bps each way);
   if Robinhood charges more, note it in CLAUDE.md and consider whether the
   edge still holds
3. **Order types** — we use market orders; confirm market orders are supported
   for crypto
4. **Rate limits** — we make 4 API calls per hour (one per coin); should be
   well within any reasonable limit
5. **Authentication method** — OAuth, API key, or something else

### Where to make the changes in `live_trader.py`:

**1. Replace `get_exchange()` with a Robinhood auth function**

Currently:
```python
def get_exchange(live: bool = False) -> ccxt.Exchange:
    # returns a ccxt.kraken instance
```

Replace with whatever the Robinhood SDK/API requires for authentication.
Keep the paper mode path (using Kraken public data for price checks) unchanged —
we still want to fetch real prices from Kraken even when executing on Robinhood.

**2. Replace order placement in `check_entries()`**

Find this block (around line 145 in the current file):
```python
if live:
    try:
        coin_amount = size_usd / entry_price
        order = exchange.create_market_buy_order(symbol, coin_amount)
        logger.info(f"  LIVE BUY {symbol} | order_id={order['id']} ...")
```

Replace with Robinhood's buy call. Keep the same logging format so the log
file stays readable.

**3. Replace order placement in `check_exits()`**

Find this block (around line 100 in the current file):
```python
if live:
    try:
        coin_amount = size_usd / entry_price
        order = exchange.create_market_sell_order(symbol, coin_amount)
        logger.info(f"  LIVE SELL {symbol} | order_id={order['id']} ...")
```

Replace with Robinhood's sell call.

**4. Update the workflow file to use Robinhood secrets**

In `.github/workflows/paper_trader.yml`, replace:
```yaml
        # env:
        #   KRAKEN_API_KEY: ${{ secrets.KRAKEN_API_KEY }}
        #   KRAKEN_API_SECRET: ${{ secrets.KRAKEN_API_SECRET }}
```

With whatever env vars the Robinhood API needs (e.g. `ROBINHOOD_API_KEY`,
`ROBINHOOD_CLIENT_ID`, etc. — depends on their auth flow).

Add those secrets to the GitHub repo under:
Settings → Secrets and variables → Actions → New repository secret

---

## Position Sizing for Live Trading

Start conservative regardless of how good paper results look:

| Phase | Position size | Max simultaneous exposure |
|-------|--------------|--------------------------|
| First 3-4 weeks live | $25–50 per trade | $100–200 total |
| After 20+ live trades match paper results | $100 per trade | $400 total |
| Scale up further only if | Live win rate > 55% sustained | — |

To change position size, use the `--size` flag:
```bash
python src/live_trader.py --live --size 50
```

Or edit `POSITION_SIZE = 100.0` at the top of `live_trader.py`.

---

## Note on Symbol Names

Kraken uses `BTC/USD`, `ETH/USD`, etc. Robinhood may use different ticker
formats (`BTC-USD`, `BTCUSD`, or just `BTC`). When integrating, check what
format the Robinhood API expects and update the `SYMBOLS` list at the top of
`live_trader.py` if needed. The signal detection logic uses the symbol string
only for logging and position tracking — it doesn't care what format it is,
as long as it's consistent.
