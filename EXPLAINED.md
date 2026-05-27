# How This Whole Thing Works — Explained From Scratch

This document explains everything: what the strategy does, how the code runs,
what GitHub Actions is, and how it all connects. No assumed knowledge.

---

## Part 1: The Trading Strategy (What Are We Actually Doing?)

### The Core Idea

The strategy is based on one simple observation about crypto markets:

> **When a crypto coin drops hard and fast (especially on high volume),
> it tends to bounce back within the next 24 hours.**

This is called **mean reversion** — prices that move too far from their average
tend to snap back.

### A Concrete Example

Imagine it's 3:00 PM on a Tuesday. Bitcoin is trading at $95,000.

At 4:00 PM, Bitcoin suddenly drops to $92,100. That's a **-3% drop in one hour**.
At the same time, the trading volume is 3x higher than usual — a lot of people
are panic-selling.

Our strategy says: *"This looks like a panic selloff. I'm going to buy right now
and sell 24 hours later when it bounces back."*

So we **buy $100 of Bitcoin at $92,100**.

24 hours later (4:00 PM Wednesday), Bitcoin is back at $94,500.

We **sell at $94,500**. After fees (0.40%), we made about **$2.50 profit** on
that $100 trade — a 2.5% return in one day.

### The Entry Signal (When Do We Buy?)

Two conditions must both be true at the same time:

| Condition | Meaning | Our Threshold |
|-----------|---------|---------------|
| Big price drop | The coin fell a lot in the last hour | More than -2% |
| High volume | More people than usual are selling | 1.5× the 24-hour average |

Both conditions together = **a signal**. One alone is not enough.

**Why both?** A -2% drop on normal volume might just be a slow news day.
A -2% drop on 3× normal volume means panic selling — and panic tends to reverse.

### The Exit (When Do We Sell?)

Simple: **hold for exactly 24 hours, then sell no matter what.**

No fancy stop losses, no "sell if it drops more" logic. Just: buy, wait 24 hours, sell.

### The Four Coins We Watch

- **BTC/USD** — Bitcoin
- **ETH/USD** — Ethereum
- **SOL/USD** — Solana
- **AVAX/USD** — Avalanche

We can hold at most one position per coin at a time. So worst case we have
4 simultaneous trades open ($400 total if each position is $100).

---

## Part 2: Paper Trading vs. Real Trading

### What Is Paper Trading?

Paper trading means **pretending to trade with fake money** to see if your
strategy works before risking real money.

When our system runs in paper mode:
- It **watches real prices** from Kraken (a real crypto exchange)
- It **detects real signals** when real coins really drop
- It **logs "I would have bought here"** and **"I would have sold here"**
- It **tracks fake P&L** (profit and loss)
- But it **never actually places any orders** or touches any real money

Think of it like a flight simulator. The instruments are real, the physics are
real, but you can't actually crash a real plane.

### Why Do This First?

The backtest (the historical test) only used **30 days of past data**. That gave
us 22 trades. That's a pretty small sample. Paper trading for 4-8 weeks gives us
another 30-50 trades on *live* market conditions, so we can see:

- Does the signal fire as often as we expected?
- Are the actual returns close to the backtested returns?
- Are there any bugs in the live execution code?

If paper trading looks good → switch to real money.
If paper trading looks bad → fix the strategy before losing real money.

---

## Part 3: The Code Files and What Each One Does

Here's every file that matters, explained in plain English:

```
crypto-stat-arb/
│
├── src/
│   ├── data_fetcher.py    — Downloads price history from Kraken (the exchange)
│   ├── strategy.py        — Looks at price history, marks which hours had a signal
│   ├── backtester.py      — Simulates what would have happened if we traded those signals
│   ├── metrics.py         — Calculates Sharpe ratio, win rate, max drawdown, etc.
│   ├── visualizations.py  — Makes the charts (equity curve, drawdown chart, etc.)
│   └── live_trader.py     — The NEW file: runs every hour to check for live signals
│
├── data/
│   └── BTC_USD_hourly.csv    — Cached price history (downloaded by data_fetcher.py)
│   └── ETH_USD_hourly.csv    — (same for other coins)
│
├── results/
│   └── signals.csv           — Which hours had signals (output of strategy.py)
│   └── backtest_results.csv  — How each holding period performed
│   └── equity_curve.png      — The chart showing portfolio growth
│
├── state/
│   └── positions.json        — THE BRAIN: remembers what we currently own
│
├── logs/
│   └── trader.log            — A diary of every decision the bot made
│
└── .github/
    └── workflows/
        └── paper_trader.yml  — The instruction sheet for GitHub Actions
```

### The Most Important File: `state/positions.json`

This file is how the bot "remembers" what it's doing between runs. Here's a
realistic example after a few weeks of paper trading with 5 closed trades and
1 still open:

```json
{
  "paper_cash": 10000.0,
  "paper_pnl": 4.73,
  "positions": {
    "ETH/USD": {
      "entry_price": 3421.50,
      "entry_time": "2026-05-20T16:00:00+00:00",
      "size_usd": 100
    }
  },
  "closed_trades": [
    {
      "symbol": "BTC/USD",
      "entry_time": "2026-05-18T09:00:00+00:00",
      "exit_time":  "2026-05-19T09:00:00+00:00",
      "entry_price": 94200,
      "exit_price":  96100,
      "net_return": 0.0182,
      "pnl_usd": 1.82
    },
    {
      "symbol": "SOL/USD",
      "entry_time": "2026-05-19T14:00:00+00:00",
      "exit_time":  "2026-05-20T14:00:00+00:00",
      "entry_price": 172.40,
      "exit_price":  175.10,
      "net_return": 0.0117,
      "pnl_usd": 1.17
    },
    {
      "symbol": "AVAX/USD",
      "entry_time": "2026-05-20T11:00:00+00:00",
      "exit_time":  "2026-05-21T11:00:00+00:00",
      "entry_price": 38.20,
      "exit_price":  39.60,
      "net_return": 0.0226,
      "pnl_usd": 2.26
    },
    {
      "symbol": "BTC/USD",
      "entry_time": "2026-05-21T02:00:00+00:00",
      "exit_time":  "2026-05-22T02:00:00+00:00",
      "entry_price": 96800,
      "exit_price":  96330,
      "net_return": -0.0089,
      "pnl_usd": -0.89
    },
    {
      "symbol": "ETH/USD",
      "entry_time": "2026-05-21T07:00:00+00:00",
      "exit_time":  "2026-05-22T07:00:00+00:00",
      "entry_price": 3390.00,
      "exit_price":  3427.50,
      "net_return": 0.0071,
      "pnl_usd": 0.37
    }
  ]
}
```

**Plain English translation of every field:**

`paper_cash: 10000.0`
→ Your **starting** fake balance. This number **never changes** — it's just a
reference point so you know what you started with. In paper mode the bot doesn't
actually "spend" money when it buys, so this stays at 10,000 forever.

`paper_pnl: 4.73`
→ **Running total of profit/loss from all closed trades.**
You can verify it yourself: 1.82 + 1.17 + 2.26 + (-0.89) + 0.37 = **4.73** ✓
This is the number that tells you if the strategy is actually working.

`positions: { "ETH/USD": ... }`
→ **What the bot currently "owns" right now (not yet closed).**
We bought ETH at $3,421.50 on May 20th at 4 PM. It's still open — we haven't
hit the 24-hour exit yet. This trade is NOT counted in paper_pnl yet, because
we don't know the outcome until we close it.

`closed_trades: [...]`
→ **The complete history of every finished trade**, win or loss. Each entry
shows which coin, when we bought, when we sold, at what prices, and the P&L.
The paper_pnl number is always exactly the sum of all the pnl_usd values here.

**How much money goes into each trade?**
→ **$100 per position, by default.** This is hardcoded as `POSITION_SIZE = 100`
in `live_trader.py`. You can change it at runtime:
```bash
python src/live_trader.py --size 250   # $250 per position
```
Since we watch 4 coins, the worst case is 4 simultaneous open positions = $400
total "at risk" at any one time. The bot never "runs out" of paper money — it
just independently tracks the P&L from each $100 trade.

Every time the bot runs, it reads this file, does its work, and writes it back
with any updates.

---

## Part 4: GitHub Actions — Free Cloud Computer

### What Problem Does This Solve?

The live trader needs to run **every single hour**, forever. That means:

- 3 AM on a Tuesday? It runs.
- You're on vacation in Europe? It runs.
- Your laptop is closed? It runs.

You can't do this by hand. You need a computer that's always on. But renting a
server costs money. This is where GitHub Actions comes in.

### What Is GitHub Actions?

GitHub Actions is a feature of GitHub (the website where you store code) that
lets you say:

> **"Every hour, wake up, run my Python script, then go back to sleep."**

GitHub's servers do this for you, for free (up to 2,000 minutes per month on
private repos, unlimited on public repos).

Think of it like setting an alarm on someone else's phone. You don't need your
own phone on — theirs will go off, run your script, and report back.

### What Does Our Workflow File Actually Say?

Here's `.github/workflows/paper_trader.yml` broken down line by line:

```yaml
name: Paper Trader (Hourly)
```
→ Just a display name you'll see on GitHub's website.

```yaml
on:
  schedule:
    - cron: "0 * * * *"
```
→ "Run this at minute 0 of every hour." (`0 * * * *` is cron syntax for "every
hour on the hour". Cron is a 50-year-old Unix scheduling system.)

```yaml
  workflow_dispatch:
```
→ Also lets you click a button on GitHub's website to run it manually right now.

```yaml
jobs:
  trade:
    runs-on: ubuntu-latest
```
→ "Spin up a fresh Linux computer (Ubuntu) to do the work."

```yaml
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4
```
→ "Download all my code files onto this temporary computer."

```yaml
      - name: Install dependencies
        run: pip install -r requirements.txt
```
→ "Install the Python packages the code needs (pandas, ccxt, etc.)."

```yaml
      - name: Run paper trader
        run: python src/live_trader.py
```
→ "Actually run the script."

```yaml
      - name: Commit updated state back to repo
        run: |
          git add state/positions.json logs/trader.log
          git diff --staged --quiet || git commit -m "chore: update trading state [skip ci]"
          git push
```
→ "Save the updated `positions.json` and `trader.log` back to GitHub so next
hour's run can see what positions we currently have."

This last step is crucial. Without it, next hour's computer would start fresh
with no memory of what we bought.

### The Full Hourly Cycle, Step by Step

Here's exactly what happens at, say, 3:00 PM every day:

```
3:00:00 PM  GitHub's alarm fires
3:00:05 PM  A fresh Linux computer boots up in GitHub's data center
3:00:15 PM  It downloads your code from GitHub
3:00:25 PM  It installs pandas, ccxt, numpy, etc.
3:00:35 PM  It reads state/positions.json (what do we currently own?)
3:00:36 PM  It asks Kraken: "What was the last hourly candle for BTC?"
3:00:36 PM  Kraken responds: close=$95,200, volume=1,200 BTC
3:00:36 PM  Code checks: did price drop >2%? Is volume >1.5x average?
            → No signal for BTC
3:00:37 PM  Repeat for ETH, SOL, AVAX...
3:00:38 PM  Also checks: have we held ETH for 24+ hours yet? → No, 18 hours
3:00:39 PM  Nothing to do. Logs "no signal for any symbol."
3:00:40 PM  Saves positions.json (unchanged) back to GitHub
3:00:45 PM  Computer shuts down and disappears forever
3:00:46 PM  GitHub shows a green checkmark on the Actions tab
```

Then at 4:00 PM, a completely fresh computer does the exact same thing.
The only continuity between runs is the `positions.json` file.

---

## Part 5: A Full Worked Example of a Real Trade Cycle

Let's walk through a complete trade from signal to close.

### Monday 2:00 PM — Normal hour, no signal

```
live_trader.py runs:
  Reads positions.json → no open positions
  Fetches BTC candle: dropped -0.8%, volume 1.1x average → NO SIGNAL
  Fetches ETH candle: up +0.3% → NO SIGNAL
  Fetches SOL candle: dropped -1.4%, volume 1.2x average → NO SIGNAL (below -2% threshold)
  Fetches AVAX candle: dropped -0.5% → NO SIGNAL
  Logs: "no signals"
  positions.json unchanged
```

### Monday 6:00 PM — Signal fires on ETH!

```
live_trader.py runs:
  Reads positions.json → no open positions
  Fetches BTC candle → no signal
  Fetches ETH candle: dropped -2.8%, volume 2.1x average → SIGNAL! ✅
    (Both conditions met: price drop > 2%, volume > 1.5x)
  
  PAPER MODE: "I would buy $100 of ETH at $3,280"
  
  Writes to positions.json:
    "ETH/USD": {
      "entry_price": 3280,
      "entry_time": "2026-05-26T18:00:00+00:00",
      "size_usd": 100
    }
  
  Logs: "PAPER ENTER ETH/USD | entry=3280.00 size=$100 exit_at=24h"
  Saves positions.json to GitHub
```

### Monday 8:00 PM — Checks for exit

```
live_trader.py runs:
  Reads positions.json → we own ETH, entered at 6 PM
  Checks: current time is 8 PM, we entered at 6 PM → only 2 hours held
  24 hours NOT reached yet, do nothing
  Checks entries: ETH already owned, skip it
  Checks BTC, SOL, AVAX → no signals
  positions.json unchanged
```

### Tuesday 6:00 PM — 24 hours elapsed, time to close!

```
live_trader.py runs:
  Reads positions.json → we own ETH, entered at 6 PM yesterday
  Checks: current time is 6 PM today → exactly 24 hours held ✅ TIME TO CLOSE
  
  Asks Kraken: "What's ETH trading at right now?"
  Kraken says: $3,362
  
  Calculates:
    gross return = (3362 - 3280) / 3280 = +2.50%
    minus fees  = -0.40%
    net return  = +2.10%
    P&L         = $100 × 2.10% = +$2.10
  
  PAPER MODE: "I would have sold $100 of ETH at $3,362, made $2.10"
  
  Updates positions.json:
    removes ETH from "positions"
    adds trade to "closed_trades"
    adds $2.10 to "paper_pnl"
  
  Logs: "PAPER CLOSE ETH/USD | WIN ✅ | entry=3280 exit=3362 return=+2.10% P&L=+$2.10"
```

### What if ETH went DOWN instead?

```
Kraken says: $3,195 (dropped further)

Calculates:
  gross return = (3195 - 3280) / 3280 = -2.59%
  minus fees  = -0.40%
  net return  = -2.99%
  P&L         = $100 × -2.99% = -$2.99

Logs: "PAPER CLOSE ETH/USD | LOSS ❌ | entry=3280 exit=3195 return=-2.99% P&L=-$2.99"
```

This happens. The backtest showed a 72.7% win rate — meaning about 1 in 4 trades
is a loser. The goal is for the winners to outweigh the losers on average.

---

## Part 6: How to Monitor It

### Option 1: Check the log file on GitHub

After each run, GitHub commits `logs/trader.log` to your repo. Go to:
`github.com/yourusername/crypto-stat-arb` → click `logs/trader.log`

You'll see something like:
```
2026-05-26 18:00:31 [INFO] Live Trader — PAPER MODE — 2026-05-26 18:00 UTC
2026-05-26 18:00:32 [INFO] Checking exits...
2026-05-26 18:00:33 [INFO] Checking entries...
2026-05-26 18:00:33 [INFO]   BTC/USD: no signal  (close=95200.0000)
2026-05-26 18:00:34 [INFO]   Signal: return=-2.80%  vol_ratio=2.10x  close=3280.0
2026-05-26 18:00:34 [INFO]   PAPER ENTER ETH/USD | entry=3280.00  size=$100  exit_at=24h
```

### Option 2: Check positions.json on GitHub

Go to `github.com/yourusername/crypto-stat-arb` → click `state/positions.json`
You'll see exactly what the bot thinks it owns right now.

### Option 3: Run the summary command locally

```bash
python src/live_trader.py --summary
```

Output:
```
────────────────────────────────────────
Open positions: 1
  ETH/USD    entry=3280.0000  held=6.2h  size=$100
Closed trades: 3  |  Win rate: 67%  |  Total P&L: +$4.23
────────────────────────────────────────
```

---

## Part 7: The Transition to Real Money

When you're satisfied with paper trading results (suggest 4-8 weeks, 20+ trades):

### Step 1: Get Kraken API keys
1. Make a Kraken account at kraken.com
2. Settings → API → Generate Key
3. Permissions needed: **Query Funds**, **Create & Modify Orders**, **Query Open Orders**
4. Copy the key and secret

### Step 2: Add keys to GitHub Secrets (NOT the code)
Never put API keys in your code files — if you commit them to GitHub, they're
public and someone will steal your money within minutes.

Instead:
1. Go to your GitHub repo
2. Settings → Secrets and variables → Actions → New repository secret
3. Add `KRAKEN_API_KEY` (paste the key)
4. Add `KRAKEN_API_SECRET` (paste the secret)

GitHub stores them encrypted. Your workflow file accesses them as
`${{ secrets.KRAKEN_API_KEY }}` without ever exposing them.

### Step 3: Uncomment 3 lines in the workflow file

In `.github/workflows/paper_trader.yml`, change:
```yaml
      - name: Run paper trader
        run: python src/live_trader.py
        # For live trading later, use:
        # run: python src/live_trader.py --live
        # env:
        #   KRAKEN_API_KEY: ${{ secrets.KRAKEN_API_KEY }}
        #   KRAKEN_API_SECRET: ${{ secrets.KRAKEN_API_SECRET }}
```

To:
```yaml
      - name: Run live trader
        run: python src/live_trader.py --live
        env:
          KRAKEN_API_KEY: ${{ secrets.KRAKEN_API_KEY }}
          KRAKEN_API_SECRET: ${{ secrets.KRAKEN_API_SECRET }}
```

Commit and push. That's it. Real orders will now be placed on Kraken.

### Suggested starting capital: $400-500

That's $100 per coin, max 4 simultaneous positions. Small enough that a bad run
won't hurt much. Once you have 20-30 live trades and they match paper results,
you can scale up.

---

## Part 8: What Could Go Wrong

It's important to know the risks before using real money:

| Risk | What it means | How likely |
|------|--------------|------------|
| **Strategy stops working** | Market regime changes, the edge disappears | Moderate — crypto markets change |
| **Exchange downtime** | Kraken goes down right when you need to sell | Low, but has happened |
| **Worse slippage than modeled** | You assumed 13 bps slippage; reality might be 30 bps | Moderate during volatile periods |
| **Small sample size** | 22 backtest trades isn't a lot of statistical evidence | This is real — paper trade first |
| **Tax complexity** | Each trade is a taxable event in the US | Certain — keep records |

---

## Summary: The Full Picture

```
YOUR STRATEGY:
  Every hour → check if any coin dropped >2% on high volume
  If yes → "buy" $100 of it
  24 hours later → "sell" it
  Hope it bounced back

YOUR INFRASTRUCTURE:
  GitHub Actions = free alarm clock that runs your Python script every hour
  positions.json = the bot's memory (what it currently owns)
  trader.log     = the bot's diary (what it has done)
  paper mode     = watch real prices, fake the trades
  live mode      = watch real prices, place real orders on Kraken

THE FLOW:
  Push code to GitHub
    → Actions runs hourly
    → Script reads positions.json
    → Script checks Kraken for signals
    → Script opens/closes paper positions
    → Script writes updated positions.json back to GitHub
    → Repeat forever
```

Paper trade for 4-8 weeks. If it looks good, flip the switch to live.
