"""
Live trading engine for the crypto mean-reversion strategy.
Designed to run as a one-shot script on a cron schedule (every hour).

Modes:
  paper  — logs simulated trades, no real orders placed (default)
  live   — places real orders on Kraken (requires API keys)

State is persisted to state/positions.json between runs so open
positions survive process restarts.

Usage:
  python src/live_trader.py                  # paper mode
  python src/live_trader.py --live           # live mode (needs env vars)
  python src/live_trader.py --summary        # print open positions and P&L, no action
"""

import os
import json
import argparse
import logging
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
import numpy as np
import ccxt

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/trader.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── Strategy parameters (must match backtester) ───────────────────────────────
SYMBOLS        = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]
PRICE_DROP     = -0.02   # -2% hourly return threshold
VOLUME_RATIO   = 1.5     # volume vs 24h average
VOLUME_LOOKBACK = 24     # hours for volume rolling average
HOLDING_HOURS  = 24      # exit after this many hours
POSITION_SIZE  = 100.0   # USD per trade (paper mode default)
ROUND_TRIP_BPS = 40      # 20 bps in + 20 bps out


# ── State helpers ─────────────────────────────────────────────────────────────
STATE_FILE = "state/positions.json"

def load_state() -> dict:
    """Load persisted state (open positions, paper portfolio)."""
    os.makedirs("state", exist_ok=True)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "paper_cash": 10_000.0,   # starting paper balance
        "paper_pnl": 0.0,
        "positions": {},           # symbol -> {entry_price, entry_time, size_usd}
        "closed_trades": [],
    }

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Exchange helpers ──────────────────────────────────────────────────────────
def get_exchange(live: bool = False) -> ccxt.Exchange:
    """
    Return a Kraken exchange instance.
    In paper mode we only need public endpoints (no keys required).
    In live mode reads KRAKEN_API_KEY and KRAKEN_API_SECRET from env.
    """
    if live:
        api_key = os.environ.get("KRAKEN_API_KEY")
        api_secret = os.environ.get("KRAKEN_API_SECRET")
        if not api_key or not api_secret:
            raise EnvironmentError(
                "Live mode requires KRAKEN_API_KEY and KRAKEN_API_SECRET env vars."
            )
        return ccxt.kraken({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })
    return ccxt.kraken({"enableRateLimit": True})


def fetch_recent_candles(exchange: ccxt.Exchange, symbol: str, hours: int = 50) -> pd.DataFrame:
    """
    Fetch the most recent `hours` hourly OHLCV candles for a symbol.
    50 candles is enough to compute the 24h volume average and detect a signal
    on the latest candle.
    """
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=hours)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()
    return df


# ── Signal logic (mirrors strategy.py) ───────────────────────────────────────
def has_signal(df: pd.DataFrame) -> bool:
    """
    Return True if the most recent completed candle fires an entry signal:
      - 1h return <= -2%  AND
      - volume >= 1.5x 24h rolling average
    The last row in df is the most recent completed candle.
    """
    df = df.copy()
    df["return_1h"]       = df["close"].pct_change()
    df["volume_24h_avg"]  = df["volume"].rolling(window=VOLUME_LOOKBACK, min_periods=1).mean()
    df["volume_ratio"]    = df["volume"] / df["volume_24h_avg"].replace(0, np.nan)
    df["volume_ratio"]    = df["volume_ratio"].fillna(1.0)

    last = df.iloc[-1]
    price_ok  = last["return_1h"]   <= PRICE_DROP
    volume_ok = last["volume_ratio"] >= VOLUME_RATIO
    if price_ok and volume_ok:
        logger.info(
            f"  Signal: return={last['return_1h']:.2%}  "
            f"vol_ratio={last['volume_ratio']:.2f}x  "
            f"close={last['close']}"
        )
    return bool(price_ok and volume_ok)


# ── Exit check ────────────────────────────────────────────────────────────────
def check_exits(state: dict, exchange: ccxt.Exchange, live: bool) -> None:
    """
    Close any position that has exceeded HOLDING_HOURS.
    Paper mode: compute P&L and log.
    Live mode: place a market sell order on Kraken.
    """
    now = datetime.now(timezone.utc)
    to_close = []

    for symbol, pos in state["positions"].items():
        entry_time = datetime.fromisoformat(pos["entry_time"])
        hours_held = (now - entry_time).total_seconds() / 3600
        if hours_held >= HOLDING_HOURS:
            to_close.append(symbol)

    for symbol in to_close:
        pos = state["positions"].pop(symbol)
        entry_price = pos["entry_price"]
        size_usd    = pos["size_usd"]

        # Get current price for P&L calculation
        try:
            ticker     = exchange.fetch_ticker(symbol)
            exit_price = ticker["last"]
        except Exception as e:
            logger.error(f"  Could not fetch exit price for {symbol}: {e}")
            # Put position back rather than lose track of it
            state["positions"][symbol] = pos
            continue

        gross_return = (exit_price - entry_price) / entry_price
        net_return   = gross_return - (ROUND_TRIP_BPS / 10_000)
        pnl_usd      = size_usd * net_return

        trade = {
            "symbol":      symbol,
            "entry_time":  pos["entry_time"],
            "exit_time":   now.isoformat(),
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "size_usd":    size_usd,
            "net_return":  round(net_return, 6),
            "pnl_usd":     round(pnl_usd, 4),
            "mode":        "live" if live else "paper",
        }
        state["closed_trades"].append(trade)
        state["paper_pnl"] += pnl_usd

        if live:
            try:
                coin_amount = size_usd / entry_price
                order = exchange.create_market_sell_order(symbol, coin_amount)
                logger.info(f"  LIVE SELL {symbol} | order_id={order['id']} | exit={exit_price:.4f}")
            except Exception as e:
                logger.error(f"  LIVE SELL FAILED for {symbol}: {e}")
        else:
            outcome = "WIN ✅" if net_return > 0 else "LOSS ❌"
            logger.info(
                f"  PAPER CLOSE {symbol} | {outcome} | "
                f"entry={entry_price:.4f}  exit={exit_price:.4f}  "
                f"return={net_return:.2%}  P&L=${pnl_usd:+.2f}"
            )


# ── Entry check ───────────────────────────────────────────────────────────────
def check_entries(state: dict, exchange: ccxt.Exchange, live: bool,
                  position_size: float = POSITION_SIZE) -> None:
    """
    For each symbol not currently held, fetch recent candles and check for
    an entry signal on the latest candle.
    """
    for symbol in SYMBOLS:
        if symbol in state["positions"]:
            logger.info(f"  {symbol}: already in position, skipping")
            continue

        try:
            df = fetch_recent_candles(exchange, symbol)
        except Exception as e:
            logger.error(f"  Failed to fetch data for {symbol}: {e}")
            continue

        entry_price = float(df["close"].iloc[-1])
        signal_time = df.index[-1].isoformat()

        if not has_signal(df):
            logger.info(f"  {symbol}: no signal  (close={entry_price:.4f})")
            continue

        # Enter position
        size_usd = position_size
        state["positions"][symbol] = {
            "entry_price": entry_price,
            "entry_time":  signal_time,
            "size_usd":    size_usd,
        }

        if live:
            try:
                coin_amount = size_usd / entry_price
                order = exchange.create_market_buy_order(symbol, coin_amount)
                logger.info(
                    f"  LIVE BUY {symbol} | order_id={order['id']} | "
                    f"entry={entry_price:.4f}  size=${size_usd}"
                )
            except Exception as e:
                logger.error(f"  LIVE BUY FAILED for {symbol}: {e}")
                state["positions"].pop(symbol)
        else:
            logger.info(
                f"  PAPER ENTER {symbol} | entry={entry_price:.4f}  "
                f"size=${size_usd}  exit_at={HOLDING_HOURS}h"
            )


# ── Summary printer ───────────────────────────────────────────────────────────
def print_summary(state: dict) -> None:
    logger.info("─" * 60)
    logger.info(f"Open positions: {len(state['positions'])}")
    now = datetime.now(timezone.utc)
    for symbol, pos in state["positions"].items():
        entry_time  = datetime.fromisoformat(pos["entry_time"])
        hours_held  = (now - entry_time).total_seconds() / 3600
        logger.info(
            f"  {symbol:10s} entry={pos['entry_price']:.4f}  "
            f"held={hours_held:.1f}h  size=${pos['size_usd']}"
        )

    closed = state["closed_trades"]
    if closed:
        wins     = sum(1 for t in closed if t["net_return"] > 0)
        win_rate = wins / len(closed)
        total_pnl = sum(t["pnl_usd"] for t in closed)
        logger.info(
            f"Closed trades: {len(closed)}  |  "
            f"Win rate: {win_rate:.0%}  |  "
            f"Total P&L: ${total_pnl:+.2f}"
        )
    logger.info("─" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Crypto mean-reversion live trader")
    parser.add_argument("--live",    action="store_true", help="Place real orders (default: paper mode)")
    parser.add_argument("--summary", action="store_true", help="Print current state and exit, no trading")
    parser.add_argument("--size",    type=float, default=POSITION_SIZE, help="USD per position (paper mode)")
    args = parser.parse_args()

    mode = "LIVE" if args.live else "PAPER"
    logger.info(f"{'='*60}")
    logger.info(f"Live Trader — {mode} MODE — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(f"{'='*60}")

    state    = load_state()
    exchange = get_exchange(live=args.live)

    if args.summary:
        print_summary(state)
        return

    position_size = args.size

    logger.info("Checking exits...")
    check_exits(state, exchange, live=args.live)

    logger.info("Checking entries...")
    check_entries(state, exchange, live=args.live, position_size=position_size)

    save_state(state)
    print_summary(state)


if __name__ == "__main__":
    main()
