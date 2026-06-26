"""
LCID mean-reversion short paper trader.

Signal  : LCID intraday price vs previous day's close >= +10%
Entry   : short at current market price when signal fires
Exit    : 3 trading days after entry (best from backtest: 75% win rate, +3.8% avg)
Costs   : 20 bps round-trip (conservative retail equity estimate)

Run every 15 minutes during market hours via cron or GitHub Actions:
  python src/lucid_short_trader.py              # check signal, act if triggered
  python src/lucid_short_trader.py --summary    # print open position + P&L, no action
  python src/lucid_short_trader.py --size 500   # override paper position size (USD)
  python src/lucid_short_trader.py --reset      # wipe state and start fresh

Live mode (Robinhood MCP — not yet enabled):
  python src/lucid_short_trader.py --live       # raises NotImplementedError until MCP is wired

State persisted to state/lcid_positions.json between runs.
Logs appended to logs/lcid_trader.log.
"""

import os
import json
import argparse
import logging
from datetime import datetime, timezone, date, timedelta
from zoneinfo import ZoneInfo
from typing import Optional

import yfinance as yf
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/lcid_trader.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Strategy parameters (match backtest) ─────────────────────────────────────
TICKER           = "LCID"
SPIKE_THRESHOLD  = 0.10     # short when intraday return vs prev close >= +10%
HOLD_DAYS        = 3        # exit after 3 trading days (best from backtest)
POSITION_SIZE    = 500.0    # paper USD per trade (override with --size)
ROUND_TRIP_COST  = 0.0020   # 20 bps round-trip

ET = ZoneInfo("America/New_York")
MARKET_OPEN  = (9, 30)   # 9:30 AM ET
MARKET_CLOSE = (16, 0)   # 4:00 PM ET

# ── State ─────────────────────────────────────────────────────────────────────
STATE_FILE = "state/lcid_positions.json"

def load_state() -> dict:
    os.makedirs("state", exist_ok=True)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "paper_pnl": 0.0,
        "position": None,       # only one at a time: LCID short
        "closed_trades": [],
        "last_signal_date": None,  # prevent double-firing same day
    }

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Market data ───────────────────────────────────────────────────────────────
def is_market_hours() -> bool:
    now_et = datetime.now(ET)
    t = (now_et.hour, now_et.minute)
    return MARKET_OPEN <= t < MARKET_CLOSE and now_et.weekday() < 5

def get_current_price() -> Optional[float]:
    try:
        info = yf.Ticker(TICKER).fast_info
        price = info.get("last_price") or info.get("regularMarketPrice")
        return float(price) if price else None
    except Exception as e:
        logger.error(f"Failed to fetch current price: {e}")
        return None

def get_prev_close() -> Optional[float]:
    """Previous *completed* trading day's close."""
    try:
        df = yf.download(TICKER, period="5d", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return None
        # Last row might be today's partial session — use second-to-last
        today = date.today()
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        completed = closes[closes.index.date < today]
        return float(completed.iloc[-1]) if len(completed) >= 1 else None
    except Exception as e:
        logger.error(f"Failed to fetch prev close: {e}")
        return None

def get_trading_days_since(entry_date_str: str) -> int:
    """Count completed trading sessions since entry (using LCID trading history)."""
    try:
        entry_date = pd.to_datetime(entry_date_str).date()
        df = yf.download(TICKER, period="30d", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        closes = df["Close"].dropna()
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        today = date.today()
        # trading days strictly after entry and strictly before today
        # (today's session may not be complete)
        days_after = closes[
            (closes.index.date > entry_date) & (closes.index.date < today)
        ]
        return len(days_after)
    except Exception as e:
        logger.error(f"Failed to count trading days: {e}")
        return 0


# ── Signal ────────────────────────────────────────────────────────────────────
def check_signal(state: dict) -> tuple[bool, float, float]:
    """
    Returns (signal_fired, current_price, intraday_return).
    Signal fires when intraday return vs prev close >= SPIKE_THRESHOLD.
    Only fires once per calendar day.
    """
    today_str = date.today().isoformat()
    if state.get("last_signal_date") == today_str:
        return False, 0.0, 0.0

    current = get_current_price()
    prev_close = get_prev_close()

    if current is None or prev_close is None:
        logger.warning("Could not fetch price data, skipping signal check")
        return False, 0.0, 0.0

    intraday_ret = (current - prev_close) / prev_close
    logger.info(
        f"  {TICKER}: current=${current:.2f}  prev_close=${prev_close:.2f}  "
        f"intraday={intraday_ret:+.2%}  threshold={SPIKE_THRESHOLD:+.0%}"
    )

    if intraday_ret >= SPIKE_THRESHOLD:
        return True, current, intraday_ret
    return False, current, intraday_ret


# ── Entry ─────────────────────────────────────────────────────────────────────
def enter_short(state: dict, entry_price: float, intraday_ret: float,
                position_size: float, live: bool) -> None:
    if state["position"] is not None:
        logger.info("  Already in a short position, skipping new entry")
        return

    today_str = date.today().isoformat()
    shares = position_size / entry_price

    state["position"] = {
        "entry_price":   entry_price,
        "entry_date":    today_str,
        "size_usd":      position_size,
        "shares":        round(shares, 6),
        "intraday_ret_at_entry": round(intraday_ret, 4),
        "mode":          "live" if live else "paper",
    }
    state["last_signal_date"] = today_str

    if live:
        # ── Robinhood MCP integration point ──────────────────────────────────
        # When the Robinhood MCP server is connected, replace this block with:
        #
        #   robinhood_mcp.place_order(
        #       symbol=TICKER,
        #       side="sell_short",
        #       quantity=shares,
        #       order_type="market",
        #   )
        #
        # Docs: robinhood.com/us/en/support/agentic-trading
        # Note: confirm short selling is supported in your agentic account type.
        # ─────────────────────────────────────────────────────────────────────
        raise NotImplementedError(
            "Live mode not yet enabled. Connect the Robinhood MCP server first."
        )
    else:
        logger.info(
            f"  PAPER SHORT {TICKER} | "
            f"entry=${entry_price:.2f}  shares={shares:.2f}  "
            f"notional=${position_size:.0f}  "
            f"intraday_spike={intraday_ret:+.2%}  "
            f"exit_target={HOLD_DAYS} trading days"
        )


# ── Exit ──────────────────────────────────────────────────────────────────────
def check_exit(state: dict, live: bool) -> None:
    pos = state["position"]
    if pos is None:
        return

    days_held = get_trading_days_since(pos["entry_date"])
    logger.info(f"  Open short: held {days_held}/{HOLD_DAYS} trading days")

    if days_held < HOLD_DAYS:
        return

    current = get_current_price()
    if current is None:
        logger.error("  Cannot fetch exit price — holding position")
        return

    entry_price = pos["entry_price"]
    size_usd    = pos["size_usd"]
    shares      = pos["shares"]

    # Short P&L: profit when price falls
    gross_return = (entry_price - current) / entry_price
    net_return   = gross_return - ROUND_TRIP_COST
    pnl_usd      = size_usd * net_return

    trade = {
        "entry_date":    pos["entry_date"],
        "exit_date":     date.today().isoformat(),
        "entry_price":   entry_price,
        "exit_price":    round(current, 4),
        "size_usd":      size_usd,
        "shares":        shares,
        "gross_return":  round(gross_return, 6),
        "net_return":    round(net_return, 6),
        "pnl_usd":       round(pnl_usd, 4),
        "spike_at_entry": pos["intraday_ret_at_entry"],
        "mode":          pos["mode"],
    }
    state["closed_trades"].append(trade)
    state["paper_pnl"] += pnl_usd
    state["position"] = None

    if live:
        # ── Robinhood MCP integration point ──────────────────────────────────
        # robinhood_mcp.place_order(
        #     symbol=TICKER,
        #     side="buy_to_cover",
        #     quantity=shares,
        #     order_type="market",
        # )
        # ─────────────────────────────────────────────────────────────────────
        raise NotImplementedError("Live mode not yet enabled.")
    else:
        outcome = "WIN  ✅" if net_return > 0 else "LOSS ❌"
        logger.info(
            f"  PAPER COVER {TICKER} | {outcome} | "
            f"entry=${entry_price:.2f}  cover=${current:.2f}  "
            f"return={net_return:+.2%}  P&L=${pnl_usd:+.2f}"
        )


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(state: dict) -> None:
    logger.info("─" * 60)
    pos = state["position"]
    if pos:
        current = get_current_price()
        entry = pos["entry_price"]
        days_held = get_trading_days_since(pos["entry_date"])
        unreal = ((entry - current) / entry - ROUND_TRIP_COST) if current else None
        logger.info(f"Open short: {TICKER}")
        logger.info(f"  Entry  : ${entry:.2f}  ({pos['entry_date']})")
        logger.info(f"  Current: ${current:.2f}" if current else "  Current: n/a")
        logger.info(f"  Held   : {days_held}/{HOLD_DAYS} trading days")
        if unreal is not None:
            logger.info(f"  Unreal : {unreal:+.2%}  (${pos['size_usd'] * unreal:+.2f})")
        logger.info(f"  Spike at entry: {pos['intraday_ret_at_entry']:+.2%}")
    else:
        logger.info("No open position")

    closed = state["closed_trades"]
    if closed:
        wins      = sum(1 for t in closed if t["net_return"] > 0)
        win_rate  = wins / len(closed)
        total_pnl = sum(t["pnl_usd"] for t in closed)
        avg_ret   = sum(t["net_return"] for t in closed) / len(closed)
        logger.info(
            f"Closed: {len(closed)} trades | "
            f"Win rate: {win_rate:.0%} | "
            f"Avg net: {avg_ret:+.2%} | "
            f"Total P&L: ${total_pnl:+.2f}"
        )
        logger.info("\nTrade history:")
        for t in closed:
            outcome = "W" if t["net_return"] > 0 else "L"
            logger.info(
                f"  [{outcome}] {t['entry_date']} → {t['exit_date']} | "
                f"${t['entry_price']:.2f} → ${t['exit_price']:.2f} | "
                f"{t['net_return']:+.2%} | ${t['pnl_usd']:+.2f}"
            )
    else:
        logger.info("No closed trades yet")
    logger.info("─" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LCID short mean-reversion paper trader")
    parser.add_argument("--live",    action="store_true", help="Live mode (Robinhood MCP required)")
    parser.add_argument("--summary", action="store_true", help="Print state only, no trading")
    parser.add_argument("--size",    type=float, default=POSITION_SIZE, help="Paper position size in USD")
    parser.add_argument("--reset",   action="store_true", help="Wipe state and start fresh")
    args = parser.parse_args()

    mode = "LIVE" if args.live else "PAPER"
    logger.info("=" * 60)
    logger.info(
        f"LCID Short Trader — {mode} | "
        f"{datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}"
    )
    logger.info("=" * 60)

    if args.reset:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
            logger.info("State wiped.")
        return

    state = load_state()

    if args.summary:
        print_summary(state)
        return

    if not is_market_hours():
        logger.info("Outside market hours (9:30–16:00 ET weekdays) — nothing to do")
        print_summary(state)
        save_state(state)
        return

    # Check exit first (always before checking for new entry)
    logger.info("Checking exit...")
    check_exit(state, live=args.live)

    # Check entry only if no position open
    if state["position"] is None:
        logger.info("Checking entry signal...")
        fired, price, intraday_ret = check_signal(state)
        if fired:
            logger.info(
                f"  SIGNAL FIRED: {TICKER} up {intraday_ret:+.2%} intraday "
                f"(threshold {SPIKE_THRESHOLD:+.0%})"
            )
            enter_short(state, price, intraday_ret, args.size, live=args.live)
        else:
            logger.info(f"  No signal (need +{SPIKE_THRESHOLD:.0%}, have {intraday_ret:+.2%})")
    else:
        logger.info("Position open — skipping entry check")

    save_state(state)
    print_summary(state)


if __name__ == "__main__":
    main()
