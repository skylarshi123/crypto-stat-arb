"""
Lucid Group (LCID) mean-reversion backtest.

Hypothesis: when LCID drops ~10% within a short window, the price reverts
over the following 24–72 hours. Tests multiple drop-detection windows
and forward holding periods to find where (if anywhere) the edge is real.

Data: Yahoo Finance daily OHLCV, last 12 months.
Entry: close of the signal day (conservative — you've seen the full-day drop).
Exit:  close N calendar days later (1d, 2d, 3d, 5d, 10d).
Costs: 0 bps (retail market; add your own if using limit orders).
"""

import sys
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ── Parameters ───────────────────────────────────────────────────────────────
TICKER          = "LCID"
LOOKBACK_YEARS  = 1          # how far back to pull data
DROP_WINDOWS    = [1, 2, 3]  # rolling calendar days over which drop is measured
DROP_THRESHOLDS = [-0.08, -0.10, -0.12, -0.15]  # e.g. -0.10 = down ≥10%
HOLD_PERIODS    = [1, 2, 3, 5, 10]              # forward calendar days held
TRANSACTION_COST = 0.0020    # 20 bps round-trip (retail equity, conservative)

OUT_DIR = Path("results/lucid")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch_data(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, period=f"{LOOKBACK_YEARS}y", interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    df.dropna(subset=["Close"], inplace=True)
    return df


# ── Signal detection ──────────────────────────────────────────────────────────
def compute_signals(df: pd.DataFrame, window: int, threshold: float) -> pd.Series:
    """
    Return boolean Series where True = signal day.
    Rolling return = (today close / close N days ago) - 1.
    """
    roll_ret = df["Close"].pct_change(periods=window)
    return roll_ret <= threshold


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, signal_mask: pd.Series,
                 hold_days: int) -> dict:
    closes = df["Close"]
    entries = df.index[signal_mask].tolist()

    trades = []
    for entry_date in entries:
        # find the close price N trading/calendar days later
        future_closes = closes[closes.index > entry_date]
        if len(future_closes) < hold_days:
            continue
        exit_close = future_closes.iloc[hold_days - 1]
        entry_close = closes[entry_date]
        gross_return = (exit_close / entry_close) - 1.0
        net_return = gross_return - TRANSACTION_COST
        trades.append({
            "entry_date": entry_date,
            "entry_price": round(float(entry_close), 4),
            "exit_price":  round(float(exit_close), 4),
            "hold_days":   hold_days,
            "gross_return": round(gross_return, 6),
            "net_return":   round(net_return, 6),
            "win": net_return > 0,
        })

    if not trades:
        return {"n_trades": 0}

    df_t = pd.DataFrame(trades)
    n = len(df_t)
    win_rate = df_t["win"].mean()
    avg_net  = df_t["net_return"].mean()
    med_net  = df_t["net_return"].median()
    std_net  = df_t["net_return"].std()
    sharpe   = (avg_net / std_net * np.sqrt(252 / hold_days)) if std_net > 0 else np.nan
    worst    = df_t["net_return"].min()
    best     = df_t["net_return"].max()

    return {
        "n_trades":   n,
        "win_rate":   round(win_rate, 4),
        "avg_net":    round(avg_net, 4),
        "median_net": round(med_net, 4),
        "std_net":    round(std_net, 4),
        "annualized_sharpe": round(sharpe, 3) if not np.isnan(sharpe) else None,
        "best":       round(best, 4),
        "worst":      round(worst, 4),
        "trades":     df_t,
    }


# ── Summary table ─────────────────────────────────────────────────────────────
def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in DROP_WINDOWS:
        for threshold in DROP_THRESHOLDS:
            signal_mask = compute_signals(df, window, threshold)
            n_signals = int(signal_mask.sum())
            for hold in HOLD_PERIODS:
                res = run_backtest(df, signal_mask, hold)
                rows.append({
                    "drop_window_days": window,
                    "drop_threshold":   f"{threshold*100:.0f}%",
                    "hold_days":        hold,
                    "n_signals":        n_signals,
                    "n_trades":         res.get("n_trades", 0),
                    "win_rate":         res.get("win_rate"),
                    "avg_net_ret":      res.get("avg_net"),
                    "median_net_ret":   res.get("median_net"),
                    "sharpe":           res.get("annualized_sharpe"),
                    "worst":            res.get("worst"),
                    "best":             res.get("best"),
                })
    return pd.DataFrame(rows)


# ── Charts ────────────────────────────────────────────────────────────────────
def plot_price_with_signals(df: pd.DataFrame, window: int, threshold: float):
    signal_mask = compute_signals(df, window, threshold)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(df.index, df["Close"], color="#1f77b4", linewidth=1.2, label="LCID Close")
    signal_dates = df.index[signal_mask]
    signal_prices = df.loc[signal_mask, "Close"]
    ax1.scatter(signal_dates, signal_prices, color="red", zorder=5, s=60,
                label=f"Signal: {window}d drop ≤ {threshold*100:.0f}%  (n={len(signal_dates)})")
    ax1.set_ylabel("Price (USD)")
    ax1.set_title(f"LCID — {window}-day drop ≤ {threshold*100:.0f}% signals (last {LOOKBACK_YEARS}y)")
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    ax2 = axes[1]
    roll_ret = df["Close"].pct_change(periods=window) * 100
    ax2.bar(df.index, roll_ret, color=["#d62728" if v <= threshold*100 else "#aec7e8"
                                        for v in roll_ret.fillna(0)], width=1)
    ax2.axhline(threshold * 100, color="red", linewidth=1, linestyle="--")
    ax2.set_ylabel(f"{window}-day return (%)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    fig.tight_layout()
    path = OUT_DIR / f"signals_w{window}_t{abs(int(threshold*100))}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_forward_returns(summary: pd.DataFrame, window: int, threshold: float):
    sub = summary[(summary["drop_window_days"] == window) &
                  (summary["drop_threshold"] == f"{threshold*100:.0f}%")].copy()
    if sub.empty or sub["n_trades"].max() == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"LCID forward returns | {window}-day drop ≤ {threshold*100:.0f}%")

    # Win rate by hold period
    axes[0].bar(sub["hold_days"], sub["win_rate"] * 100, color="#2ca02c")
    axes[0].axhline(50, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Hold days")
    axes[0].set_ylabel("Win rate (%)")
    axes[0].set_title("Win Rate")

    # Avg net return
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in sub["avg_net_ret"]]
    axes[1].bar(sub["hold_days"], sub["avg_net_ret"] * 100, color=colors)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Hold days")
    axes[1].set_ylabel("Avg net return (%)")
    axes[1].set_title("Average Net Return")

    # Annualized Sharpe
    sharpe_vals = sub["sharpe"].fillna(0)
    axes[2].bar(sub["hold_days"], sharpe_vals, color="#ff7f0e")
    axes[2].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Hold days")
    axes[2].set_ylabel("Ann. Sharpe")
    axes[2].set_title("Annualized Sharpe")

    fig.tight_layout()
    path = OUT_DIR / f"forward_returns_w{window}_t{abs(int(threshold*100))}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_trade_distribution(df_data: pd.DataFrame, window: int,
                            threshold: float, hold_days: int):
    signal_mask = compute_signals(df_data, window, threshold)
    res = run_backtest(df_data, signal_mask, hold_days)
    if res.get("n_trades", 0) < 3:
        return
    trades = res["trades"]

    fig, ax = plt.subplots(figsize=(9, 5))
    returns_pct = trades["net_return"] * 100
    ax.hist(returns_pct, bins=20, color="#1f77b4", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--", label="Break-even")
    ax.axvline(returns_pct.mean(), color="green", linewidth=1.5,
               linestyle="-", label=f"Mean = {returns_pct.mean():.1f}%")
    ax.set_xlabel("Net return (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"LCID trade distribution | {window}d drop ≤ {threshold*100:.0f}%, hold {hold_days}d  (n={len(trades)})")
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / f"distribution_w{window}_t{abs(int(threshold*100))}_h{hold_days}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  LCID Mean-Reversion Backtest")
    print(f"{'='*60}\n")

    print("Fetching LCID data...")
    df = fetch_data(TICKER)
    print(f"  {len(df)} trading days  ({df.index[0].date()} → {df.index[-1].date()})")
    print(f"  Price range: ${df['Close'].min():.2f} – ${df['Close'].max():.2f}")

    # Show recent big-drop days for reference
    daily_ret = df["Close"].pct_change()
    big_drops = df[daily_ret <= -0.08][["Close"]].copy()
    big_drops["1d_return_%"] = (daily_ret[big_drops.index] * 100).round(2)
    print(f"\n  Days with ≥8% single-day drop: {len(big_drops)}")
    print(big_drops.tail(20).to_string())

    print("\nBuilding summary grid...")
    summary = build_summary(df)
    summary_path = OUT_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Full grid saved → {summary_path}")

    # Print the most signal-rich combos
    print("\n── Top combinations by win rate (≥3 trades) ──────────────────")
    filtered = summary[summary["n_trades"] >= 3].sort_values(
        "win_rate", ascending=False
    )
    cols = ["drop_window_days", "drop_threshold", "hold_days",
            "n_trades", "win_rate", "avg_net_ret", "sharpe"]
    print(filtered[cols].head(15).to_string(index=False))

    print("\n── Best average net return (≥3 trades) ───────────────────────")
    filtered2 = summary[summary["n_trades"] >= 3].sort_values(
        "avg_net_ret", ascending=False
    )
    print(filtered2[cols].head(15).to_string(index=False))

    # Focus on the most likely Robinhood-alert scenario: 1-day 10% drop
    print("\n── 1-day ≥10% drop: all hold periods ─────────────────────────")
    focus = summary[
        (summary["drop_window_days"] == 1) &
        (summary["drop_threshold"] == "-10%")
    ][cols + ["median_net_ret", "worst", "best"]]
    print(focus.to_string(index=False))

    print("\nGenerating charts...")
    # Price chart + signals for the Robinhood-like scenario
    plot_price_with_signals(df, window=1, threshold=-0.10)
    # Forward return bars for 1d/10% and 2d/10%
    for w in [1, 2]:
        plot_forward_returns(summary, window=w, threshold=-0.10)
    # Trade distribution for the most natural hold periods
    for h in [1, 2, 3, 5]:
        plot_trade_distribution(df, window=1, threshold=-0.10, hold_days=h)

    print(f"\nAll outputs in: {OUT_DIR.resolve()}/")
    print("Done.\n")


if __name__ == "__main__":
    main()
