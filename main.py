"""
DCA vs Buy-the-Dip Strategy Comparison on NQ Futures
=====================================================
Both strategies receive $200/week.

DCA: Invests $200 into NQ every week.

Dip Strategy:
  - Default: $200/week goes into a HYSA earning the average of Fed Funds Rate
    and 3-Month T-Bill yield (floored at 0.25% annual).
  - Buy Signal: Invest ALL accumulated HYSA cash into NQ when EITHER:
      1) Distance from ATH >= 90th percentile of the rolling 3-year window
      2) Rolling 100-day standard deviation >= 90th percentile (rolling 3-year window)
  - StdDev signal can fire as soon as its rolling window is ready (no 3-year wait)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
WEEKLY_CONTRIBUTION = 200.0
ROLLING_WINDOW_YEARS = 3
PERCENTILE_THRESHOLD = 90


# ── Data Loading ────────────────────────────────────────────────────────────

def load_nq() -> pd.DataFrame:
    """Load NQ futures data, handling the multi-row header from Yahoo Finance."""
    df = pd.read_csv(DATA_DIR / "NQ_Futures_Historical_Data.csv", skiprows=[1, 2])
    df.rename(columns={"Price": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df[["Close"]].dropna()
    df.sort_index(inplace=True)
    return df


def load_rates() -> pd.DataFrame:
    """Load Fed Funds Rate and 3-Month T-Bill, merge, and compute HYSA rate."""
    dff = pd.read_csv(DATA_DIR / "DFF.csv", parse_dates=["observation_date"])
    dff.rename(columns={"observation_date": "Date", "DFF": "fed_funds"}, inplace=True)
    dff.set_index("Date", inplace=True)
    dff["fed_funds"] = pd.to_numeric(dff["fed_funds"], errors="coerce")

    dtb3 = pd.read_csv(DATA_DIR / "DTB3.csv", parse_dates=["observation_date"])
    dtb3.rename(columns={"observation_date": "Date", "DTB3": "tbill_3m"}, inplace=True)
    dtb3.set_index("Date", inplace=True)
    dtb3["tbill_3m"] = pd.to_numeric(dtb3["tbill_3m"], errors="coerce")

    rates = dff.join(dtb3, how="outer").sort_index()
    rates.ffill(inplace=True)  # forward-fill missing days (holidays etc.)

    # HYSA rate = average of fed funds and 3m T-Bill, floored at 0.25%
    rates["hysa_annual_pct"] = rates[["fed_funds", "tbill_3m"]].mean(axis=1)
    rates["hysa_annual_pct"] = rates["hysa_annual_pct"].clip(lower=0.25)

    # Daily rate for compounding: (1 + annual%)^(1/365) - 1
    rates["hysa_daily_rate"] = (1 + rates["hysa_annual_pct"] / 100) ** (1 / 365) - 1

    return rates[["hysa_annual_pct", "hysa_daily_rate"]]


# ── Signal Generation ───────────────────────────────────────────────────────

def compute_signals(nq: pd.DataFrame) -> pd.DataFrame:
    """
    For each trading day, compute:
      - rolling 3-year ATH
      - distance from ATH (%)
      - rolling 100-day close stddev
      - 90th percentile thresholds (rolling 3-year) for both metrics
      - buy signal (True/False)

    The ATH distance signal requires the full 3-year window for its percentile.
    The stddev signal can fire as soon as its own rolling window is ready.
    """
    df = nq.copy()
    trading_days_3y = 252 * ROLLING_WINDOW_YEARS  # ~756 trading days
    stddev_window = 100

    # Rolling 3-year ATH
    df["ath_3y"] = df["Close"].rolling(window=trading_days_3y, min_periods=trading_days_3y).max()

    # Distance from ATH (as a positive percentage, i.e. how far below ATH)
    df["dist_from_ath"] = (df["ath_3y"] - df["Close"]) / df["ath_3y"] * 100

    # Rolling 200-day standard deviation of close prices
    df["stddev_100d"] = df["Close"].rolling(window=stddev_window, min_periods=stddev_window).std()

    # 90th percentile for ATH distance (top bracket = big drawdowns)
    df["dist_p90"] = df["dist_from_ath"].rolling(window=trading_days_3y, min_periods=trading_days_3y).quantile(
        PERCENTILE_THRESHOLD / 100
    )
    # 10th percentile for stddev (bottom bracket = low volatility compression)
    df["stddev_p10"] = df["stddev_100d"].rolling(window=trading_days_3y, min_periods=trading_days_3y).quantile(
        (100 - PERCENTILE_THRESHOLD) / 100
    )

    # ATH distance signal: only valid once 3-year window is ready
    ath_signal = df["dist_from_ath"] >= df["dist_p90"]
    ath_signal = ath_signal.fillna(False)

    # StdDev signal: buy when volatility is in the bottom 10% (compressed)
    # Before 3-year percentile is available, use expanding percentile
    df["stddev_p10_early"] = df["stddev_100d"].expanding(min_periods=stddev_window).quantile(
        (100 - PERCENTILE_THRESHOLD) / 100
    )
    # Use 3-year rolling threshold when available, otherwise expanding
    stddev_threshold = df["stddev_p10"].fillna(df["stddev_p10_early"])
    stddev_signal = df["stddev_100d"] <= stddev_threshold
    stddev_signal = stddev_signal.fillna(False)

    # Sell signal: stddev > 99th percentile AND close to ATH (≤ 30th percentile distance)
    # Only sell near highs — don't panic sell during a crash when already deep in drawdown
    df["stddev_p98"] = df["stddev_100d"].rolling(window=trading_days_3y, min_periods=trading_days_3y).quantile(0.98)
    df["stddev_p98_early"] = df["stddev_100d"].expanding(min_periods=stddev_window).quantile(0.98)
    sell_threshold = df["stddev_p98"].fillna(df["stddev_p98_early"])
    high_vol = (df["stddev_100d"] > sell_threshold).fillna(False)

    # ATH proximity: distance from ATH must be in the bottom 30% (i.e. close to ATH)
    df["dist_p30"] = df["dist_from_ath"].rolling(window=trading_days_3y, min_periods=trading_days_3y).quantile(0.30)
    df["dist_p30_early"] = df["dist_from_ath"].expanding(min_periods=50).quantile(0.30)
    dist_sell_threshold = df["dist_p30"].fillna(df["dist_p30_early"])
    near_ath = (df["dist_from_ath"] <= dist_sell_threshold).fillna(False)

    df["sell_signal"] = high_vol & near_ath

    # Buy signal: either condition met (but not if sell signal also active)
    df["buy_signal"] = (ath_signal | stddev_signal) & ~df["sell_signal"]

    # Keep rows from when the first signal can possibly fire (stddev window ready)
    return df.dropna(subset=["stddev_100d"])


# ── Simulation ──────────────────────────────────────────────────────────────

def simulate(signals: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate both strategies on a weekly basis (every Monday, or first trading day of the week).
    Returns a DataFrame indexed by contribution week with portfolio values.
    """
    # Align rates to NQ trading days
    rates_aligned = rates.reindex(signals.index, method="ffill")
    signals = signals.join(rates_aligned)

    # Identify weekly contribution dates (first trading day each week)
    signals["year_week"] = signals.index.to_period("W")
    weekly_mask = ~signals["year_week"].duplicated(keep="first")
    contribution_dates = signals.index[weekly_mask]

    # ── DCA state ───
    dca_shares = 0.0
    dca_total_invested = 0.0

    # ── Dip strategy state ───
    dip_shares = 0.0
    dip_hysa_balance = 0.0
    dip_total_invested = 0.0
    dip_last_date = None  # track last processed date for daily HYSA compounding

    # Results storage
    records = []

    for date in signals.index:
        row = signals.loc[date]
        price = row["Close"]
        daily_rate = row["hysa_daily_rate"] if not np.isnan(row["hysa_daily_rate"]) else 0.0

        is_contribution_day = date in contribution_dates

        # ── DCA ─────────────────────────────────────────────────────────
        if is_contribution_day:
            shares_bought = WEEKLY_CONTRIBUTION / price
            dca_shares += shares_bought
            dca_total_invested += WEEKLY_CONTRIBUTION

        dca_portfolio_value = dca_shares * price

        # ── Dip Strategy ────────────────────────────────────────────────
        # Compound HYSA daily
        dip_hysa_balance *= (1 + daily_rate)

        # Add weekly contribution to HYSA
        if is_contribution_day:
            dip_hysa_balance += WEEKLY_CONTRIBUTION
            dip_total_invested += WEEKLY_CONTRIBUTION

        # Check sell signal — liquidate ALL equity to HYSA
        if row["sell_signal"] and dip_shares > 0:
            dip_hysa_balance += dip_shares * price
            dip_shares = 0.0

        # Check buy signal — invest ALL HYSA cash
        elif row["buy_signal"] and dip_hysa_balance > 0:
            shares_bought = dip_hysa_balance / price
            dip_shares += shares_bought
            dip_hysa_balance = 0.0

        dip_equity_value = dip_shares * price
        dip_total_value = dip_equity_value + dip_hysa_balance

        # Record on contribution days for cleaner output (daily for accuracy)
        if is_contribution_day:
            records.append({
                "Date": date,
                "Price": price,
                "Buy_Signal": row["buy_signal"],
                "Sell_Signal": row["sell_signal"],
                "DCA_Shares": dca_shares,
                "DCA_Portfolio": dca_portfolio_value,
                "DCA_Total_Invested": dca_total_invested,
                "Dip_Shares": dip_shares,
                "Dip_HYSA_Balance": dip_hysa_balance,
                "Dip_Equity": dip_equity_value,
                "Dip_Total_Value": dip_total_value,
                "Dip_Total_Invested": dip_total_invested,
                "HYSA_Rate_Annual": row["hysa_annual_pct"],
                "Dist_From_ATH_Pct": row["dist_from_ath"],
                "StdDev_100d": row["stddev_100d"],
            })

    results = pd.DataFrame(records)
    results.set_index("Date", inplace=True)
    return results


# ── Reporting ───────────────────────────────────────────────────────────────

def print_summary(results: pd.DataFrame) -> None:
    """Print key summary statistics including Sharpe ratios."""
    final = results.iloc[-1]
    first = results.iloc[0]

    total_weeks = len(results)
    total_invested = final["DCA_Total_Invested"]

    dca_final = final["DCA_Portfolio"]
    dip_final = final["Dip_Total_Value"]

    dca_return = (dca_final - total_invested) / total_invested * 100
    dip_return = (dip_final - total_invested) / total_invested * 100

    years = (results.index[-1] - results.index[0]).days / 365.25
    dca_cagr = ((dca_final / total_invested) ** (1 / years) - 1) * 100 if years > 0 else 0
    dip_cagr = ((dip_final / total_invested) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Count buy signals
    buy_signal_weeks = results["Buy_Signal"].sum()

    # ── Sharpe Ratio ──
    # Weekly returns for each strategy
    dca_weekly_returns = results["DCA_Portfolio"].pct_change().dropna()
    dip_weekly_returns = results["Dip_Total_Value"].pct_change().dropna()

    # Weekly risk-free rate from HYSA (annual % → weekly decimal)
    rf_weekly = results["HYSA_Rate_Annual"].iloc[1:] / 100 / 52
    rf_weekly = rf_weekly.reindex(dca_weekly_returns.index, method="ffill").fillna(0)

    # Excess returns
    dca_excess = dca_weekly_returns - rf_weekly
    dip_excess = dip_weekly_returns - rf_weekly

    # Annualized Sharpe = (mean excess weekly return / std weekly return) * sqrt(52)
    dca_sharpe = (dca_excess.mean() / dca_excess.std() * np.sqrt(52)) if dca_excess.std() > 0 else 0
    dip_sharpe = (dip_excess.mean() / dip_excess.std() * np.sqrt(52)) if dip_excess.std() > 0 else 0

    # ── Drawdown Analysis ──
    def compute_drawdown_stats(portfolio_series):
        """Compute drawdown stats from a portfolio value series."""
        running_max = portfolio_series.cummax()
        drawdown = (running_max - portfolio_series) / running_max * 100  # as %
        max_dd = drawdown.max()
        avg_dd = drawdown.mean()

        # Drawdown duration: count consecutive weeks in drawdown, avg those durations
        in_dd = drawdown > 0
        # Group consecutive drawdown periods
        groups = (~in_dd).cumsum()
        dd_durations = []
        for _, group in in_dd.groupby(groups):
            if group.any():  # this group is in drawdown
                dd_durations.append(len(group))
        avg_dd_duration = np.mean(dd_durations) if dd_durations else 0
        max_dd_duration = max(dd_durations) if dd_durations else 0
        return avg_dd, max_dd, avg_dd_duration, max_dd_duration

    dca_avg_dd, dca_max_dd, dca_avg_dd_dur, dca_max_dd_dur = compute_drawdown_stats(results["DCA_Portfolio"])
    dip_avg_dd, dip_max_dd, dip_avg_dd_dur, dip_max_dd_dur = compute_drawdown_stats(results["Dip_Total_Value"])

    print("=" * 70)
    print("DCA vs BUY-THE-DIP STRATEGY COMPARISON")
    print("=" * 70)
    print(f"Period: {first.name.date()} → {final.name.date()} ({years:.1f} years)")
    print(f"Total Weeks: {total_weeks}")
    print(f"Total Invested: ${total_invested:,.2f}")
    print(f"Weekly Contribution: ${WEEKLY_CONTRIBUTION:,.2f}")
    print()
    print("─── DCA Strategy ───")
    print(f"  Final Portfolio Value:  ${dca_final:,.2f}")
    print(f"  Total Return:           {dca_return:,.2f}%")
    print(f"  CAGR:                   {dca_cagr:.2f}%")
    print(f"  Sharpe Ratio:           {dca_sharpe:.3f}")
    print(f"  Avg Drawdown:           {dca_avg_dd:.2f}%")
    print(f"  Max Drawdown:           {dca_max_dd:.2f}%")
    print(f"  Avg DD Duration:        {dca_avg_dd_dur:.1f} weeks")
    print(f"  Max DD Duration:        {dca_max_dd_dur} weeks")
    print(f"  Shares Accumulated:     {final['DCA_Shares']:,.6f}")
    print()
    print("─── Buy-the-Dip Strategy ───")
    print(f"  Final Equity Value:     ${final['Dip_Equity']:,.2f}")
    print(f"  Final HYSA Balance:     ${final['Dip_HYSA_Balance']:,.2f}")
    print(f"  Final Total Value:      ${dip_final:,.2f}")
    print(f"  Total Return:           {dip_return:,.2f}%")
    print(f"  CAGR:                   {dip_cagr:.2f}%")
    print(f"  Sharpe Ratio:           {dip_sharpe:.3f}")
    print(f"  Avg Drawdown:           {dip_avg_dd:.2f}%")
    print(f"  Max Drawdown:           {dip_max_dd:.2f}%")
    print(f"  Avg DD Duration:        {dip_avg_dd_dur:.1f} weeks")
    print(f"  Max DD Duration:        {dip_max_dd_dur} weeks")
    print(f"  Shares Accumulated:     {final['Dip_Shares']:,.6f}")
    print(f"  Buy Signal Weeks:       {buy_signal_weeks} / {total_weeks} ({buy_signal_weeks/total_weeks*100:.1f}%)")
    print()
    print("─── Comparison ───")
    diff = dip_final - dca_final
    winner = "Dip" if diff > 0 else "DCA"
    print(f"  Difference:             ${abs(diff):,.2f} in favor of {winner}")
    print(f"  Dip/DCA Ratio:          {dip_final/dca_final:.4f}")
    sharpe_winner = "Dip" if dip_sharpe > dca_sharpe else "DCA"
    print(f"  Sharpe:  DCA={dca_sharpe:.3f}  Dip={dip_sharpe:.3f}  → {sharpe_winner}")
    dd_winner = "Dip" if dip_avg_dd < dca_avg_dd else "DCA"
    print(f"  Avg DD:  DCA={dca_avg_dd:.2f}%  Dip={dip_avg_dd:.2f}%  → {dd_winner}")
    dd_max_winner = "Dip" if dip_max_dd < dca_max_dd else "DCA"
    print(f"  Max DD:  DCA={dca_max_dd:.2f}%  Dip={dip_max_dd:.2f}%  → {dd_max_winner}")
    print("=" * 70)


def plot_results(results: pd.DataFrame, signals: pd.DataFrame) -> None:
    """Generate comparison charts."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
    fig.suptitle("DCA vs Buy-the-Dip Strategy Comparison (NQ Futures)", fontsize=16, fontweight="bold")

    # 1) Portfolio Value
    ax = axes[0]
    ax.plot(results.index, results["DCA_Portfolio"], label="DCA Portfolio", linewidth=1.5, color="#2196F3")
    ax.plot(results.index, results["Dip_Total_Value"], label="Dip Total Value", linewidth=1.5, color="#FF9800")
    ax.plot(results.index, results["DCA_Total_Invested"], label="Total Invested", linewidth=1, linestyle="--", color="gray", alpha=0.7)
    ax.set_ylabel("Value ($)")
    ax.set_title("Portfolio Value Over Time")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # 2) NQ Price with buy signals marked
    ax = axes[1]
    ax.plot(signals.index, signals["Close"], color="#555", linewidth=0.8, label="NQ Close")
    buy_dates = signals[signals["buy_signal"]].index
    ax.scatter(buy_dates, signals.loc[buy_dates, "Close"], color="red", s=3, alpha=0.5, label="Buy Signals", zorder=5)
    ax.set_ylabel("NQ Price")
    ax.set_title("NQ Futures Price & Buy Signals")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # 3) Distance from ATH & threshold
    ax = axes[2]
    ax.plot(signals.index, signals["dist_from_ath"], label="Dist from ATH (%)", linewidth=0.8, color="#E91E63")
    ax.plot(signals.index, signals["dist_p90"], label="90th Percentile", linewidth=0.8, linestyle="--", color="#9C27B0")
    ax.set_ylabel("Distance from ATH (%)")
    ax.set_title("Distance from Rolling 3-Year ATH")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # 4) Rolling 200-day StdDev & threshold
    ax = axes[3]
    ax.plot(signals.index, signals["stddev_100d"], label="100-day StdDev", linewidth=0.8, color="#4CAF50")
    # Show effective thresholds (3-year rolling when available, expanding otherwise)
    stddev_thresh_low = signals["stddev_p10"].fillna(signals["stddev_p10_early"])
    ax.plot(signals.index, stddev_thresh_low, label="10th Percentile (Buy)", linewidth=0.8, linestyle="--", color="#009688")
    stddev_thresh_high = signals["stddev_p98"].fillna(signals["stddev_p98_early"])
    ax.plot(signals.index, stddev_thresh_high, label="98th Percentile (Sell)", linewidth=0.8, linestyle="--", color="#F44336")
    ax.set_ylabel("StdDev")
    ax.set_title("Rolling 100-Day Standard Deviation")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Date")

    plt.tight_layout()
    output_path = Path(__file__).parent / "dca_vs_dip_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {output_path}")
    plt.show()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    nq = load_nq()
    rates = load_rates()

    print(f"NQ data: {nq.index[0].date()} → {nq.index[-1].date()} ({len(nq)} trading days)")

    print("Computing signals...")
    signals = compute_signals(nq)
    print(f"Signal period: {signals.index[0].date()} → {signals.index[-1].date()} ({len(signals)} days)")

    total_days = len(signals)
    signal_days = signals["buy_signal"].sum()
    print(f"Buy signal days: {signal_days} / {total_days} ({signal_days/total_days*100:.1f}%)")

    print("Running simulation...")
    results = simulate(signals, rates)

    print_summary(results)
    plot_results(results, signals)

    # Save detailed results
    output_csv = Path(__file__).parent / "dca_vs_dip_results.csv"
    results.to_csv(output_csv)
    print(f"Detailed results saved to: {output_csv}")


if __name__ == "__main__":
    main()
