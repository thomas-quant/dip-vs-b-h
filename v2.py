"""
V2: Cash-Secured Put Selling + Covered Calls vs DCA on NQ Futures
=================================================================
Both strategies receive $200/week.

DCA: Invests $200 into NQ every week (same as v1).

Put-Selling + Covered Call Strategy:
  - Cash accumulated ($200/week + premiums) sits in HYSA.
  - Each week, sell a cash-secured put (7 DTE) on NQ.
  - Strike depth depends on signal state:
      Buy signal       -> 0.5 x stddev OTM (near ATM, want assignment)
      Re-entry mode    -> 0.5 x stddev OTM (aggressive re-buy after call assignment)
      Normal           -> 2.0 x stddev OTM (moderate, collect premium)
      Sell signal      -> skip (no new exposure in extreme vol)
  - Re-entry mode activates after call assignment and stays active
    until shares are re-acquired, as long as dist_from_ath is within
    the 60th percentile of its 100-day rolling window.
  - When holding shares, sell covered calls:
      Aggressive (sell signal: p98 stddev + near ATH) -> 0.30 delta
      Non-aggressive (near ATH, <= p30 distance)      -> 0.10 delta
  - Options priced via Black-Scholes using 100-day realized vol.
  - If put assigned (close < strike), buy NQ at strike.
  - If call assigned (close > strike), shares called away -> cash.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path

# Import shared data loading & signal generation from v1
import sys
sys.path.insert(0, str(Path(__file__).parent))
from main import load_nq, load_rates, compute_signals, WEEKLY_CONTRIBUTION, DATA_DIR

ROLLING_WINDOW_YEARS = 3
PERCENTILE_THRESHOLD = 90

# Strike depth multipliers (x stddev_100d below spot)
STRIKE_MULT_BUY_SIGNAL = 0.5    # near ATM when buy signal fires
STRIKE_MULT_REENTRY = 0.5       # near ATM when re-entering after call assignment
STRIKE_MULT_NORMAL = 2.0        # moderate OTM normally
PUT_DTE = 7                     # weekly puts
CALL_DTE = 7                    # weekly calls
CALL_DELTA_AGGRESSIVE = 0.30    # closer to ATM when sell signal (want to reduce exposure)
CALL_DELTA_PASSIVE = 0.10       # far OTM when near ATH (collect premium, keep upside)
REENTRY_DIST_PERCENTILE = 0.60  # re-entry allowed when dist_from_ath <= 60th pctl (100d)


# -- Black-Scholes ---------------------------------------------------------------

def _bs_d1d2(S: float, K: float, T: float, r: float, sigma: float):
    """Compute d1, d2 for Black-Scholes."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1, d2 = _bs_d1d2(S, K, T, r, sigma)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(put, 0.0)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1, d2 = _bs_d1d2(S, K, T, r, sigma)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return max(call, 0.0)


def strike_from_call_delta(S: float, T: float, r: float, sigma: float, target_delta: float) -> float:
    """
    Find the OTM call strike for a given BS delta target.
    Call delta = N(d1), so d1 = N^-1(target_delta).
    Then solve for K from the d1 equation.
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return S * 1.5  # fallback far OTM
    d1_target = norm.ppf(target_delta)
    # d1 = (ln(S/K) + (r + s^2/2)T) / (s*sqrt(T))  ->  ln(S/K) = d1*s*sqrt(T) - (r + s^2/2)*T
    ln_s_over_k = d1_target * sigma * np.sqrt(T) - (r + 0.5 * sigma ** 2) * T
    K = S * np.exp(-ln_s_over_k)
    return max(K, S * 1.001)  # ensure OTM (K > S)


def compute_realized_vol(signals: pd.DataFrame, window: int = 100) -> pd.Series:
    """
    Annualized realized volatility from daily log returns over a rolling window.
    """
    log_returns = np.log(signals["Close"] / signals["Close"].shift(1))
    return log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)


# -- Simulation -------------------------------------------------------------------

def simulate_v2(signals: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate both DCA and put-selling + covered call strategies on a weekly basis.
    Returns a DataFrame indexed by contribution week with portfolio values.
    """
    # Merge rates into signals
    signals = signals.join(rates, how="left")
    signals["hysa_daily_rate"] = signals["hysa_daily_rate"].ffill().fillna(0)
    signals["hysa_annual_pct"] = signals["hysa_annual_pct"].ffill().fillna(0.25)

    # Compute annualized realized vol
    signals["realized_vol"] = compute_realized_vol(signals)
    signals["realized_vol"] = signals["realized_vol"].ffill().fillna(0.15)  # default 15%

    # 100-day rolling 60th percentile of dist_from_ath for re-entry gating
    signals["dist_p60_100d"] = signals["dist_from_ath"].rolling(
        window=100, min_periods=50
    ).quantile(REENTRY_DIST_PERCENTILE)
    signals["dist_p60_100d"] = signals["dist_p60_100d"].ffill().fillna(float("inf"))

    # Weekly contribution dates: every Monday (or first trading day of the week)
    signals["week"] = signals.index.to_period("W")
    first_of_week = signals.groupby("week").head(1).index
    contribution_dates = set(first_of_week)

    # -- DCA state ---
    dca_shares = 0.0
    dca_total_invested = 0.0

    # -- Put-selling + covered call state ---
    put_cash = 0.0              # HYSA cash balance
    put_shares = 0.0            # assigned NQ shares held
    put_total_invested = 0.0    # total $ contributed
    total_put_premium = 0.0     # lifetime put premium
    total_call_premium = 0.0    # lifetime call premium
    put_assignment_count = 0    # times assigned on puts
    call_assignment_count = 0   # times shares called away
    recently_called_away = False  # flag: aggressive re-entry after call assignment

    # Active put contract (if any)
    active_put_strike = None
    active_put_notional = 0.0   # fractional contracts (cash / strike)
    active_put_expiry = None

    # Active covered call (if any)
    active_call_strike = None
    active_call_shares = 0.0    # shares covered by the call
    active_call_expiry = None

    records = []

    for date in signals.index:
        row = signals.loc[date]
        price = row["Close"]
        daily_rate = row["hysa_daily_rate"] if not np.isnan(row["hysa_daily_rate"]) else 0.0
        is_contribution_day = date in contribution_dates

        # -- DCA ---------------------------------------------------------
        if is_contribution_day:
            shares_bought = WEEKLY_CONTRIBUTION / price
            dca_shares += shares_bought
            dca_total_invested += WEEKLY_CONTRIBUTION

        dca_portfolio_value = dca_shares * price

        # -- Put-Selling + Covered Call Strategy --------------------------
        # Compound HYSA daily
        put_cash *= (1 + daily_rate)

        # Add weekly contribution
        if is_contribution_day:
            put_cash += WEEKLY_CONTRIBUTION
            put_total_invested += WEEKLY_CONTRIBUTION

        # Common vol/rate values for options pricing
        sigma = row["realized_vol"] if not np.isnan(row["realized_vol"]) else 0.15
        stddev = row["stddev_100d"] if not np.isnan(row["stddev_100d"]) else 0.0
        rf = row["hysa_annual_pct"] / 100 if not np.isnan(row["hysa_annual_pct"]) else 0.0025

        # -- Check put expiry (weekly) --
        if is_contribution_day and active_put_strike is not None:
            if price < active_put_strike:
                # Assigned: buy at strike
                cost = active_put_notional * active_put_strike
                if cost <= put_cash:
                    put_shares += active_put_notional
                    put_cash -= cost
                    put_assignment_count += 1
                else:
                    # Partial assignment: use all available cash
                    affordable = put_cash / active_put_strike
                    put_shares += affordable
                    put_cash = 0.0
                    put_assignment_count += 1
                recently_called_away = False  # got shares back, exit re-entry mode
            active_put_strike = None
            active_put_notional = 0.0
            active_put_expiry = None

        # -- Check call expiry (weekly) --
        if is_contribution_day and active_call_strike is not None:
            if price > active_call_strike:
                # Shares called away: sell at strike
                proceeds = active_call_shares * active_call_strike
                put_shares -= active_call_shares
                put_cash += proceeds
                call_assignment_count += 1
                recently_called_away = True  # enter re-entry mode
            active_call_strike = None
            active_call_shares = 0.0
            active_call_expiry = None

        # -- Sell new put (unless sell signal) --
        # Re-entry check: after call assignment, sell aggressive puts
        # as long as dist_from_ath is within the 60th percentile (100d window)
        dist_val = row["dist_from_ath"] if not np.isnan(row["dist_from_ath"]) else float("inf")
        dist_p60 = row["dist_p60_100d"] if not np.isnan(row["dist_p60_100d"]) else float("inf")
        want_reentry = recently_called_away and (dist_val <= dist_p60) and (put_shares == 0)

        if is_contribution_day and put_cash > 0:
            if row["sell_signal"] and not want_reentry:
                pass  # Skip: don't sell puts in extreme vol near ATH
            else:
                if want_reentry:
                    # Aggressive re-entry: near ATM puts to get back in fast
                    strike = max(price - STRIKE_MULT_REENTRY * stddev, price * 0.5)
                elif row["buy_signal"]:
                    strike = max(price - STRIKE_MULT_BUY_SIGNAL * stddev, price * 0.5)
                else:
                    strike = max(price - STRIKE_MULT_NORMAL * stddev, price * 0.5)

                notional = put_cash / strike
                T = PUT_DTE / 365.0
                premium_per_unit = bs_put_price(price, strike, T, rf, sigma)
                premium_collected = premium_per_unit * notional

                put_cash += premium_collected
                total_put_premium += premium_collected

                active_put_strike = strike
                active_put_notional = notional
                active_put_expiry = date + pd.Timedelta(days=PUT_DTE)

        # -- Sell covered call (if holding shares and no active call) --
        if is_contribution_day and put_shares > 0 and active_call_strike is None:
            T_call = CALL_DTE / 365.0

            # Check if near ATH (dist_from_ath <= 30th percentile)
            dist_val = row["dist_from_ath"] if not np.isnan(row["dist_from_ath"]) else float("inf")
            dist_p30 = row.get("dist_p30", float("inf"))
            if np.isnan(dist_p30):
                dist_p30 = float("inf")
            near_ath = dist_val <= dist_p30

            if row["sell_signal"]:
                # Aggressive: closer to ATM, want shares called away
                call_strike = strike_from_call_delta(price, T_call, rf, sigma, CALL_DELTA_AGGRESSIVE)
            elif near_ath:
                # Non-aggressive: far OTM, collect premium, keep upside
                call_strike = strike_from_call_delta(price, T_call, rf, sigma, CALL_DELTA_PASSIVE)
            else:
                call_strike = None  # Not near ATH, skip calls

            if call_strike is not None:
                call_premium = bs_call_price(price, call_strike, T_call, rf, sigma) * put_shares
                put_cash += call_premium
                total_call_premium += call_premium

                active_call_strike = call_strike
                active_call_shares = put_shares
                active_call_expiry = date + pd.Timedelta(days=CALL_DTE)

        put_equity = put_shares * price
        put_total_value = put_equity + put_cash
        total_premium = total_put_premium + total_call_premium

        # Record on contribution days
        if is_contribution_day:
            records.append({
                "Date": date,
                "Price": price,
                "Buy_Signal": row["buy_signal"],
                "Sell_Signal": row["sell_signal"],
                "DCA_Shares": dca_shares,
                "DCA_Portfolio": dca_portfolio_value,
                "DCA_Total_Invested": dca_total_invested,
                "Put_Shares": put_shares,
                "Put_Cash": put_cash,
                "Put_Equity": put_equity,
                "Put_Total_Value": put_total_value,
                "Put_Total_Invested": put_total_invested,
                "Total_Put_Premium": total_put_premium,
                "Total_Call_Premium": total_call_premium,
                "Total_Premium": total_premium,
                "Put_Assignments": put_assignment_count,
                "Call_Assignments": call_assignment_count,
                "Active_Put_Strike": active_put_strike,
                "Active_Call_Strike": active_call_strike,
                "Realized_Vol": row["realized_vol"],
                "HYSA_Rate_Annual": row["hysa_annual_pct"],
                "StdDev_100d": row["stddev_100d"],
            })

    results = pd.DataFrame(records)
    results.set_index("Date", inplace=True)
    return results


# -- Reporting --------------------------------------------------------------------

def print_summary(results: pd.DataFrame) -> None:
    """Print key summary statistics including Sharpe ratios and drawdowns."""
    final = results.iloc[-1]
    first = results.iloc[0]

    total_weeks = len(results)
    total_invested = final["DCA_Total_Invested"]

    dca_final = final["DCA_Portfolio"]
    put_final = final["Put_Total_Value"]

    dca_return = (dca_final - total_invested) / total_invested * 100
    put_return = (put_final - total_invested) / total_invested * 100

    years = (results.index[-1] - results.index[0]).days / 365.25
    dca_cagr = ((dca_final / total_invested) ** (1 / years) - 1) * 100 if years > 0 else 0
    put_cagr = ((put_final / total_invested) ** (1 / years) - 1) * 100 if years > 0 else 0

    # -- Sharpe Ratio --
    dca_weekly_returns = results["DCA_Portfolio"].pct_change().dropna()
    put_weekly_returns = results["Put_Total_Value"].pct_change().dropna()

    rf_weekly = results["HYSA_Rate_Annual"].iloc[1:] / 100 / 52
    rf_weekly = rf_weekly.reindex(dca_weekly_returns.index, method="ffill").fillna(0)

    dca_excess = dca_weekly_returns - rf_weekly
    put_excess = put_weekly_returns - rf_weekly

    dca_sharpe = (dca_excess.mean() / dca_excess.std() * np.sqrt(52)) if dca_excess.std() > 0 else 0
    put_sharpe = (put_excess.mean() / put_excess.std() * np.sqrt(52)) if put_excess.std() > 0 else 0

    # -- Drawdown Analysis --
    def compute_drawdown_stats(portfolio_series):
        running_max = portfolio_series.cummax()
        drawdown = (running_max - portfolio_series) / running_max * 100
        max_dd = drawdown.max()
        avg_dd = drawdown.mean()
        in_dd = drawdown > 0
        groups = (~in_dd).cumsum()
        dd_durations = []
        for _, group in in_dd.groupby(groups):
            if group.any():
                dd_durations.append(len(group))
        avg_dd_duration = np.mean(dd_durations) if dd_durations else 0
        max_dd_duration = max(dd_durations) if dd_durations else 0
        return avg_dd, max_dd, avg_dd_duration, max_dd_duration

    dca_avg_dd, dca_max_dd, dca_avg_dd_dur, dca_max_dd_dur = compute_drawdown_stats(results["DCA_Portfolio"])
    put_avg_dd, put_max_dd, put_avg_dd_dur, put_max_dd_dur = compute_drawdown_stats(results["Put_Total_Value"])

    print("=" * 70)
    print("V2: PUT SELLING + COVERED CALLS vs DCA (NQ Futures)")
    print("=" * 70)
    print(f"Period: {first.name.date()} -> {final.name.date()} ({years:.1f} years)")
    print(f"Total Weeks: {total_weeks}")
    print(f"Total Invested: ${total_invested:,.2f}")
    print(f"Weekly Contribution: ${WEEKLY_CONTRIBUTION:,.2f}")
    print()
    print("--- DCA Strategy ---")
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
    print("--- Put-Selling + Covered Call Strategy ---")
    print(f"  Final Equity Value:     ${final['Put_Equity']:,.2f}")
    print(f"  Final Cash Balance:     ${final['Put_Cash']:,.2f}")
    print(f"  Final Total Value:      ${put_final:,.2f}")
    print(f"  Total Return:           {put_return:,.2f}%")
    print(f"  CAGR:                   {put_cagr:.2f}%")
    print(f"  Sharpe Ratio:           {put_sharpe:.3f}")
    print(f"  Avg Drawdown:           {put_avg_dd:.2f}%")
    print(f"  Max Drawdown:           {put_max_dd:.2f}%")
    print(f"  Avg DD Duration:        {put_avg_dd_dur:.1f} weeks")
    print(f"  Max DD Duration:        {put_max_dd_dur} weeks")
    print(f"  Shares Accumulated:     {final['Put_Shares']:,.6f}")
    print(f"  Put Premium Collected:  ${final['Total_Put_Premium']:,.2f}")
    print(f"  Call Premium Collected: ${final['Total_Call_Premium']:,.2f}")
    print(f"  Total Premium:          ${final['Total_Premium']:,.2f}")
    print(f"  Put Assignments:        {int(final['Put_Assignments'])}")
    print(f"  Call Assignments:       {int(final['Call_Assignments'])}")
    print(f"  Buy Signal Weeks:       {results['Buy_Signal'].sum()} / {total_weeks}")
    print()
    print("--- Comparison ---")
    diff = put_final - dca_final
    winner = "Puts" if diff > 0 else "DCA"
    print(f"  Difference:             ${abs(diff):,.2f} in favor of {winner}")
    print(f"  Put/DCA Ratio:          {put_final/dca_final:.4f}")
    sharpe_winner = "Puts" if put_sharpe > dca_sharpe else "DCA"
    print(f"  Sharpe:  DCA={dca_sharpe:.3f}  Puts={put_sharpe:.3f}  -> {sharpe_winner}")
    dd_winner = "Puts" if put_avg_dd < dca_avg_dd else "DCA"
    print(f"  Avg DD:  DCA={dca_avg_dd:.2f}%  Puts={put_avg_dd:.2f}%  -> {dd_winner}")
    dd_max_winner = "Puts" if put_max_dd < dca_max_dd else "DCA"
    print(f"  Max DD:  DCA={dca_max_dd:.2f}%  Puts={put_max_dd:.2f}%  -> {dd_max_winner}")
    print("=" * 70)


def plot_results(results: pd.DataFrame, signals: pd.DataFrame) -> None:
    """Generate comparison charts."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
    fig.suptitle("V2: Put Selling + Covered Calls vs DCA (NQ Futures)", fontsize=16, fontweight="bold")

    # 1) Portfolio Value
    ax = axes[0]
    ax.plot(results.index, results["DCA_Portfolio"], label="DCA Portfolio", linewidth=1.5, color="#2196F3")
    ax.plot(results.index, results["Put_Total_Value"], label="Put+Call Strategy Total", linewidth=1.5, color="#FF9800")
    ax.plot(results.index, results["Put_Equity"], label="Equity Only", linewidth=1, linestyle=":", color="#FF5722", alpha=0.7)
    ax.plot(results.index, results["DCA_Total_Invested"], label="Total Invested", linewidth=1, linestyle="--", color="gray", alpha=0.7)
    ax.set_ylabel("Value ($)")
    ax.set_title("Portfolio Value Over Time")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # 2) NQ Price with assignment markers
    ax = axes[1]
    ax.plot(signals.index, signals["Close"], color="#555", linewidth=0.8, label="NQ Close")
    # Mark put assignments
    put_assign = results[results["Put_Assignments"].diff().fillna(0) > 0]
    if len(put_assign) > 0:
        ax.scatter(put_assign.index, put_assign["Price"],
                   color="green", s=30, alpha=0.8, label="Put Assignments", zorder=5, marker="^")
    # Mark call assignments
    call_assign = results[results["Call_Assignments"].diff().fillna(0) > 0]
    if len(call_assign) > 0:
        ax.scatter(call_assign.index, call_assign["Price"],
                   color="red", s=30, alpha=0.8, label="Call Assignments", zorder=5, marker="v")
    ax.set_ylabel("NQ Price")
    ax.set_title("NQ Futures Price & Assignments")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # 3) Premium collected over time (put + call)
    ax = axes[2]
    ax.plot(results.index, results["Total_Premium"], label="Total Premium", linewidth=1.5, color="#4CAF50")
    ax.plot(results.index, results["Total_Put_Premium"], label="Put Premium", linewidth=1, linestyle="--", color="#8BC34A")
    ax.plot(results.index, results["Total_Call_Premium"], label="Call Premium", linewidth=1, linestyle="--", color="#FF9800")
    ax.fill_between(results.index, 0, results["Total_Premium"], alpha=0.1, color="#4CAF50")
    ax.set_ylabel("Premium ($)")
    ax.set_title("Cumulative Premium Collected (Puts + Calls)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # 4) Realized Vol
    ax = axes[3]
    ax.plot(signals.index, signals["realized_vol"] * 100, label="100-day Realized Vol (%)",
            linewidth=0.8, color="#9C27B0")
    ax.set_ylabel("Volatility (%)")
    ax.set_title("100-Day Annualized Realized Volatility")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Date")

    plt.tight_layout()
    out = Path(__file__).parent / "v2_put_vs_dca_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {out}")
    plt.close()


# -- Main -------------------------------------------------------------------------

def main():
    print("Loading data...")
    nq = load_nq()
    rates = load_rates()

    print("Computing signals...")
    signals = compute_signals(nq)

    # Add realized vol to signals for plotting
    signals["realized_vol"] = compute_realized_vol(signals)
    signals["realized_vol"] = signals["realized_vol"].ffill().fillna(0.15)

    print("Running simulation...")
    results = simulate_v2(signals, rates)

    print_summary(results)
    plot_results(results, signals)

    # Save detailed results
    out_csv = Path(__file__).parent / "v2_put_vs_dca_results.csv"
    results.to_csv(out_csv)
    print(f"Results saved to {out_csv}")


if __name__ == "__main__":
    main()
