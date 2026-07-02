import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "experiments.db"

def get_connection():
    return sqlite3.connect(DB_PATH)


def check_srm(conn, experiment_id="EXP_001", alpha=0.05):
    """
    Sample Ratio Mismatch detection using chi-square test.
    Checks overall assignment AND per-tier assignment
    (since stratified randomisation should balance each tier).
    
    Returns: dict with overall + per-tier SRM status
    """
    df_assignments = pd.read_sql(
        f"SELECT * FROM assignments "
        f"WHERE experiment_id = '{experiment_id}'",
        conn
    )
    df_users = pd.read_sql("SELECT * FROM users", conn)

    df = df_assignments.merge(
        df_users[["user_id", "spend_tier"]], on="user_id"
    )

    results = {}

    # ── Overall SRM check ────────────────────────────────────
    overall_counts = df["variant"].value_counts()
    observed = [
        overall_counts.get("control", 0),
        overall_counts.get("treatment", 0)
    ]
    total = sum(observed)
    expected = [total / 2, total / 2]

    chi2, p_value = stats.chisquare(observed, expected)

    results["overall"] = {
        "observed_control": observed[0],
        "observed_treatment": observed[1],
        "expected_per_group": expected[0],
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 4),
        "srm_detected": p_value < alpha
    }

    # ── Per-tier SRM check (stratification validation) ──────
    tier_results = {}
    for tier in df["spend_tier"].unique():
        tier_df = df[df["spend_tier"] == tier]
        tier_counts = tier_df["variant"].value_counts()
        obs = [
            tier_counts.get("control", 0),
            tier_counts.get("treatment", 0)
        ]
        tier_total = sum(obs)
        exp = [tier_total / 2, tier_total / 2]

        chi2_t, p_t = stats.chisquare(obs, exp)

        tier_results[tier] = {
            "observed_control": obs[0],
            "observed_treatment": obs[1],
            "p_value": round(p_t, 4),
            "srm_detected": p_t < alpha
        }

    results["by_tier"] = tier_results

    # ── Print readable report ────────────────────────────────
    print("=" * 55)
    print("SRM DETECTION REPORT")
    print("=" * 55)
    print(f"\nOverall: {results['overall']['observed_control']} "
          f"control vs "
          f"{results['overall']['observed_treatment']} treatment")
    print(f"Chi2 = {results['overall']['chi2_statistic']}, "
          f"p = {results['overall']['p_value']}")
    print(f"SRM Detected: {results['overall']['srm_detected']}")

    print("\nPer-tier breakdown:")
    for tier, r in tier_results.items():
        flag = "FLAGGED" if r["srm_detected"] else "OK"
        print(
            f"  {tier:15s} control={r['observed_control']:4d} "
            f"treatment={r['observed_treatment']:4d}  "
            f"p={r['p_value']:.4f}  {flag}"
        )
    print("=" * 55)

    return results


def check_power(conn, experiment_id="EXP_001",
                alpha=0.05, mde=0.005):
    """
    Post-hoc power analysis.
    Calculates actual statistical power achieved given
    the real sample size and observed metric variance —
    rather than the planned/assumed values from Phase 1.
    
    A low achieved power means even a real effect may
    have been missed — results should be treated with caution.
    """
    df_assignments = pd.read_sql(
        f"SELECT * FROM assignments "
        f"WHERE experiment_id = '{experiment_id}'",
        conn
    )
    df_events = pd.read_sql(
        f"SELECT * FROM events "
        f"WHERE experiment_id = '{experiment_id}' "
        f"AND event_type = 'purchase'",
        conn
    )
    df_users = pd.read_sql("SELECT * FROM users", conn)

    # ── Actual sample size per group ─────────────────────────
    n_control = (
        df_assignments["variant"] == "control"
    ).sum()
    n_treatment = (
        df_assignments["variant"] == "treatment"
    ).sum()
    n_per_group = min(n_control, n_treatment)

    # ── Observed conversion rate and variance ────────────────
    # All assigned users — purchasers get 1, others get 0
    df_all = df_assignments[["user_id", "variant"]].copy()
    df_all["purchased"] = df_all["user_id"].isin(
        df_events["user_id"]
    ).astype(int)

    conv_control = df_all[
        df_all["variant"] == "control"
    ]["purchased"].mean()
    conv_treatment = df_all[
        df_all["variant"] == "treatment"
    ]["purchased"].mean()

    # Pooled conversion rate for variance estimate
    p_pooled = df_all["purchased"].mean()

    # Variance of a binary metric = p * (1 - p)
    sigma_sq = p_pooled * (1 - p_pooled)

    # ── Planned sample size (from Phase 1 formula) ───────────
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # 1.96 for α=0.05
    z_beta_planned = 0.842                    # 80% power

    n_planned = (
        2 * (z_alpha + z_beta_planned) ** 2 * sigma_sq
    ) / (mde ** 2)

    # ── Achieved power (flipped formula) ────────────────────
    z_beta_achieved = (
        np.sqrt(
            (n_per_group * mde ** 2) /
            (2 * sigma_sq)
        ) - z_alpha
    )
    power_achieved = stats.norm.cdf(z_beta_achieved)

    # ── Print report ─────────────────────────────────────────
    print("=" * 55)
    print("POST-HOC POWER ANALYSIS")
    print("=" * 55)
    print(f"Planned sample size per group: {n_planned:,.0f}")
    print(f"Actual sample size per group:  {n_per_group:,}")
    print(f"MDE:                           {mde*100:.1f}%")
    print(f"Observed conversion (control): "
          f"{conv_control:.4f}")
    print(f"Observed conversion (treatment): "
          f"{conv_treatment:.4f}")
    print(f"Pooled variance (p*(1-p)):     "
          f"{sigma_sq:.6f}")
    print(f"\nPlanned power:    80.0%")
    print(f"Achieved power:   {power_achieved*100:.1f}%")

    if power_achieved < 0.80:
        print(f"\n  UNDERPOWERED EXPERIMENT")
        print(f"   Achieved power ({power_achieved*100:.1f}%) "
              f"is below the 80% threshold.")
        print(f"   Results should be treated with caution.")
        print(f"   To achieve 80% power, you would need")
        print(f"   ~{n_planned:,.0f} users per group")
        print(f"   ({n_planned*2:,.0f} total).")
    else:
        print(f"\n Experiment adequately powered.")

    print("=" * 55)

    return {
        "n_planned": n_planned,
        "n_actual": n_per_group,
        "power_achieved": round(power_achieved, 4),
        "adequately_powered": power_achieved >= 0.80,
        "conv_control": round(conv_control, 4),
        "conv_treatment": round(conv_treatment, 4),
        "mde": mde
    }

def check_guardrails(conn, experiment_id="EXP_001",
                     alpha=0.05):
    """
    Guardrail metric monitoring with multiple testing
    correction (Bonferroni).
    Three guardrail metrics derived/simulated from
    available event data:
    1. Cart-to-purchase rate (proxy for review/trust)
    2. Seller diversity score (simulated)
    3. Multi-day retention rate (from event timestamps)
    If ANY guardrail is violated, experiment should NOT
    be shipped regardless of primary metric result.
    """
    np.random.seed(42)

    df_assignments = pd.read_sql(
        f"SELECT * FROM assignments "
        f"WHERE experiment_id = '{experiment_id}'",
        conn
    )
    df_events = pd.read_sql(
        f"SELECT * FROM events "
        f"WHERE experiment_id = '{experiment_id}'",
        conn
    )

    from statsmodels.stats.proportion import proportions_ztest

    # ── Bonferroni correction ────────────────────────────────
    n_tests = 3
    alpha_corrected = alpha / n_tests
    print(f"Bonferroni corrected α: "
          f"{alpha_corrected:.4f} (original α={alpha})")

    guardrail_results = {}

    # ── Guardrail 1: Cart-to-purchase rate ──────────────────
    cart_events = df_events[
        df_events["event_type"] == "add_to_cart"
    ][["user_id", "variant"]].copy()

    purchase_users = set(
        df_events[
            df_events["event_type"] == "purchase"
        ]["user_id"]
    )

    cart_events["purchased"] = cart_events[
        "user_id"
    ].isin(purchase_users).astype(int)

    cart_control = cart_events[
        cart_events["variant"] == "control"
    ]["purchased"]
    cart_treatment = cart_events[
        cart_events["variant"] == "treatment"
    ]["purchased"]

    rate_control_cart   = cart_control.mean()
    rate_treatment_cart = cart_treatment.mean()
    relative_change_cart = (
        (rate_treatment_cart - rate_control_cart) /
        rate_control_cart
    )

    count1 = np.array([
        cart_treatment.sum(), cart_control.sum()
    ])
    nobs1 = np.array([
        len(cart_treatment), len(cart_control)
    ])
    z1, p1 = proportions_ztest(count1, nobs1)

    guardrail_results["cart_to_purchase_rate"] = {
        "control_rate":    round(rate_control_cart, 4),
        "treatment_rate":  round(rate_treatment_cart, 4),
        "relative_change": round(relative_change_cart, 4),
        "p_value":         round(p1, 4),
        "threshold":       -0.05,
        "violated": (
            relative_change_cart < -0.05 and
            p1 < alpha_corrected
        ),
        "warning": (
            relative_change_cart < -0.05 and
            p1 >= alpha_corrected
        )
    }

    # ── Guardrail 2: Seller diversity score ──────────────────
    n_control   = (df_assignments["variant"] == "control").sum()
    n_treatment = (df_assignments["variant"] == "treatment").sum()

    diversity_control   = np.random.normal(
        0.72, 0.15, n_control
    ).clip(0, 1)
    diversity_treatment = np.random.normal(
        0.68, 0.15, n_treatment
    ).clip(0, 1)

    mean_div_control   = diversity_control.mean()
    mean_div_treatment = diversity_treatment.mean()
    relative_change_div = (
        (mean_div_treatment - mean_div_control) /
        mean_div_control
    )

    t2, p2 = stats.ttest_ind(
        diversity_treatment, diversity_control
    )

    guardrail_results["seller_diversity"] = {
        "control_rate":    round(mean_div_control, 4),
        "treatment_rate":  round(mean_div_treatment, 4),
        "relative_change": round(relative_change_div, 4),
        "p_value":         round(p2, 4),
        "threshold":       -0.10,
        "violated": (
            relative_change_div < -0.10 and
            p2 < alpha_corrected
        ),
        "warning": (
            relative_change_div < -0.10 and
            p2 >= alpha_corrected
        )
    }

    # ── Guardrail 3: Multi-day retention rate ────────────────
    df_events_copy = df_events.copy()
    df_events_copy["date"] = pd.to_datetime(
        df_events_copy["event_timestamp"]
    ).dt.date

    user_dates = df_events_copy.groupby(
        "user_id"
    )["date"].nunique().reset_index()
    user_dates.columns = ["user_id", "active_days"]

    df_retention = df_assignments[
        ["user_id", "variant"]
    ].merge(user_dates, on="user_id", how="left")
    df_retention["active_days"] = df_retention[
        "active_days"
    ].fillna(0)
    df_retention["retained"] = (
        df_retention["active_days"] > 1
    ).astype(int)

    ret_control = df_retention[
        df_retention["variant"] == "control"
    ]["retained"]
    ret_treatment = df_retention[
        df_retention["variant"] == "treatment"
    ]["retained"]

    rate_control_ret   = ret_control.mean()
    rate_treatment_ret = ret_treatment.mean()
    relative_change_ret = (
        (rate_treatment_ret - rate_control_ret) /
        rate_control_ret
    )

    count3 = np.array([
        ret_treatment.sum(), ret_control.sum()
    ])
    nobs3 = np.array([
        len(ret_treatment), len(ret_control)
    ])
    z3, p3 = proportions_ztest(count3, nobs3)

    guardrail_results["retention_rate"] = {
        "control_rate":    round(rate_control_ret, 4),
        "treatment_rate":  round(rate_treatment_ret, 4),
        "relative_change": round(relative_change_ret, 4),
        "p_value":         round(p3, 4),
        "threshold":       -0.05,
        "violated": (
            relative_change_ret < -0.05 and
            p3 < alpha_corrected
        ),
        "warning": (
            relative_change_ret < -0.05 and
            p3 >= alpha_corrected
        )
    }

    # ── Print report ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("GUARDRAIL MONITORING REPORT")
    print("=" * 55)

    any_violated = False
    any_warning  = False

    for metric, r in guardrail_results.items():
        if r["violated"]:
            status = "VIOLATED"
            any_violated = True
        elif r.get("warning"):
            status = "WARNING"
            any_warning = True
        else:
            status = "PASSED"

        print(f"\n{metric}:")
        print(f"  Control:         {r['control_rate']:.4f}")
        print(f"  Treatment:       {r['treatment_rate']:.4f}")
        print(f"  Relative change: "
              f"{r['relative_change']*100:.2f}%")
        print(f"  Threshold:       "
              f"{r['threshold']*100:.0f}% max drop")
        print(f"  p-value:         {r['p_value']:.4f}")
        print(f"  Status:          {status}")

    print("\n" + "=" * 55)
    if any_violated:
        print("GUARDRAIL VIOLATION DETECTED")
        print("   DO NOT SHIP — investigate before proceeding")
    elif any_warning:
        print("GUARDRAIL WARNING")
        print("   Threshold breached but not statistically")
        print("   significant — experiment may be underpowered.")
        print("   Proceed with caution. Monitor closely")
        print("   post-ship if decision is made to launch.")
    else:
        print("ALL GUARDRAILS PASSED")
        print("  Safe to proceed to statistical inference")
    print("=" * 55)

    return guardrail_results

def run_inference(conn, experiment_id="EXP_001",
                  alpha=0.05, mde=0.005):
    """
    Primary metric statistical inference.
    Tests conversion rate difference between control
    and treatment using two-proportion z-test.
    
    Reports p-value, confidence interval, effect size,
    and recommendation — always contextualised against
    Phase 3 validation warnings.
    """
    from statsmodels.stats.proportion import (
        proportions_ztest, proportion_confint
    )

    df_assignments = pd.read_sql(
        f"SELECT * FROM assignments "
        f"WHERE experiment_id = '{experiment_id}'",
        conn
    )
    df_events = pd.read_sql(
        f"SELECT * FROM events "
        f"WHERE experiment_id = '{experiment_id}' "
        f"AND event_type = 'purchase'",
        conn
    )

    # ── Build conversion flag per user ───────────────────────
    df_all = df_assignments[["user_id", "variant"]].copy()
    df_all["purchased"] = df_all["user_id"].isin(
        df_events["user_id"]
    ).astype(int)

    control   = df_all[df_all["variant"] == "control"]
    treatment = df_all[df_all["variant"] == "treatment"]

    n_control      = len(control)
    n_treatment    = len(treatment)
    conv_control   = control["purchased"].mean()
    conv_treatment = treatment["purchased"].mean()

    purchases_control   = control["purchased"].sum()
    purchases_treatment = treatment["purchased"].sum()

    # ── Two-proportion z-test ────────────────────────────────
    count = np.array([purchases_treatment, purchases_control])
    nobs  = np.array([n_treatment, n_control])
    z_stat, p_value = proportions_ztest(count, nobs)

    # ── Confidence interval for treatment rate ───────────────
    ci_low, ci_high = proportion_confint(
        purchases_treatment, n_treatment,
        alpha=alpha, method="normal"
    )

    # ── Relative lift ────────────────────────────────────────
    relative_lift = (
        (conv_treatment - conv_control) / conv_control
    )

    # ── Effect size (Cohen's h for proportions) ──────────────
    cohens_h = (
        2 * np.arcsin(np.sqrt(conv_treatment)) -
        2 * np.arcsin(np.sqrt(conv_control))
    )

    # ── Confidence interval on lift ──────────────────────────
    se_diff = np.sqrt(
        conv_control * (1 - conv_control) / n_control +
        conv_treatment * (1 - conv_treatment) / n_treatment
    )
    z_critical = stats.norm.ppf(1 - alpha / 2)
    diff = conv_treatment - conv_control
    ci_diff_low  = diff - z_critical * se_diff
    ci_diff_high = diff + z_critical * se_diff

    # ── Print report ─────────────────────────────────────────
    print("=" * 55)
    print("PRIMARY METRIC INFERENCE REPORT")
    print("=" * 55)
    print(f"\nMetric: Conversion Rate")
    print(f"Test:   Two-proportion z-test (two-sided)")
    print(f"\nControl:   {conv_control:.4f} "
          f"({purchases_control}/{n_control} users)")
    print(f"Treatment: {conv_treatment:.4f} "
          f"({purchases_treatment}/{n_treatment} users)")
    print(f"\nAbsolute lift:  {diff*100:+.3f} pp")
    print(f"Relative lift:  {relative_lift*100:+.2f}%")
    print(f"Cohen's h:      {cohens_h:.4f}")
    print(f"\n95% CI on lift: "
          f"[{ci_diff_low*100:.3f}pp, "
          f"{ci_diff_high*100:.3f}pp]")
    print(f"Z-statistic:    {z_stat:.4f}")
    print(f"P-value:        {p_value:.4f}")

    significant = p_value < alpha
    print(f"\nStatistically significant "
          f"(α={alpha}): {significant}")

    # ── Contextualised interpretation ────────────────────────
    print("\n" + "─" * 55)
    print("CONTEXTUALISED INTERPRETATION")
    print("─" * 55)

    if significant and diff > 0:
        raw_recommendation = "SHIP"
    elif significant and diff < 0:
        raw_recommendation = "DO NOT SHIP"
    else:
        raw_recommendation = "INCONCLUSIVE"

    print(f"Raw statistical result: {raw_recommendation}")
    print(
        f"\nIMPORTANT CAVEATS (from Phase 3 validation):"
    )
    print(f"  • Achieved power: 10.5% (planned: 80%)")
    print(f"  • Experiment needs ~32,604 users "
          f"(actual: {n_control + n_treatment:,})")
    print(f"  • Retention rate shows -18.62% directional")
    print(f"    drop (WARNING — unconfirmed due to low power)")
    print(
        f"\nFinal recommendation: EXTEND OR RERUN"
    )
    print(
        f"  Statistical result alone is unreliable at"
    )
    print(
        f"  10.5% power. Retention warning unresolved."
    )
    print(
        f"  Collect more data before shipping decision."
    )
    print("=" * 55)

    return {
        "n_control":          n_control,
        "n_treatment":        n_treatment,
        "conv_control":       round(conv_control, 4),
        "conv_treatment":     round(conv_treatment, 4),
        "absolute_lift":      round(diff, 6),
        "relative_lift":      round(relative_lift, 4),
        "cohens_h":           round(cohens_h, 4),
        "ci_diff_low":        round(ci_diff_low, 6),
        "ci_diff_high":       round(ci_diff_high, 6),
        "z_statistic":        round(z_stat, 4),
        "p_value":            round(p_value, 4),
        "significant":        significant,
        "raw_recommendation": raw_recommendation
    }


def demonstrate_peeking(conn, experiment_id="EXP_001",
                         alpha=0.05, n_simulations=10000):
    """
    Demonstrates how peeking at experiment results daily
    inflates false positive rate far beyond nominal alpha.
    
    Simulates experiments under H0 (no true effect) and
    shows how often peeking leads to incorrect rejection.
    
    This is why sequential testing methods exist —
    used at Optimizely, Booking.com, and Netflix.
    """
    from statsmodels.stats.proportion import proportions_ztest

    print("=" * 55)
    print("PEEKING PROBLEM DEMONSTRATION")
    print("=" * 55)
    print(f"\nSimulating {n_simulations:,} experiments")
    print(f"where H0 is TRUE (no real effect)")
    print(f"Nominal α = {alpha}")
    print(f"Experiment duration: 14 days")

    # ── Simulation parameters ────────────────────────────────
    # Based on your actual experiment parameters
    true_conversion_rate = 0.027  # pooled rate, H0 true
    daily_users_per_group = 75    # ~1050 users / 14 days

    false_positive_single  = 0    # look only at Day 14
    false_positive_peeking = 0    # look every day

    np.random.seed(42)

    for _ in range(n_simulations):
        # Simulate 14 days of data under H0
        # (same true rate for both groups)
        control_daily   = []
        treatment_daily = []

        for day in range(1, 15):
            control_daily.append(
                np.random.binomial(
                    daily_users_per_group,
                    true_conversion_rate
                )
            )
            treatment_daily.append(
                np.random.binomial(
                    daily_users_per_group,
                    true_conversion_rate
                )
            )

        # ── Single look at Day 14 ────────────────────────────
        total_control   = sum(control_daily)
        total_treatment = sum(treatment_daily)
        n_total         = daily_users_per_group * 14

        count = np.array([total_treatment, total_control])
        nobs  = np.array([n_total, n_total])
        _, p  = proportions_ztest(count, nobs)

        if p < alpha:
            false_positive_single += 1

        # ── Peeking — check every day ────────────────────────
        peeked_significant = False
        cumulative_control   = 0
        cumulative_treatment = 0

        for day in range(14):
            cumulative_control   += control_daily[day]
            cumulative_treatment += treatment_daily[day]
            n_so_far = daily_users_per_group * (day + 1)

            # Need at least 5 conversions to run test
            if (cumulative_control < 5 or
                    cumulative_treatment < 5):
                continue

            count_d = np.array([
                cumulative_treatment, cumulative_control
            ])
            nobs_d  = np.array([n_so_far, n_so_far])
            _, p_d  = proportions_ztest(count_d, nobs_d)

            if p_d < alpha:
                peeked_significant = True
                break

        if peeked_significant:
            false_positive_peeking += 1

    # ── Results ──────────────────────────────────────────────
    fpr_single  = false_positive_single  / n_simulations
    fpr_peeking = false_positive_peeking / n_simulations

    print(f"\nResults (H0 true — no real effect exists):")
    print(f"{'─'*45}")
    print(f"Single look at Day 14:")
    print(f"  False positive rate: {fpr_single*100:.1f}%")
    print(f"  (Expected: ~{alpha*100:.0f}%)")
    print(f"\nPeeking daily (stop when p<{alpha}):")
    print(f"  False positive rate: {fpr_peeking*100:.1f}%")
    print(f"  (Expected: ~{alpha*100:.0f}%, "
          f"Actual: {fpr_peeking*100:.1f}%)")
    print(f"\nPeeking inflates false positive rate by "
          f"{fpr_peeking/fpr_single:.1f}x")
    print(f"From {fpr_single*100:.1f}% → "
          f"{fpr_peeking*100:.1f}%")
    print(f"\nThis is why Optimizely, Netflix, and")
    print(f"Booking.com use sequential testing methods")
    print(f"that mathematically correct for multiple looks.")
    print("=" * 55)

    return {
        "fpr_single":        round(fpr_single, 4),
        "fpr_peeking":       round(fpr_peeking, 4),
        "inflation_factor":  round(fpr_peeking/fpr_single, 2),
        "n_simulations":     n_simulations
    }

def run_cuped(conn, experiment_id="EXP_001", alpha=0.05):
    """
    CUPED: Controlled-experiment Using Pre-Experiment Data.
    
    Uses historical_revenue_30d as pre-experiment covariate
    to reduce variance in conversion rate outcome metric.
    
    Demonstrates sensitivity improvement without increasing
    sample size — used in production at Microsoft, Netflix,
    and Booking.com.
    
    Formula: Y_cuped = Y - theta * (X - mean(X))
    where theta = Cov(Y,X) / Var(X)
    """
    from statsmodels.stats.proportion import proportions_ztest

    df_assignments = pd.read_sql(
        f"SELECT * FROM assignments "
        f"WHERE experiment_id = '{experiment_id}'",
        conn
    )
    df_events = pd.read_sql(
        f"SELECT * FROM events "
        f"WHERE experiment_id = '{experiment_id}' "
        f"AND event_type = 'purchase'",
        conn
    )
    df_users = pd.read_sql("SELECT * FROM users", conn)

    # ── Build analysis dataframe ─────────────────────────────
    df = df_assignments[["user_id", "variant"]].merge(
        df_users[["user_id", "historical_revenue_30d"]],
        on="user_id"
    )
    df["purchased"] = df["user_id"].isin(
        df_events["user_id"]
    ).astype(int)

    # ── Standard analysis (without CUPED) ───────────────────
    control   = df[df["variant"] == "control"]
    treatment = df[df["variant"] == "treatment"]

    conv_control   = control["purchased"].mean()
    conv_treatment = treatment["purchased"].mean()
    var_standard   = df["purchased"].var()

    count = np.array([
        treatment["purchased"].sum(),
        control["purchased"].sum()
    ])
    nobs = np.array([len(treatment), len(control)])
    _, p_standard = proportions_ztest(count, nobs)

    # ── CUPED adjustment ─────────────────────────────────────
    # theta = Cov(Y, X) / Var(X)
    # Calculated on combined data to avoid treatment leakage
    X = df["historical_revenue_30d"].values
    Y = df["purchased"].values

    theta = np.cov(Y, X)[0, 1] / np.var(X)

    # Apply CUPED adjustment
    X_mean = X.mean()
    df["purchased_cuped"] = (
        df["purchased"] - theta * (
            df["historical_revenue_30d"] - X_mean
        )
    )

    # ── CUPED analysis ───────────────────────────────────────
    control_cuped   = df[df["variant"] == "control"]
    treatment_cuped = df[df["variant"] == "treatment"]

    conv_control_cuped   = (
        control_cuped["purchased_cuped"].mean()
    )
    conv_treatment_cuped = (
        treatment_cuped["purchased_cuped"].mean()
    )
    var_cuped = df["purchased_cuped"].var()

    # T-test on CUPED metric (no longer strictly binary)
    t_stat, p_cuped = stats.ttest_ind(
        treatment_cuped["purchased_cuped"],
        control_cuped["purchased_cuped"]
    )

    # ── Variance reduction ───────────────────────────────────
    variance_reduction = (
        (var_standard - var_cuped) / var_standard
    )

    # ── Confidence intervals ─────────────────────────────────
    diff_standard = conv_treatment   - conv_control
    diff_cuped    = conv_treatment_cuped - conv_control_cuped

    se_standard = np.sqrt(
        conv_control * (1 - conv_control) / len(control) +
        conv_treatment * (1 - conv_treatment) / len(treatment)
    )
    se_cuped = np.sqrt(
        var_cuped / len(control) +
        var_cuped / len(treatment)
    )

    z_crit = stats.norm.ppf(1 - alpha / 2)

    ci_standard = (
        diff_standard - z_crit * se_standard,
        diff_standard + z_crit * se_standard
    )
    ci_cuped = (
        diff_cuped - z_crit * se_cuped,
        diff_cuped + z_crit * se_cuped
    )

    # ── Print report ─────────────────────────────────────────
    print("=" * 55)
    print("CUPED VARIANCE REDUCTION REPORT")
    print("=" * 55)
    print(f"\nPre-experiment covariate: historical_revenue_30d")
    print(f"Theta (regression coefficient): {theta:.6f}")
    print(f"  → Each ₹1 of historical revenue predicts")
    print(f"    {theta*100:.4f}pp change in conversion rate")

    print(f"\n{'─'*45}")
    print(f"{'Metric':<35} {'Standard':>8} {'CUPED':>8}")
    print(f"{'─'*45}")
    print(f"{'Control conversion':<35} "
          f"{conv_control:>8.4f} "
          f"{conv_control_cuped:>8.4f}")
    print(f"{'Treatment conversion':<35} "
          f"{conv_treatment:>8.4f} "
          f"{conv_treatment_cuped:>8.4f}")
    print(f"{'Metric variance':<35} "
          f"{var_standard:>8.6f} "
          f"{var_cuped:>8.6f}")
    print(f"{'P-value':<35} "
          f"{p_standard:>8.4f} "
          f"{p_cuped:>8.4f}")
    print(f"{'95% CI lower (pp)':<35} "
          f"{ci_standard[0]*100:>8.3f} "
          f"{ci_cuped[0]*100:>8.3f}")
    print(f"{'95% CI upper (pp)':<35} "
          f"{ci_standard[1]*100:>8.3f} "
          f"{ci_cuped[1]*100:>8.3f}")
    print(f"{'─'*45}")
    print(f"\nVariance reduction: "
          f"{variance_reduction*100:.1f}%")

    ci_width_standard = (
        ci_standard[1] - ci_standard[0]
    ) * 100
    ci_width_cuped = (
        ci_cuped[1] - ci_cuped[0]
    ) * 100
    ci_improvement = (
        (ci_width_standard - ci_width_cuped) /
        ci_width_standard
    )

    print(f"CI width (standard): {ci_width_standard:.3f}pp")
    print(f"CI width (CUPED):    {ci_width_cuped:.3f}pp")
    print(f"CI narrowing:        "
          f"{ci_improvement*100:.1f}%")

    print(f"\nBusiness interpretation:")
    print(f"  CUPED reduced variance by "
          f"{variance_reduction*100:.1f}%, producing")
    print(f"  {ci_improvement*100:.1f}% narrower confidence "
          f"intervals.")
    print(f"  This means you could detect the same effect")
    print(f"  with fewer users — or detect smaller effects")
    print(f"  with the same {len(df):,} users.")
    print("=" * 55)

    return {
        "theta":               round(theta, 6),
        "variance_standard":   round(var_standard, 6),
        "variance_cuped":      round(var_cuped, 6),
        "variance_reduction":  round(variance_reduction, 4),
        "p_standard":          round(p_standard, 4),
        "p_cuped":             round(p_cuped, 4),
        "ci_width_standard":   round(ci_width_standard, 4),
        "ci_width_cuped":      round(ci_width_cuped, 4)
    }


if __name__ == "__main__":
    conn = get_connection()
    srm_results       = check_srm(conn)
    print()
    power_results     = check_power(conn)
    print()
    guardrail_results = check_guardrails(conn)
    print()
    inference_results = run_inference(conn)
    print()
    peeking_results   = demonstrate_peeking(conn)
    print()
    cuped_results     = run_cuped(conn)
    conn.close()