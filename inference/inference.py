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

if __name__ == "__main__":
    conn = get_connection()
    srm_results = check_srm(conn)
    print()
    power_results = check_power(conn)
    conn.close()
    