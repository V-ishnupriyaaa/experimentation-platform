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


if __name__ == "__main__":
    conn = get_connection()
    results = check_srm(conn)
    conn.close()