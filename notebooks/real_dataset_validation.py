"""
Phase 6B — Real Dataset Validation
Dataset: E-Commerce A/B Testing (Kaggle, ~294k rows)
Goal: Validate platform generalises beyond simulated data.
Runs SRM detection and primary metric inference only —
guardrail metrics not available in this dataset.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from pathlib import Path

DATA_PATH = Path(__file__).parent / "ab_data.csv"


def load_and_prepare(path):
    """
    Load Kaggle dataset and map to platform's expected
    column names.
    """
    df = pd.read_csv(path)

    # ── Data quality checks ──────────────────────────────────
    print("=" * 55)
    print("REAL DATASET — DATA QUALITY REPORT")
    print("=" * 55)
    print(f"Total rows:        {len(df):,}")
    print(f"Unique users:      {df['user_id'].nunique():,}")
    print(f"Duplicate user IDs:{len(df) - df['user_id'].nunique():,}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # ── Check for mismatched group/page assignments ───────────
    # Real data quality issue: some users in treatment
    # got old_page and vice versa — a real SRM-adjacent problem
    mismatch = df[
        ((df['group'] == 'treatment') &
         (df['landing_page'] == 'old_page')) |
        ((df['group'] == 'control') &
         (df['landing_page'] == 'new_page'))
    ]
    print(f"\nGroup/page mismatches: {len(mismatch):,}")
    print(f"  (users assigned to group but served wrong page)")

    # ── Remove mismatches — contaminated assignments ──────────
    df_clean = df[
        ((df['group'] == 'treatment') &
         (df['landing_page'] == 'new_page')) |
        ((df['group'] == 'control') &
         (df['landing_page'] == 'old_page'))
    ].copy()

    # Remove duplicate user IDs — keep first exposure only
    df_clean = df_clean.drop_duplicates(
        subset='user_id', keep='first'
    )

    # ── Map to platform column names ─────────────────────────
    df_clean = df_clean.rename(columns={
        'group':     'variant',
        'converted': 'purchased'
    })

    print(f"\nAfter cleaning:")
    print(f"  Rows:     {len(df_clean):,}")
    print(f"  Control:  "
          f"{(df_clean['variant']=='control').sum():,}")
    print(f"  Treatment:"
          f"{(df_clean['variant']=='treatment').sum():,}")
    print(f"  Overall conversion: "
          f"{df_clean['purchased'].mean()*100:.2f}%")
    print("=" * 55)

    return df_clean


def check_srm_real(df, alpha=0.05):
    """SRM detection on real dataset."""
    counts = df['variant'].value_counts()
    observed = [
        counts.get('control', 0),
        counts.get('treatment', 0)
    ]
    total = sum(observed)
    expected = [total/2, total/2]

    chi2, p_value = stats.chisquare(observed, expected)
    srm_detected = p_value < alpha

    print("\n" + "=" * 55)
    print("SRM DETECTION — REAL DATASET")
    print("=" * 55)
    print(f"Control:   {observed[0]:,}")
    print(f"Treatment: {observed[1]:,}")
    print(f"Chi2:      {chi2:.4f}")
    print(f"P-value:   {p_value:.4f}")
    print(f"SRM Detected: {srm_detected}")
    print("=" * 55)

    return srm_detected


def run_inference_real(df, alpha=0.05):
    """Primary metric inference on real dataset."""
    control   = df[df['variant'] == 'control']
    treatment = df[df['variant'] == 'treatment']

    n_control      = len(control)
    n_treatment    = len(treatment)
    conv_control   = control['purchased'].mean()
    conv_treatment = treatment['purchased'].mean()

    purchases_control   = control['purchased'].sum()
    purchases_treatment = treatment['purchased'].sum()

    # Two-proportion z-test
    count = np.array([purchases_treatment, purchases_control])
    nobs  = np.array([n_treatment, n_control])
    z_stat, p_value = proportions_ztest(count, nobs)

    # Lift and CI
    diff = conv_treatment - conv_control
    se_diff = np.sqrt(
        conv_control * (1-conv_control) / n_control +
        conv_treatment * (1-conv_treatment) / n_treatment
    )
    z_crit = stats.norm.ppf(1 - alpha/2)
    ci_low  = diff - z_crit * se_diff
    ci_high = diff + z_crit * se_diff

    # Post-hoc power
    p_pooled = df['purchased'].mean()
    sigma_sq = p_pooled * (1 - p_pooled)
    mde = 0.005
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta_achieved = (
        np.sqrt((min(n_control, n_treatment) * mde**2) /
                (2 * sigma_sq)) - z_alpha
    )
    power_achieved = stats.norm.cdf(z_beta_achieved)

    significant = p_value < alpha
    relative_lift = (
        (conv_treatment - conv_control) / conv_control
    )

    print("\n" + "=" * 55)
    print("PRIMARY METRIC INFERENCE — REAL DATASET")
    print("=" * 55)
    print(f"Control:   {conv_control:.4f} "
          f"({purchases_control:,}/{n_control:,})")
    print(f"Treatment: {conv_treatment:.4f} "
          f"({purchases_treatment:,}/{n_treatment:,})")
    print(f"\nAbsolute lift:  {diff*100:+.3f}pp")
    print(f"Relative lift:  {relative_lift*100:+.2f}%")
    print(f"95% CI:         [{ci_low*100:.3f}pp, "
          f"{ci_high*100:.3f}pp]")
    print(f"Z-statistic:    {z_stat:.4f}")
    print(f"P-value:        {p_value:.4f}")
    print(f"Significant:    {significant}")
    print(f"Achieved power: {power_achieved*100:.1f}%")

    print("\n" + "─"*55)
    print("RECOMMENDATION")
    print("─"*55)

    if not significant:
        rec = "EXTEND / DO NOT SHIP"
        reason = (
            f"No statistically significant difference "
            f"detected (p={p_value:.4f}). "
            f"New landing page does not outperform control."
        )
    elif significant and diff > 0:
        rec = "SHIP"
        reason = (
            f"Significant positive result (p={p_value:.4f},"
            f" lift={diff*100:+.3f}pp). Deploy new page."
        )
    else:
        rec = "DO NOT SHIP"
        reason = (
            f"Significant negative result (p={p_value:.4f},"
            f" lift={diff*100:+.3f}pp). "
            f"New page performs worse."
        )

    print(f"Recommendation: {rec}")
    print(f"Reason: {reason}")
    print("=" * 55)

    return {
        "conv_control":   round(conv_control, 4),
        "conv_treatment": round(conv_treatment, 4),
        "p_value":        round(p_value, 4),
        "significant":    significant,
        "recommendation": rec,
        "power_achieved": round(power_achieved, 4)
    }


def compare_with_simulation():
    """
    Side-by-side comparison of simulation vs real dataset
    results — the core narrative of Phase 6B.
    """
    print("\n" + "=" * 55)
    print("SIMULATION vs REAL DATASET COMPARISON")
    print("=" * 55)
    print(f"{'Metric':<30} {'Small Sim':>10} "
          f"{'Large Sim':>10} {'Real Data':>10}")
    print("─" * 62)
    print(f"{'Users':<30} {'2,100':>10} "
          f"{'50,000':>10} {'~290k':>10}")
    print(f"{'Purchase events':<30} {'56':>10} "
          f"{'1,300':>10} {'~35k':>10}")
    print(f"{'Achieved power':<30} {'10.5%':>10} "
          f"{'93.9%':>10} {'100%+':>10}")
    print(f"{'P-value':<30} {'0.1972':>10} "
          f"{'0.0001':>10} {'TBD':>10}")
    print(f"{'Recommendation':<30} {'EXTEND':>10} "
          f"{'SHIP':>10} {'TBD':>10}")
    print("─" * 62)
    print("\nKey insight: Same platform, same statistical")
    print("engine, three different data sources.")
    print("Platform correctly adapts recommendation")
    print("based on evidence quality, not assumptions.")
    print("=" * 55)


if __name__ == "__main__":
    # Load and prepare
    df = load_and_prepare(DATA_PATH)

    # Show comparison table first
    compare_with_simulation()

    # Run platform checks
    srm = check_srm_real(df)
    results = run_inference_real(df)