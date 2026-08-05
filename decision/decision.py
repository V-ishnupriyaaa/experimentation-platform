import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import sys
import os

# ── Add parent directory to path for imports ─────────────
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from inference.inference import (
    check_srm,
    check_power,
    check_guardrails,
    run_inference,
    run_cuped
)

DB_PATH = Path(__file__).parent.parent / "data" / "experiments.db"

def get_connection():
    return sqlite3.connect(DB_PATH)


def calculate_financial_impact(
    conv_control, ci_low, ci_high,
    daily_traffic=10000,
    avg_order_value=1200,
    experiment_days=14
):
    """
    Translates statistical lift into financial impact range.
    Uses confidence interval bounds — not just point estimate —
    to give honest best/worst case revenue projections.
    
    daily_traffic: estimated daily platform visits
    avg_order_value: average purchase value in ₹
    """
    # Annual revenue impact at lower and upper CI bounds
    annual_days = 365

    # Lower bound (conservative case)
    impact_lower = (
        ci_low * daily_traffic *
        avg_order_value * annual_days
    )

    # Upper bound (optimistic case)
    impact_upper = (
        ci_high * daily_traffic *
        avg_order_value * annual_days
    )

    # Point estimate
    lift_point = (ci_low + ci_high) / 2
    impact_point = (
        lift_point * daily_traffic *
        avg_order_value * annual_days
    )

    return {
        "daily_traffic":    daily_traffic,
        "avg_order_value":  avg_order_value,
        "impact_lower":     round(impact_lower),
        "impact_point":     round(impact_point),
        "impact_upper":     round(impact_upper),
        "ci_low_pp":        round(ci_low * 100, 3),
        "ci_high_pp":       round(ci_high * 100, 3)
    }


def make_decision(experiment_id="EXP_001"):
    """
    Master decision function — integrates all Phase 3 and 4
    outputs into a single, structured recommendation.
    
    Decision hierarchy (in order of precedence):
    1. INVALIDATED  — SRM detected
    2. DO NOT SHIP  — guardrail violated
    3. DO NOT SHIP  — significant negative result
    4. EXTEND       — inconclusive or underpowered
    5. SHIP         — significant positive, all checks pass
    
    Writes final result to results table in database.
    """
    conn = get_connection()

    print("\n" + "=" * 55)
    print("EXPERIMENT DECISION ENGINE")
    print(f"Experiment: {experiment_id}")
    print(f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 55)

    # ── Run all validation and inference checks ──────────────
    print("\n[1/4] Running SRM detection...")
    srm = check_srm(conn, experiment_id)

    print("\n[2/4] Running power analysis...")
    power = check_power(conn, experiment_id)

    print("\n[3/4] Running guardrail checks...")
    guardrails = check_guardrails(conn, experiment_id)

    print("\n[4/4] Running primary metric inference...")
    inference = run_inference(conn, experiment_id)

    # ── Extract key signals ──────────────────────────────────
    srm_detected     = srm["overall"]["srm_detected"]
    power_achieved   = power["power_achieved"]
    adequately_powered = power["adequately_powered"]

    any_violated = any(
        r["violated"] for r in guardrails.values()
    )
    any_warning  = any(
        r.get("warning", False) for r in guardrails.values()
    )

    p_value     = inference["p_value"]
    significant = inference["significant"]
    lift        = inference["absolute_lift"]
    ci_low      = inference["ci_diff_low"]
    ci_high     = inference["ci_diff_high"]

    # ── Financial impact calculation ─────────────────────────
    financial = calculate_financial_impact(
        conv_control = inference["conv_control"],
        ci_low       = ci_low,
        ci_high      = ci_high,
        daily_traffic    = 10000,
        avg_order_value  = 1200,
        experiment_days  = 14
    )

    # ── Decision logic (hierarchical) ───────────────────────
    if srm_detected:
        recommendation = "INVALIDATED"
        reason = (
            "Sample Ratio Mismatch detected — "
            "randomisation integrity compromised. "
            "Results cannot be trusted. "
            "Investigate assignment pipeline and restart."
        )

    elif any_violated:
        recommendation = "DO NOT SHIP"
        violated = [
            m for m, r in guardrails.items()
            if r["violated"]
        ]
        reason = (
            f"Guardrail violation detected in: "
            f"{', '.join(violated)}. "
            f"Shipping would cause measurable harm "
            f"beyond acceptable thresholds. "
            f"Investigate root cause before proceeding."
        )

    elif significant and lift < 0:
        recommendation = "DO NOT SHIP"
        reason = (
            f"Statistically significant negative result "
            f"(p={p_value:.4f}, lift={lift*100:.3f}pp). "
            f"New algorithm performs worse than control. "
            f"Do not deploy."
        )

    elif not significant or not adequately_powered:
        recommendation = "EXTEND"
        reasons = []
        if not adequately_powered:
            reasons.append(
                f"experiment underpowered "
                f"({power_achieved*100:.1f}% achieved vs 80% planned)"
            )
        if not significant:
            reasons.append(
                f"result inconclusive (p={p_value:.4f})"
            )
        if any_warning:
            reasons.append(
                "guardrail warning requires monitoring"
            )
        reason = (
            f"Insufficient evidence to decide: "
            f"{'; '.join(reasons)}. "
            f"Extend experiment to ~32,604 users per group "
            f"or increase daily traffic allocation."
        )

    else:
        recommendation = "SHIP"
        reason = (
            f"Statistically significant positive result "
            f"(p={p_value:.4f}, lift={lift*100:.3f}pp). "
            f"All guardrails passed. "
            f"Experiment adequately powered. "
            f"Recommend deployment."
        )

    # ── Print final decision report ──────────────────────────
    print("\n" + "=" * 55)
    print("FINAL DECISION REPORT")
    print("=" * 55)

    print(f"\n{'RECOMMENDATION':.<30} {recommendation}")
    print(f"{'P-value':.<30} {p_value:.4f}")
    print(f"{'Significant':.<30} {significant}")
    print(f"{'Absolute lift':.<30} {lift*100:+.3f}pp")
    print(f"{'95% CI':.<30} "
          f"[{ci_low*100:.3f}pp, {ci_high*100:.3f}pp]")
    print(f"{'Achieved power':.<30} "
          f"{power_achieved*100:.1f}%")
    print(f"{'SRM detected':.<30} {srm_detected}")
    print(f"{'Guardrail violated':.<30} {any_violated}")
    print(f"{'Guardrail warning':.<30} {any_warning}")

    print(f"\n{'─'*55}")
    print("FINANCIAL IMPACT ESTIMATE")
    print(f"{'─'*55}")
    print(f"Daily traffic assumption:  "
          f"{financial['daily_traffic']:,} users")
    print(f"Avg order value:           "
          f"₹{financial['avg_order_value']:,}")
    print(f"Annual impact (lower):     "
          f"₹{financial['impact_lower']:,}")
    print(f"Annual impact (point est): "
          f"₹{financial['impact_point']:,}")
    print(f"Annual impact (upper):     "
          f"₹{financial['impact_upper']:,}")

    print(f"\n{'─'*55}")
    print("REASONING")
    print(f"{'─'*55}")
    print(f"{reason}")

    if any_warning:
        print(f"\nACTIVE WARNINGS:")
        for metric, r in guardrails.items():
            if r.get("warning"):
                print(
                    f"  • {metric}: "
                    f"{r['relative_change']*100:.1f}% change "
                    f"(threshold: "
                    f"{r['threshold']*100:.0f}%, "
                    f"p={r['p_value']:.4f} — unconfirmed)"
                )

    print("=" * 55)

    # ── Write to results table ───────────────────────────────
    result_record = {
        "result_id":                str(__import__('uuid').uuid4()),
        "experiment_id":            experiment_id,
        "control_mean":             inference["conv_control"],
        "treatment_mean":           inference["conv_treatment"],
        "lift":                     inference["absolute_lift"],
        "effect_size":              inference["cohens_h"],
        "confidence_interval_lower": ci_low,
        "confidence_interval_upper": ci_high,
        "p_value":                  p_value,
        "sample_size_control":      inference["n_control"],
        "sample_size_treatment":    inference["n_treatment"],
        "srm_detected":             int(srm_detected),
        "guardrail_status":         json.dumps({
            m: {
                "violated": bool(r["violated"]),
                "warning":  bool(r.get("warning", False)),
                "relative_change": float(r["relative_change"])
            }
            for m, r in guardrails.items()
        }),
        "recommendation":           recommendation,
        "recommendation_reason":    reason
    }

    conn.execute("""
        INSERT OR REPLACE INTO results VALUES (
            :result_id, :experiment_id,
            :control_mean, :treatment_mean,
            :lift, :effect_size,
            :confidence_interval_lower,
            :confidence_interval_upper,
            :p_value,
            :sample_size_control,
            :sample_size_treatment,
            :srm_detected, :guardrail_status,
            :recommendation, :recommendation_reason
        )
    """, result_record)
    conn.commit()

    print(f"\nResult written to database "
          f"(result_id: {result_record['result_id'][:8]}...)")

    conn.close()
    return {
        "recommendation": recommendation,
        "reason":         reason,
        "financial":      financial,
        "p_value":        p_value,
        "power_achieved": power_achieved
    }


if __name__ == "__main__":
    result = make_decision()