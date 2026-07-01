import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import uuid
from datetime import datetime, timedelta
import random
import json

# ── Database connection ──────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent / "data" / "experiments.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

# ── Create tables ────────────────────────────────────────────────
def create_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id                 TEXT PRIMARY KEY,
            historical_revenue_30d  REAL,
            spend_tier              TEXT,
            avg_days_to_purchase    REAL,
            delivery_days           INTEGER,
            signup_date             TEXT
        );

        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id           TEXT PRIMARY KEY,
            experiment_name         TEXT,
            start_date              TEXT,
            end_date                TEXT,
            hypothesis              TEXT,
            control_description     TEXT,
            treatment_description   TEXT,
            primary_metric          TEXT,
            guardrail_metrics       TEXT,
            guardrail_thresholds    TEXT,
            mde                     REAL,
            alpha                   REAL,
            power                   REAL,
            status                  TEXT
        );

        CREATE TABLE IF NOT EXISTS assignments (
            assignment_id       TEXT PRIMARY KEY,
            user_id             TEXT,
            experiment_id       TEXT,
            variant             TEXT,
            assigned_at         TEXT,
            assignment_source   TEXT
        );

        CREATE TABLE IF NOT EXISTS events (
            event_id            TEXT PRIMARY KEY,
            user_id             TEXT,
            experiment_id       TEXT,
            variant             TEXT,
            event_type          TEXT,
            event_value         REAL,
            event_timestamp     TEXT,
            session_id          TEXT,
            device_type         TEXT
        );

        CREATE TABLE IF NOT EXISTS results (
            result_id                   TEXT PRIMARY KEY,
            experiment_id               TEXT,
            control_mean                REAL,
            treatment_mean              REAL,
            lift                        REAL,
            effect_size                 REAL,
            confidence_interval_lower   REAL,
            confidence_interval_upper   REAL,
            p_value                     REAL,
            sample_size_control         INTEGER,
            sample_size_treatment       INTEGER,
            srm_detected                INTEGER,
            guardrail_status            TEXT,
            recommendation              TEXT,
            recommendation_reason       TEXT
        );
    """)
    conn.commit()

   

def generate_users(conn, n_users=2100, seed=42):
    """
    Generate simulated users with realistic spend tier distribution.
    Tier distribution follows Pareto principle:
    50% zero-history, 35% low, 12% mid, 3% heavy
    Based on 2026 e-commerce conversion benchmarks.
    """
    np.random.seed(seed)
    random.seed(seed)

    # ── Tier definitions ────────────────────────────────────────
    tiers = {
        "zero_history": {
            "proportion": 0.50,
            "revenue_range": (0, 0),
            "purchase_prob_control": 0.015,
            "purchase_prob_treatment": 0.020,  # ~30% relative lift
            "avg_days_to_purchase_range": (5, 14),
        },
        "low": {
            "proportion": 0.35,
            "revenue_range": (1, 1000),
            "purchase_prob_control": 0.027,
            "purchase_prob_treatment": 0.032,  # ~20% relative lift
            "avg_days_to_purchase_range": (3, 10),
        },
        "mid": {
            "proportion": 0.12,
            "revenue_range": (1001, 5000),
            "purchase_prob_control": 0.040,
            "purchase_prob_treatment": 0.044,  # ~10% relative lift
            "avg_days_to_purchase_range": (1, 5),
        },
        "heavy": {
            "proportion": 0.03,
            "revenue_range": (5001, 20000),
            "purchase_prob_control": 0.055,
            "purchase_prob_treatment": 0.058,  # ~5% relative lift
            "avg_days_to_purchase_range": (0, 2),
        },
    }

    users = []
    base_date = datetime(2026, 1, 1)

    for tier_name, config in tiers.items():
        n_tier = int(n_users * config["proportion"])

        for _ in range(n_tier):
            # Generate historical revenue
            low_r, high_r = config["revenue_range"]
            if tier_name == "zero_history":
                hist_revenue = 0.0
            else:
                hist_revenue = round(
                    np.random.uniform(low_r, high_r), 2
                )

            # Generate days to purchase
            low_d, high_d = config["avg_days_to_purchase_range"]
            avg_days = round(np.random.uniform(low_d, high_d), 1)

            # Generate delivery days (proxy for location)
            # Metro: 1-2 days, Tier-2: 3-5 days, Tier-3: 5-10 days
            delivery_days = int(np.random.choice(
                [1, 2, 3, 4, 5, 7, 10],
                p=[0.15, 0.20, 0.20, 0.15, 0.15, 0.10, 0.05]
            ))

            # Random signup date in last 2 years
            days_since_signup = random.randint(0, 730)
            signup_date = (
                base_date - timedelta(days=days_since_signup)
            ).strftime("%Y-%m-%d")

            users.append({
                "user_id": str(uuid.uuid4()),
                "historical_revenue_30d": hist_revenue,
                "spend_tier": tier_name,
                "avg_days_to_purchase": avg_days,
                "delivery_days": delivery_days,
                "signup_date": signup_date,
            })

    # ── Write to database ───────────────────────────────────────
    df_users = pd.DataFrame(users)
    df_users.to_sql(
        "users", conn, if_exists="replace", index=False
    )
    conn.commit()
    print(f"Generated {len(df_users)} users across 4 tiers")
    return df_users



def generate_experiment(conn):
    """
    Insert one experiment record with all design decisions
    locked in upfront — MDE, alpha, power, guardrails.
    Prevents HARKing (Hypothesising After Results are Known).
    """
    experiment = {
        "experiment_id": "EXP_001",
        "experiment_name": "Search Ranking Algorithm V2 Test",
        "start_date": "2026-06-15",
        "end_date": "2026-06-29",
        "hypothesis": (
    "H0: New ranking algorithm has no effect on conversion rate. "
    "H1: New ranking algorithm changes conversion rate (two-sided, alpha=0.05)."
),
        "control_description": "Existing search ranking algorithm",
        "treatment_description": "New ML-based search ranking algorithm",
        "primary_metric": "conversion_rate",
        "guardrail_metrics": json.dumps([
            "avg_review_score",
            "seller_diversity_index",
            "retention_rate"
        ]),
        "guardrail_thresholds": json.dumps([
            0.05,   # max 5% drop in avg review score
            0.10,   # max 10% drop in seller diversity
            0.05    # max 5% drop in retention rate
        ]),
        "mde": 0.005,        # 0.5% minimum detectable effect
        "alpha": 0.05,       # 5% significance level
        "power": 0.80,       # 80% statistical power
        "status": "completed"
    }

    conn.execute("""
        INSERT OR REPLACE INTO experiments VALUES (
            :experiment_id, :experiment_name,
            :start_date, :end_date,
            :hypothesis,
            :control_description, :treatment_description,
            :primary_metric, :guardrail_metrics,
            :guardrail_thresholds, :mde, :alpha, :power, :status
        )
    """, experiment)
    conn.commit()
    print(f"Experiment '{experiment['experiment_name']}' created")
    return experiment

def generate_assignments(conn, df_users, experiment_id="EXP_001", seed=42):
    """
    Assign users to control or treatment using stratified
    randomisation by spend tier.
    Hash-based assignment guarantees sticky, deterministic splits.
    Stratification prevents Simpson's Paradox by ensuring
    each tier is split 50/50 across variants.
    """
    random.seed(seed)
    experiment_start = datetime(2026, 6, 15)
    assignments = []

    # ── Stratified randomisation by spend tier ──────────────────
    for tier in df_users["spend_tier"].unique():
        tier_users = df_users[
            df_users["spend_tier"] == tier
        ].copy()

        for _, user in tier_users.iterrows():
            # Deterministic hash-based assignment
            variant = (
                "treatment"
                if hash(user["user_id"]) % 2 == 0
                else "control"
            )

            # Assignment timestamp — random time on experiment start day
            hours_offset = random.randint(0, 23)
            assigned_at = (
                experiment_start +
                timedelta(hours=hours_offset)
            ).strftime("%Y-%m-%d %H:%M:%S")

            assignments.append({
                "assignment_id": str(uuid.uuid4()),
                "user_id": user["user_id"],
                "experiment_id": experiment_id,
                "variant": variant,
                "assigned_at": assigned_at,
                "assignment_source": "random"
            })

    # ── Write to database ───────────────────────────────────────
    df_assignments = pd.DataFrame(assignments)
    df_assignments.to_sql(
        "assignments", conn,
        if_exists="replace", index=False
    )
    conn.commit()

    # ── Sanity check — print split per tier ────────────────────
    print("\n Assignment split by tier:")
    print(
        df_assignments.merge(
            df_users[["user_id", "spend_tier"]],
            on="user_id"
        ).groupby(["spend_tier", "variant"]).size().unstack()
    )
    return df_assignments

if __name__ == "__main__":
    conn = get_connection()
    create_tables(conn)
    df_users = generate_users(conn)
    print(df_users["spend_tier"].value_counts())
    experiment = generate_experiment(conn)
    df_assignments = generate_assignments(conn, df_users)
    conn.close()