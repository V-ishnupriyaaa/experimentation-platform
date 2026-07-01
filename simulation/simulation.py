import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import uuid
from datetime import datetime, timedelta
import random

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

if __name__ == "__main__":
    conn = get_connection()
    create_tables(conn)
    df_users = generate_users(conn)
    print(df_users.head())
    print(df_users["spend_tier"].value_counts())
    conn.close()