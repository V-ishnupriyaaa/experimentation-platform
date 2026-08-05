"""
Phase 6A — GenAI Insight Layer (Production Architecture)
Generates plain-English stakeholder memos via Anthropic API.

Note: Requires ANTHROPIC_API_KEY environment variable.
For live demo without API key, see the interactive
artifact in notebooks/genai_demo.html

In production, this would be called after make_decision()
completes, passing the results table record as structured
JSON context to the LLM.
"""

import json
import sqlite3
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / "data" / "experiments.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def build_experiment_context(
    experiment_id="EXP_001",
    simulation_type="large"
):
    """
    Pulls experiment results from database and builds
    structured JSON context for LLM.
    
    simulation_type: "small" (2100 users) or 
                     "large" (50000 users)
    """
    conn = get_connection()

    # ── Pull experiment design ───────────────────────────────
    exp = conn.execute(
        f"SELECT * FROM experiments "
        f"WHERE experiment_id = '{experiment_id}'"
    ).fetchone()

    exp_cols = [
        "experiment_id", "experiment_name",
        "start_date", "end_date", "hypothesis",
        "control_description", "treatment_description",
        "primary_metric", "guardrail_metrics",
        "guardrail_thresholds", "mde", "alpha",
        "power", "status"
    ]
    exp_dict = dict(zip(exp_cols, exp))

    # ── Pull latest result ───────────────────────────────────
    result = conn.execute(
        f"SELECT * FROM results "
        f"WHERE experiment_id = '{experiment_id}' "
        f"ORDER BY rowid DESC LIMIT 1"
    ).fetchone()

    result_cols = [
        "result_id", "experiment_id",
        "control_mean", "treatment_mean",
        "lift", "effect_size",
        "confidence_interval_lower",
        "confidence_interval_upper",
        "p_value", "sample_size_control",
        "sample_size_treatment", "srm_detected",
        "guardrail_status", "recommendation",
        "recommendation_reason"
    ]
    result_dict = dict(zip(result_cols, result))
    conn.close()

    # ── Parse guardrail status ───────────────────────────────
    guardrail_status = json.loads(
        result_dict["guardrail_status"]
    )

    # ── Financial impact (from last run) ─────────────────────
    # These match your large simulation outputs
    if simulation_type == "large":
        financial = {
            "annual_impact_lower":  11944260,
            "annual_impact_point":  24155700,
            "annual_impact_upper":  36367140,
            "currency":             "INR"
        }
        n_users = (
            result_dict["sample_size_control"] +
            result_dict["sample_size_treatment"]
        )
    else:
        financial = {
            "annual_impact_lower":  -99969120,
            "annual_impact_point":  -39715650,
            "annual_impact_upper":  20537820,
            "currency":             "INR"
        }
        n_users = 2100

    # ── Build structured context ─────────────────────────────
    context = {
        "experiment": {
            "name":                  exp_dict["experiment_name"],
            "hypothesis":            exp_dict["hypothesis"],
            "control_description":   exp_dict["control_description"],
            "treatment_description": exp_dict["treatment_description"],
            "primary_metric":        exp_dict["primary_metric"],
            "start_date":            exp_dict["start_date"],
            "end_date":              exp_dict["end_date"],
            "planned_mde_pct":       exp_dict["mde"] * 100,
            "planned_alpha":         exp_dict["alpha"],
            "planned_power_pct":     exp_dict["power"] * 100
        },
        "results": {
            "total_users":           n_users,
            "control_conversion":
                round(result_dict["control_mean"] * 100, 3),
            "treatment_conversion":
                round(result_dict["treatment_mean"] * 100, 3),
            "absolute_lift_pp":
                round(result_dict["lift"] * 100, 3),
            "relative_lift_pct":     round(
                result_dict["lift"] /
                result_dict["control_mean"] * 100, 2
            ),
            "confidence_interval": {
                "lower_pp": round(
                    result_dict[
                        "confidence_interval_lower"
                    ] * 100, 3
                ),
                "upper_pp": round(
                    result_dict[
                        "confidence_interval_upper"
                    ] * 100, 3
                )
            },
            "p_value":           result_dict["p_value"],
            "statistically_significant":
                result_dict["p_value"] < 0.05,
            "srm_detected":      bool(result_dict["srm_detected"]),
            "recommendation":    result_dict["recommendation"],
            "recommendation_reason":
                result_dict["recommendation_reason"]
        },
        "validation": {
            "srm_detected": bool(result_dict["srm_detected"]),
            "guardrail_results": guardrail_status
        },
        "financial_impact": financial,
        "generated_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        )
    }

    return context


def generate_stakeholder_memo(context):
    """
    Calls LLM API with structured experiment context.
    Prompts for a specific, data-grounded stakeholder memo
    — not a generic summary.
    """

    system_prompt = """You are a Senior Experimentation 
Scientist at a top e-commerce company writing a decision 
memo for a non-technical Product Manager.

Your memo must:
1. Reference specific numbers from the experiment data provided
2. Explain what the recommendation means in business terms
3. Flag any risks or warnings clearly
4. Be written in plain English — no statistical jargon
5. Be concise — maximum 300 words
6. Follow this exact structure:
   - EXPERIMENT SUMMARY (2-3 sentences)
   - KEY FINDINGS (3-4 bullet points with specific numbers)
   - RISKS & WARNINGS (if any)
   - RECOMMENDATION & NEXT STEPS (2-3 sentences)

Never use terms like p-value, statistical significance, 
or confidence interval without immediately explaining 
what they mean in plain English."""

    user_prompt = f"""Write a stakeholder decision memo 
for the following experiment. Use the exact numbers 
provided — do not generalise or make up values.

Experiment Data:
{json.dumps(context, indent=2)}

Write the memo now:"""

    # ── API call ─────────────────────────────────────────────
    payload = json.dumps({
        "model":      "claude-sonnet-4-6",
        "max_tokens": 1000,
        "system":     system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            memo = data["content"][0]["text"]
            return memo

    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"API Error {e.code}: {error_body}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def run_genai_layer(simulation_type="large"):
    """
    Main function — builds context, calls LLM,
    prints stakeholder memo.
    """
    print("=" * 55)
    print("GENAI INSIGHT LAYER")
    print(f"Experiment: EXP_001 ({simulation_type} simulation)")
    print("=" * 55)

    # ── Build structured context ─────────────────────────────
    print("\n[1/2] Building experiment context...")
    context = build_experiment_context(
        simulation_type=simulation_type
    )

    print(f"      Experiment: "
          f"{context['experiment']['name']}")
    print(f"      Users: "
          f"{context['results']['total_users']:,}")
    print(f"      Recommendation: "
          f"{context['results']['recommendation']}")

    # ── Generate memo ────────────────────────────────────────
    print("\n[2/2] Generating stakeholder memo...")
    memo = generate_stakeholder_memo(context)

    if memo:
        print("\n" + "=" * 55)
        print("STAKEHOLDER DECISION MEMO")
        print("=" * 55)
        print(memo)
        print("=" * 55)

        # ── Save memo to file ────────────────────────────────
        output_path = (
            Path(__file__).parent.parent /
            "data" /
            f"memo_{simulation_type}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        )
        with open(output_path, "w") as f:
            f.write(f"EXPERIMENT: "
                    f"{context['experiment']['name']}\n")
            f.write(f"Generated: "
                    f"{context['generated_at']}\n")
            f.write("=" * 55 + "\n\n")
            f.write(memo)

        print(f"\n✓ Memo saved to {output_path.name}")
    else:
        print("✗ Memo generation failed — check API key")

    return memo


if __name__ == "__main__":
    # Generate memo for large simulation (SHIP result)
    memo_large = run_genai_layer(simulation_type="large")