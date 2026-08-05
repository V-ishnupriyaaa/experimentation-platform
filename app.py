import streamlit as st
import anthropic
import json
from pathlib import Path

st.set_page_config(
    page_title="Experimentation Decision Platform",
    page_icon="🧪",
    layout="wide"
)

st.title("Intelligent Experimentation & Decision Platform")
st.caption("Built by Vishnu | MSc Data Analytics, Christ University")

# ── Presets ──────────────────────────────────────────────────────
PRESETS = {
    "Small simulation — EXTEND (2,100 users)": {
        "exp_name": "Search Ranking Algorithm V2 Test",
        "metric": "Conversion rate",
        "mde": 0.5, "alpha": 0.05, "power": 80.0,
        "users": 2100, "cc": 3.12, "ct": 2.21,
        "pval": 0.1972, "ci_low": -2.282, "ci_high": 0.469,
        "ap": 10.5, "imp_low": -99969120, "imp_high": 20537820,
        "srm": False,
        "g1": "Warning", "g2": "Passed", "g3": "Warning"
    },
    "Large simulation — SHIP (50,000 users)": {
        "exp_name": "Search Ranking Algorithm V2 Test",
        "metric": "Conversion rate",
        "mde": 0.5, "alpha": 0.05, "power": 80.0,
        "users": 50000, "cc": 2.32, "ct": 2.87,
        "pval": 0.0001, "ci_low": 0.273, "ci_high": 0.830,
        "ap": 93.9, "imp_low": 11944260, "imp_high": 36367140,
        "srm": False,
        "g1": "Passed", "g2": "Passed", "g3": "Passed"
    },
    "Real dataset — DO NOT SHIP (290,000 users)": {
        "exp_name": "Landing Page A/B Test (Kaggle Dataset)",
        "metric": "Conversion rate",
        "mde": 0.5, "alpha": 0.05, "power": 80.0,
        "users": 290584, "cc": 12.04, "ct": 11.88,
        "pval": 0.1899, "ci_low": -0.394, "ci_high": 0.078,
        "ap": 98.6, "imp_low": -17000000, "imp_high": 3400000,
        "srm": False,
        "g1": "Passed", "g2": "Passed", "g3": "Passed"
    }
}

# ── Decision logic ────────────────────────────────────────────────
def get_recommendation(srm, g1, g2, g3, pval, ap, cc, ct):
    violated = any(g == "Violated" for g in [g1, g2, g3])
    sig = pval < 0.05
    if srm:
        return "INVALIDATED"
    if violated:
        return "DO NOT SHIP"
    if sig and ct < cc:
        return "DO NOT SHIP"
    if not sig or ap < 80:
        return "EXTEND"
    return "SHIP"

REC_COLORS = {
    "SHIP":        ("✅", "green"),
    "EXTEND":      ("⏳", "orange"),
    "DO NOT SHIP": ("🚫", "red"),
    "INVALIDATED": ("⚠️", "red")
}

# ── Sidebar — preset loader ───────────────────────────────────────
st.sidebar.header("Load experiment preset")
preset_choice = st.sidebar.radio(
    "Select experiment:",
    list(PRESETS.keys()),
    index=0
)
p = PRESETS[preset_choice]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About this platform**\n\n"
    "Built from first principles — no AutoML, no black boxes. "
    "Each component (SRM detection, power analysis, guardrail "
    "monitoring, CUPED variance reduction) was reasoned through "
    "and implemented independently.\n\n"
    "**Stack:** Python, SQLite, scipy, statsmodels, Streamlit, Claude API\n\n"
    "**GitHub:** [V-ishnupriyaaa/experimentation-platform]"
    "(https://github.com/V-ishnupriyaaa/experimentation-platform)"
)

# ── Main layout ───────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Experiment design")

    exp_name = st.text_input("Experiment name", value=p["exp_name"])
    metric   = st.text_input("Primary metric", value=p["metric"])

    c1, c2, c3 = st.columns(3)
    mde   = c1.number_input("MDE (%)", value=p["mde"], step=0.1)
    alpha = c2.number_input("Alpha", value=p["alpha"], step=0.01)
    power = c3.number_input("Planned power (%)", value=p["power"])

    st.subheader("Experiment results")

    users = st.number_input(
        "Total users", value=p["users"], step=100
    )

    c1, c2 = st.columns(2)
    cc = c1.number_input(
        "Control conversion (%)", value=p["cc"], step=0.01
    )
    ct = c2.number_input(
        "Treatment conversion (%)", value=p["ct"], step=0.01
    )

    c1, c2, c3 = st.columns(3)
    pval   = c1.number_input("P-value", value=p["pval"], step=0.0001, format="%.4f")
    ci_low = c2.number_input("CI lower (pp)", value=p["ci_low"], step=0.001, format="%.3f")
    ci_high= c3.number_input("CI upper (pp)", value=p["ci_high"], step=0.001, format="%.3f")

    c1, c2, c3 = st.columns(3)
    ap      = c1.number_input("Achieved power (%)", value=p["ap"], step=0.1)
    imp_low = c2.number_input("Annual impact lower (₹)", value=float(p["imp_low"]))
    imp_high= c3.number_input("Annual impact upper (₹)", value=float(p["imp_high"]))

    st.subheader("Validation status")

    srm = st.selectbox(
        "SRM detected",
        [False, True],
        index=0 if not p["srm"] else 1,
        format_func=lambda x: "Yes — flagged ⚠️" if x else "No — clean ✓"
    )

    c1, c2, c3 = st.columns(3)
    g1 = c1.selectbox(
        "Cart-to-purchase",
        ["Passed", "Warning", "Violated"],
        index=["Passed","Warning","Violated"].index(p["g1"])
    )
    g2 = c2.selectbox(
        "Seller diversity",
        ["Passed", "Warning", "Violated"],
        index=["Passed","Warning","Violated"].index(p["g2"])
    )
    g3 = c3.selectbox(
        "Retention rate",
        ["Passed", "Warning", "Violated"],
        index=["Passed","Warning","Violated"].index(p["g3"])
    )

with col2:
    st.subheader("Decision summary")

    rec = get_recommendation(srm, g1, g2, g3, pval, ap, cc, ct)
    icon, color = REC_COLORS[rec]
    lift = ct - cc

    st.markdown(f"### {icon} Recommendation: :{color}[**{rec}**]")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Absolute lift", f"{lift:+.3f}pp")
    m2.metric("P-value", f"{pval:.4f}")
    m3.metric("Achieved power", f"{ap:.1f}%")
    m4.metric("Significant", "Yes" if pval < 0.05 else "No")

    st.markdown("---")

    # ── Financial impact ─────────────────────────────────────
    st.markdown("**Financial impact estimate**")
    fi1, fi2 = st.columns(2)
    fi1.metric(
        "Annual impact (lower)",
        f"₹{imp_low:,.0f}",
        delta="conservative case"
    )
    fi2.metric(
        "Annual impact (upper)",
        f"₹{imp_high:,.0f}",
        delta="optimistic case"
    )

    if ci_low < 0 < ci_high:
        st.warning(
            "Confidence interval crosses zero — "
            "true effect direction is uncertain."
        )
    elif ci_low > 0:
        st.success(
            "Confidence interval entirely positive — "
            "effect direction is clear."
        )
    else:
        st.error(
            "Confidence interval entirely negative — "
            "treatment is likely worse than control."
        )

    st.markdown("---")

    # ── Guardrail summary ────────────────────────────────────
    st.markdown("**Guardrail status**")
    STATUS_ICONS = {
        "Passed": "✅", "Warning": "⚠️", "Violated": "🚫"
    }
    gc1, gc2, gc3 = st.columns(3)
    gc1.markdown(
        f"{STATUS_ICONS[g1]} **Cart-to-purchase**\n\n{g1}"
    )
    gc2.markdown(
        f"{STATUS_ICONS[g2]} **Seller diversity**\n\n{g2}"
    )
    gc3.markdown(
        f"{STATUS_ICONS[g3]} **Retention rate**\n\n{g3}"
    )

    st.markdown("---")

    # ── GenAI memo ───────────────────────────────────────────
    st.subheader("AI stakeholder memo")

    api_key = st.text_input(
        "Anthropic API key (optional — needed for memo generation)",
        type="password",
        placeholder="sk-ant-..."
    )

    if st.button("Generate stakeholder memo ↗", type="primary"):
        if not api_key:
            st.warning(
                "Enter an Anthropic API key above to generate "
                "a memo. You can get one at console.anthropic.com"
            )
        else:
            context = {
                "experiment_name": exp_name,
                "primary_metric":  metric,
                "total_users":     int(users),
                "control_conversion_pct":   cc,
                "treatment_conversion_pct": ct,
                "absolute_lift_pp":  round(lift, 3),
                "p_value":           pval,
                "statistically_significant": pval < 0.05,
                "ci_lower_pp":       ci_low,
                "ci_upper_pp":       ci_high,
                "achieved_power_pct": ap,
                "srm_detected":      srm,
                "guardrails": {
                    "cart_to_purchase": g1,
                    "seller_diversity": g2,
                    "retention_rate":   g3
                },
                "recommendation": rec,
                "financial_impact": {
                    "annual_lower_inr": int(imp_low),
                    "annual_upper_inr": int(imp_high)
                }
            }

            system_prompt = """You are a Senior Experimentation Scientist writing a decision memo for a non-technical Product Manager. Use plain English — no jargon. Reference exact numbers. Structure:

EXPERIMENT SUMMARY (2-3 sentences)
KEY FINDINGS (3-4 bullet points with specific numbers)
RISKS AND WARNINGS (if any)
RECOMMENDATION AND NEXT STEPS (2-3 sentences)

Maximum 280 words."""

            user_prompt = f"""Write a stakeholder decision memo using exactly these experiment results:

{json.dumps(context, indent=2)}"""

            with st.spinner("Generating memo..."):
                try:
                    client = anthropic.Anthropic(api_key=api_key)
                    message = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=1000,
                        system=system_prompt,
                        messages=[{
                            "role": "user",
                            "content": user_prompt
                        }]
                    )
                    memo = message.content[0].text
                    st.markdown(memo)

                    st.download_button(
                        "Download memo as .txt",
                        data=memo,
                        file_name=f"memo_{rec.lower().replace(' ','_')}.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"API error: {e}")
    else:
        st.info(
            "Click 'Generate stakeholder memo' above to produce "
            "a plain-English decision memo for this experiment."
        )