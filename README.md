# Intelligent Experimentation & Decision Platform

> Built to answer one question most A/B testing tools ignore:
> **not just "is this result significant?" but "should we ship it — and what does getting that decision wrong cost?"**

---

## The Business Problem

Every product team runs experiments. Most stop at a p-value.

But a p-value doesn't tell you:
- Whether your randomisation was trustworthy
- Whether something important broke while your primary metric improved
- Whether you had enough data to trust the result at all
- What the financial cost of a wrong decision looks like

This platform answers all four — before recommending ship, extend, or do not ship.

---

## What This Platform Does

Most experimentation tools stop at a p-value. This platform 
treats statistical output as the beginning of a decision 
process, not the end.

The validation layer runs before any inference — checking 
randomisation integrity (SRM), statistical power, and 
guardrail metrics. Only when all three pass does the 
platform proceed to inference and recommendation.

The decision layer translates statistical output into one 
of four structured recommendations — SHIP, EXTEND, 
DO NOT SHIP, or INVALIDATED — each with a plain-English 
reason and a financial impact range.

---

## Three Experiments, Three Recommendations

The platform was tested across three scenarios — same statistical engine, different data:

| Experiment | Users | Achieved Power | P-value | Recommendation | Reason |
|---|---|---|---|---|---|
| Small simulation | 2,100 | 10.5% | 0.1972 | **EXTEND** | Critically underpowered — cannot trust results |
| Large simulation | 50,000 | 93.9% | 0.0001 | **SHIP** | Significant positive result, all guardrails passed |
| Real dataset (Kaggle) | 290,584 | 98.6% | 0.1899 | **DO NOT SHIP** | Highly powered, no true effect exists |

The contrast between experiments 1 and 3 is the most important finding:
- Experiment 1 was inconclusive because it **lacked statistical power**
- Experiment 3 was inconclusive because the **effect genuinely doesn't exist**

A platform that only reports p-values cannot distinguish these two situations.
This one can.

---

## Platform Architecture
Simulation Layer → Validation Layer → Inference Engine → Decision Layer
───────────────── ────────────────── ───────────────── ──────────────
simulation.py inference.py inference.py decision.py

5-table SQLite schema • SRM detection • Two-proportion z-test • 4-state logic
Pareto user distribution (chi-square test) • Post-hoc power analysis • Financial impact
Stratified randomisation • Guardrail monitoring • CUPED variance reduction • AI memo (app.py)
Heterogeneous treatment (Bonferroni corrected) • Peeking problem demo
effects by spend tier • Power analysis • Confidence intervals


---

## Key Technical Decisions — And Why

**Stratified randomisation by spend tier**
Simple coin-flip randomisation risks Simpson's Paradox — heavy spenders accidentally concentrated in one group can create a fake effect. Stratifying by tier guarantees each segment splits 50/50, eliminating this source of distortion before the experiment runs.

**Dual metric approach for zero-inflated revenue**
97.3% of users had zero revenue during the experiment. A t-test on this distribution violates normality assumptions. The platform uses conversion rate (z-test on binary outcome) and revenue-per-purchaser (t-test on non-zero values only) separately — with survivorship bias explicitly documented for the second metric.

**Three-state guardrail system**
Standard guardrail systems flag PASSED or VIOLATED. This platform adds a third state — WARNING — for cases where a threshold is breached but statistical significance isn't reached due to low power. Silently passing an 18.6% retention drop as "clean" would be dangerous; flagging it as an unconfirmed warning is honest.

**Post-hoc power analysis before inference**
The platform calculates achieved power using actual sample size and observed variance — not planned assumptions. When achieved power is below 80%, the decision engine automatically shifts toward EXTEND regardless of the p-value, because an inconclusive result from an underpowered experiment means "we don't know yet," not "it doesn't work."

**Confidence interval-based financial impact**
The platform reports annual revenue impact at the lower and upper CI bounds — not just the point estimate. An experiment showing ₹-9.99 crore to ₹+2.05 crore looks very different from one showing ₹+1.19 crore to ₹+3.64 crore, even if both have similar p-values. The CI range is what drives the ship/extend decision.

---

## EDA Findings

Before building the inference engine, exploratory analysis on the simulated data revealed:

- **97.3% zero-inflation** in revenue per user — confirmed the dual metric approach was statistically necessary, not just a design preference
- **Simpson's Paradox** independently discovered: treatment won in 3 of 4 spend tiers but lost in the blended aggregate due to sampling variability in the largest segment. This finding directly shaped the platform's requirement to always report segment-level results alongside aggregate outputs.
- **Funnel shape validated**: 2,100 searches → 1,042 clicks (49.6%) → 256 cart additions (12.2%) → 56 purchases (2.7%) — consistent with 2026 industry benchmarks

---

## Concepts Implemented From First Principles

Every statistical component was reasoned through and implemented without black-box libraries where possible:

- Sample Ratio Mismatch detection (chi-square test on assignment counts)
- Post-hoc power analysis (flipped sample size formula to solve for achieved power)
- Bonferroni multiple testing correction across three simultaneous guardrail tests
- CUPED variance reduction (regression-based covariate adjustment using pre-experiment revenue)
- Peeking problem demonstration (10,000-simulation Monte Carlo showing 5% → 14.3% false positive inflation)
- Two-proportion z-test with confidence intervals on absolute lift
- Cohen's h effect size for binary metrics
- Heterogeneous treatment effects by user segment

---

## Real Dataset Validation

The platform was run on a real public A/B test dataset (Kaggle, 294,478 rows, e-commerce landing page test) to confirm it generalises beyond simulation.

Key finding: the platform independently produced p=0.1899 (DO NOT SHIP) — consistent with published community analyses of the same dataset. The real data also revealed 3,893 group/page mismatches (users assigned to one group but served the wrong page) — a real data quality issue absent from clean simulated data that the platform's cleaning layer detected and removed before analysis.

---

### Project Structure

```
experimentation_platform/
│
├── simulation/
│   └── simulation.py          # Data generation
│
├── inference/
│   └── inference.py           # SRM, power analysis,
│                              # guardrails, z-test, CUPED
│
├── decision/
│   ├── decision.py            # Decision integration
│   └── genai_layer.py         # Production GenAI pipeline
│                              # (requires API key)
│
├── notebooks/
│   ├── EDA.ipynb              # Exploratory analysis
│   └── real_dataset_validation.py
│
├── data/
│   ├── experiments.db         # SQLite database (5 tables)
│   └── real_dataset/
│       └── ab_data.csv        # Kaggle A/B test dataset
│
├── app.py                     # Streamlit interactive demo
├── requirements.txt
└── README.md
```

---

## How To Run

```bash
# Clone the repository
git clone https://github.com/V-ishnupriyaaa/experimentation-platform.git
cd experimentation-platform

# Install dependencies
pip install -r requirements.txt

# Generate simulation data
python simulation/simulation.py

# Run full validation + inference + decision pipeline
python decision/decision.py

# Launch interactive Streamlit app
streamlit run app.py
```

---

## Stack

Python, SQLite, pandas, numpy, scipy, statsmodels, matplotlib, seaborn, Streamlit, Anthropic Claude API

---


## Background

Built during MSc Data Analytics (Christ University, 
Bangalore, 2026). Design inspired by published 
experimentation approaches from Airbnb, Microsoft, 
Netflix, and Booking.com. Simulation parameters 
grounded in 2026 public industry benchmarks.

---

## Key References

## References and Inspiration

- Kohavi, R. et al. — *Trustworthy Online Controlled 
  Experiments* (Cambridge University Press) — general 
  framework for experiment design and SRM detection
- Deng, A. et al. (Microsoft Research) — *Improving the 
  Sensitivity of Online Controlled Experiments by Utilizing 
  Pre-Experiment Data* — CUPED implementation
- Optimizely Engineering Blog — sequential testing and 
  the peeking problem
- Dynamic Yield / Contentsquare 2026 Benchmarks — 
  e-commerce conversion rate parameters for simulation

## Limitations

**Simulated data:** The primary dataset is simulated, 
not from a live production system. Simulation parameters 
are grounded in 2026 public benchmarks but cannot capture 
every real-world edge case (bot traffic, network effects, 
extreme seasonality).

**Guardrail proxies:** Two of three guardrail metrics 
are proxies derived from event data rather than direct 
measurements. Seller diversity is simulated; cart-to-
purchase rate proxies product quality. In production, 
these would come from dedicated data pipelines.

**CUPED covariate:** Pre-experiment revenue produced 
only 0.2% variance reduction due to 50% zero-history 
users. In production with richer behavioural history, 
CUPED typically produces 10–50% variance reduction.

**Scale:** The platform is designed for single-experiment 
analysis. Production experimentation platforms handle 
thousands of concurrent experiments with interaction 
effect detection — not implemented here.
