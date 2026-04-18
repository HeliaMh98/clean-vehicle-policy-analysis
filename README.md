# Clean Vehicle Policy Analysis Pipeline

**Causal Evaluation of U.S. State Clean Vehicle Policies Using LLM-Assisted Classification and Doubly Robust Machine Learning**

---

## Overview

This repository contains the research pipeline developed to:

1. **Classify** a large corpus of U.S. state-level clean vehicle policies using large language models (LLMs), and  
2. **Estimate causal effects** of those policy mechanisms on alternative fuel vehicle (AFV) adoption shares using Doubly Robust Machine Learning (DR/DML).

The project targets five vehicle technology types — Battery Electric Vehicles (BEV), Plug-in Hybrid Electric Vehicles (PHEV), Hybrid Electric Vehicles (HEV), Hydrogen Fuel Cell Vehicles, and Propane/LPG — across 51 U.S. states and 8 years of panel data.

This work contributes to ongoing research on transportation decarbonization policy effectiveness and is intended to support data-driven policymaking for zero-emission vehicle transitions.

---

## Research Questions

- Which state-level policy mechanisms (e.g., access & convenience benefits, operating cost incentives, weight/capacity exemptions) have a statistically significant causal effect on AFV adoption?
- Does the magnitude of these effects vary by vehicle technology type?
- What state-level characteristics (population density, GDP, electricity price) moderate policy effectiveness?

---

## Data Sources

| Dataset | Source | Description |
|---|---|---|
| Alternative Fuel Vehicle Policy Database | [AFDC / DOE](https://afdc.energy.gov/laws) | State-year panel of AFV-related policies, 2013–2020 |
| Vehicle Registration Data | [AFDC / FHWA](https://afdc.energy.gov/vehicle-registration) | State-year AFV adoption shares by fuel type |
| State Socioeconomic Covariates | [U.S. Census](https://data.census.gov) / [BLS](https://www.bls.gov/data/) / [EIA](https://www.eia.gov/electricity/data.php) | GDP per capita, population density, electricity prices, median age, employment rate |

> **Data availability:** Raw data files are not included in this repository. All datasets are publicly available and can be downloaded directly from the sources linked above. See [`data/README.md`](data/README.md) for exact download instructions, file naming conventions, and the panel construction procedure needed to reproduce the analysis.

---

## Methods

### Stage 1 — LLM-Assisted Policy Classification (`src/llm_classification/`)

Raw policy text is passed through a dual-LLM classification pipeline using:
- **GPT-4.1-mini** (OpenAI) and **Claude Haiku** (Anthropic) in parallel
- A structured prompt schema that assigns each policy to one of six mechanism clusters:
  - Upfront Cost Reduction
  - Operating Cost Incentives
  - Access & Convenience Benefits
  - Weight & Capacity Advantages
  - Regulatory Relief
  - Restrictions & Penalties
- Cross-provider agreement rate computed to validate classification robustness
- Disagreements are flagged for human review

### Stage 2 — Causal Inference via DR/DML (`src/drdml/`)

For each vehicle type × policy mechanism combination:
- **Treatment:** Binary indicator of whether a given policy mechanism is active in a state-year
- **Outcome:** AFV adoption share (%) for the corresponding vehicle type
- **Method:** Doubly Robust Machine Learning with 5-fold cross-fitting
  - Outcome model: Random Forest (scikit-learn)
  - Propensity model: Gradient Boosting Classifier
  - Final estimator: OLS on residuals with cluster-robust standard errors (state-level clustering)
- **Heterogeneity analysis:** Conditional Average Treatment Effects (CATEs) estimated across state-level moderators
- Results visualized as forest plots, event study plots, and state-level choropleth maps

---

## Repository Structure

```
clean-vehicle-policy-analysis/
│
├── data/
│   └── README.md                   # Data sources, download instructions, panel construction
│
│
├── src/
│   ├── llm_classification/
│   │   ├── classify_policies.py    # Main LLM classification pipeline
│   │   └── prompt_templates.py     # Structured prompts for GPT and Claude
│   │   
│   │
│   └── drdml/
│       └── drdml_estimator.py      # DR/DML estimation with cross-fitting
│
├── figures/                        # Output figures
├── outputs/                        # ATE tables, CATE results, classification outputs
│
├── requirements.txt
└── README.md
```

---

## Key Findings from the authors' analysis

| Vehicle Type | Policy Mechanism | ATE (pp) | 95% CI | Significant |
|---|---|---|---|---|
| HEV | Access & Convenience Benefits | +3.62 | [0.94, 6.30] | ✓ |
| BEV | Weight & Capacity Advantages | +2.47 | [0.88, 4.05] | ✓ |
| BEV | Access & Convenience Benefits | +1.60 | [0.02, 3.18] | ✓ |

*ATE = Average Treatment Effect in percentage points. Estimation uses cluster-robust standard errors (state-level).*

---

## Setup & Reproduction

### Requirements

```bash
git clone https://github.com/HeliaMh98/clean-vehicle-policy-analysis.git
cd clean-vehicle-policy-analysis
pip install -r requirements.txt
```

### API Keys

The LLM classification pipeline requires API keys for OpenAI and Anthropic. Set these as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Running the Pipeline

First, download and prepare data by following the instructions in [`data/README.md`](data/README.md). Then run the scripts directly:

```bash
# LLM classification only
python src/llm_classification/classify_policies.py \
    --input path/to/your/policies.csv \
    --output outputs/classified_policies.csv

# DR/DML estimation
python src/drdml/drdml_estimator.py \
    --input path/to/your/panel_data.csv \
    --output outputs/ate_results.csv \
    --plot
```

---

## Dependencies

See `requirements.txt` for the full list. Core packages:

- `pandas`, `numpy`, `scipy` — data manipulation and statistics
- `scikit-learn` — ML models for DR/DML nuisance functions
- `econml` — causal inference framework
- `openai`, `anthropic` — LLM API clients
- `matplotlib`, `seaborn`, `plotly`, `geopandas` — visualization

