"""
drdml_estimator.py
------------------
Doubly Robust Machine Learning (DR/DML) estimation of Average Treatment Effects (ATEs)
for U.S. state clean vehicle policies on alternative fuel vehicle adoption shares.

Panel structure: 51 states x 8 years = 408 state-year observations per vehicle type.

Usage:
    python drdml_estimator.py --input ../../data/processed/panel_data.csv \
                              --output ../../outputs/ate_results.csv
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

VEHICLE_TYPES = ["BEV", "PHEV", "HEV", "Hydrogen", "Propane"]

POLICY_MECHANISMS = [
    "upfront_cost_reduction",
    "operating_cost_incentives",
    "access_convenience",
    "weight_capacity_advantages",
    "regulatory_relief",
    "restrictions_penalties",
]

COVARIATES = [
    "electricity_price",
    "gdp_per_capita",
    "population_density",
    "median_age",
    "employment_rate",
]

N_FOLDS = 5
RANDOM_STATE = 42


# ── DR/DML Core Estimator ──────────────────────────────────────────────────────

def fit_drdml(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    cluster_ids: np.ndarray,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Estimate the Average Treatment Effect (ATE) using Doubly Robust Machine Learning
    with cross-fitting and cluster-robust standard errors.

    Parameters
    ----------
    Y : array-like of shape (n,)
        Outcome variable (AFV adoption share, %).
    T : array-like of shape (n,)
        Binary treatment indicator (policy active = 1).
    X : array-like of shape (n, p)
        Covariates (state socioeconomic characteristics).
    cluster_ids : array-like of shape (n,)
        State identifiers for cluster-robust SE computation.
    n_folds : int
        Number of cross-fitting folds.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: ate, se, ci_lower, ci_upper, t_stat, p_value, n_obs
    """
    n = len(Y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Arrays to store cross-fitted residuals
    Y_res = np.zeros(n)
    T_res = np.zeros(n)
    psi = np.zeros(n)  # DR scores

    scaler = StandardScaler()

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Outcome model: E[Y | X] (mu model)
        mu_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=random_state,
        )
        mu_model.fit(X_train_s, Y_train)
        mu_hat = mu_model.predict(X_test_s)

        # Propensity model: P(T=1 | X) (pi model)
        pi_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state,
        )
        pi_model.fit(X_train_s, T_train)
        pi_hat = np.clip(pi_model.predict_proba(X_test_s)[:, 1], 0.05, 0.95)

        # Residuals
        Y_res[test_idx] = Y_test - mu_hat
        T_res[test_idx] = T_test - pi_hat

        # DR score (efficient influence function)
        psi[test_idx] = (
            (T_test - pi_hat) / (pi_hat * (1 - pi_hat))
        ) * (Y_test - mu_hat)

    # ATE: regress Y residuals on T residuals (Frisch-Waugh-Lovell)
    # theta = (T_res' T_res)^{-1} T_res' Y_res
    theta = np.dot(T_res, Y_res) / np.dot(T_res, T_res)

    # Cluster-robust standard errors
    scores = psi + theta * T_res - Y_res  # influence function
    clusters = np.unique(cluster_ids)
    n_clusters = len(clusters)

    cluster_scores = np.array([
        scores[cluster_ids == c].sum() for c in clusters
    ])

    sandwich_var = (
        np.dot(cluster_scores, cluster_scores)
        / (np.dot(T_res, T_res) ** 2)
        * (n_clusters / (n_clusters - 1))
    )
    se = np.sqrt(sandwich_var)

    t_stat = theta / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_clusters - 1))
    ci_lower = theta - 1.96 * se
    ci_upper = theta + 1.96 * se

    return {
        "ate": theta,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_obs": n,
        "n_clusters": n_clusters,
    }


# ── Main Estimation Loop ───────────────────────────────────────────────────────

def run_estimation(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over all vehicle type × policy mechanism combinations
    and estimate DR/DML ATEs.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with columns for outcomes, treatments, covariates, and state IDs.

    Returns
    -------
    pd.DataFrame with ATE results for each combination.
    """
    results = []

    for vehicle in VEHICLE_TYPES:
        outcome_col = f"share_{vehicle.lower()}"
        if outcome_col not in panel_df.columns:
            print(f"[skip] Outcome column '{outcome_col}' not found.")
            continue

        for mechanism in POLICY_MECHANISMS:
            treatment_col = f"policy_{mechanism}"
            if treatment_col not in panel_df.columns:
                print(f"[skip] Treatment column '{treatment_col}' not found.")
                continue

            subset = panel_df.dropna(
                subset=[outcome_col, treatment_col] + COVARIATES
            ).copy()

            # Require minimum policy variation for identification
            n_treated = subset[treatment_col].sum()
            n_control = len(subset) - n_treated
            if n_treated < 10 or n_control < 10:
                print(
                    f"[skip] {vehicle} × {mechanism}: "
                    f"insufficient variation (treated={n_treated}, control={n_control})"
                )
                continue

            Y = subset[outcome_col].values
            T = subset[treatment_col].values.astype(float)
            X = subset[COVARIATES].values
            cluster_ids = subset["state_fips"].values

            print(f"Estimating: {vehicle} × {mechanism} (n={len(subset)})")

            res = fit_drdml(Y, T, X, cluster_ids)
            results.append({
                "vehicle_type": vehicle,
                "policy_mechanism": mechanism,
                "n_obs": res["n_obs"],
                "n_clusters": res["n_clusters"],
                "ate_pp": round(res["ate"], 4),
                "se": round(res["se"], 4),
                "ci_lower": round(res["ci_lower"], 4),
                "ci_upper": round(res["ci_upper"], 4),
                "t_stat": round(res["t_stat"], 4),
                "p_value": round(res["p_value"], 4),
                "significant_05": res["p_value"] < 0.05,
            })

    return pd.DataFrame(results).sort_values("p_value")


# ── Forest Plot ───────────────────────────────────────────────────────────────

def plot_forest(results_df: pd.DataFrame, output_path: str = None):
    """
    Generate a forest plot of ATE estimates with 95% confidence intervals,
    grouped by vehicle type.
    """
    df = results_df.copy()
    df["label"] = df["vehicle_type"] + " | " + df["policy_mechanism"].str.replace("_", " ").str.title()
    df = df.sort_values(["vehicle_type", "ate_pp"])

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.5)))

    colors = {True: "#2ecc71", False: "#95a5a6"}
    y_positions = range(len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        color = colors[row["significant_05"]]
        ax.plot(
            [row["ci_lower"], row["ci_upper"]], [i, i],
            color=color, linewidth=2, solid_capstyle="round"
        )
        ax.scatter(row["ate_pp"], i, color=color, s=60, zorder=5)

    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.set_xlabel("Average Treatment Effect (percentage points)", fontsize=11)
    ax.set_title(
        "DR/DML Estimated Effects of Clean Vehicle Policy Mechanisms\non AFV Adoption Shares (95% CI, Cluster-Robust SE)",
        fontsize=12, pad=12
    )

    legend_elements = [
        plt.scatter([], [], color="#2ecc71", label="p < 0.05"),
        plt.scatter([], [], color="#95a5a6", label="p ≥ 0.05"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Forest plot saved to {output_path}")
    else:
        plt.show()


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DR/DML estimation of clean vehicle policy effects."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to processed panel CSV (state-year observations)."
    )
    parser.add_argument(
        "--output", default="../../outputs/ate_results.csv",
        help="Path to save ATE results CSV."
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate and save forest plot."
    )
    args = parser.parse_args()

    print(f"\nLoading panel data from: {args.input}")
    panel_df = pd.read_csv(args.input)
    print(f"Panel shape: {panel_df.shape}")

    results_df = run_estimation(panel_df)

    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
    print(f"\nTop results (p < 0.05):")
    print(results_df[results_df["significant_05"]].to_string(index=False))

    if args.plot:
        plot_path = args.output.replace(".csv", "_forest_plot.png")
        plot_forest(results_df, output_path=plot_path)


if __name__ == "__main__":
    main()
