"""
classify_policies.py
--------------------
Dual-LLM pipeline for classifying U.S. clean vehicle policies into
mechanism categories using GPT-4.1-mini (OpenAI) and Claude Haiku (Anthropic)
in parallel, with cross-provider agreement analysis.

Policy text is sourced from the AFDC (Alternative Fuels Data Center) database.

Usage:
    python classify_policies.py --input ../../data/raw/policies.csv \
                                --output ../../outputs/classified_policies.csv

Environment variables required:
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
"""

import os
import argparse
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from prompt_templates import build_classification_prompt, MECHANISM_LABELS

# ── API Clients ────────────────────────────────────────────────────────────────

def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ── Classification Functions ───────────────────────────────────────────────────

def classify_with_gpt(client, policy_text: str, retries: int = 3) -> dict:
    """
    Classify a single policy using GPT-4.1-mini.
    Returns dict with 'mechanism', 'confidence', 'rationale'.
    """
    prompt = build_classification_prompt(policy_text)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            result["provider"] = "openai"
            return result
        except Exception as e:
            if attempt == retries - 1:
                return {"mechanism": "error", "confidence": 0.0,
                        "rationale": str(e), "provider": "openai"}
            time.sleep(2 ** attempt)


def classify_with_claude(client, policy_text: str, retries: int = 3) -> dict:
    """
    Classify a single policy using Claude Haiku.
    Returns dict with 'mechanism', 'confidence', 'rationale'.
    """
    prompt = build_classification_prompt(policy_text)

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            # Claude returns text; extract JSON from response
            text = response.content[0].text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text.strip())
            result["provider"] = "anthropic"
            return result
        except Exception as e:
            if attempt == retries - 1:
                return {"mechanism": "error", "confidence": 0.0,
                        "rationale": str(e), "provider": "anthropic"}
            time.sleep(2 ** attempt)


# ── Agreement Analysis ─────────────────────────────────────────────────────────

def compute_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-provider agreement for each classified policy.
    Flags disagreements for human review.
    """
    df = df.copy()
    df["agreement"] = df["mechanism_gpt"] == df["mechanism_claude"]
    df["needs_review"] = (
        (~df["agreement"]) |
        (df["confidence_gpt"] < 0.7) |
        (df["confidence_claude"] < 0.7)
    )

    # Final classification: use GPT if agreement, else flag for review
    df["mechanism_final"] = np.where(
        df["agreement"],
        df["mechanism_gpt"],
        "REVIEW_REQUIRED"
    )

    agreement_rate = df["agreement"].mean()
    print(f"\nCross-provider agreement rate: {agreement_rate:.1%}")
    print(f"Flagged for review: {df['needs_review'].sum()} / {len(df)}")

    return df


# ── Summary Statistics ─────────────────────────────────────────────────────────

def summarize_classifications(df: pd.DataFrame):
    """Print mechanism distribution and agreement stats."""
    print("\n── Mechanism Distribution (Final Classifications) ──")
    counts = (
        df[df["mechanism_final"] != "REVIEW_REQUIRED"]["mechanism_final"]
        .value_counts()
    )
    for mechanism, count in counts.items():
        label = MECHANISM_LABELS.get(mechanism, mechanism)
        pct = count / len(df) * 100
        print(f"  {label:<40} n={count:>4}  ({pct:.1f}%)")

    print(f"\n── Agreement by Mechanism ──")
    agreement_by_mech = (
        df.groupby("mechanism_gpt")["agreement"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "agreement_rate", "count": "n"})
        .sort_values("agreement_rate")
    )
    print(agreement_by_mech.to_string())


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def run_classification(
    policies_df: pd.DataFrame,
    text_col: str = "policy_text",
    id_col: str = "policy_id",
    batch_delay: float = 0.5,
) -> pd.DataFrame:
    """
    Run dual-LLM classification on all policies.

    Parameters
    ----------
    policies_df : pd.DataFrame
        DataFrame with policy text and identifiers.
    text_col : str
        Column name containing policy text.
    id_col : str
        Column name containing unique policy identifiers.
    batch_delay : float
        Seconds to wait between API calls (rate limit buffer).

    Returns
    -------
    pd.DataFrame with classification results from both providers.
    """
    oai_client = get_openai_client()
    ant_client = get_anthropic_client()

    records = []

    for _, row in tqdm(policies_df.iterrows(), total=len(policies_df),
                       desc="Classifying policies"):
        policy_id = row[id_col]
        policy_text = str(row[text_col])

        # Skip empty/short texts
        if len(policy_text.strip()) < 20:
            continue

        gpt_result = classify_with_gpt(oai_client, policy_text)
        claude_result = classify_with_claude(ant_client, policy_text)

        records.append({
            "policy_id": policy_id,
            "policy_text": policy_text[:500],  # truncate for output
            "mechanism_gpt": gpt_result.get("mechanism", "error"),
            "confidence_gpt": gpt_result.get("confidence", 0.0),
            "rationale_gpt": gpt_result.get("rationale", ""),
            "mechanism_claude": claude_result.get("mechanism", "error"),
            "confidence_claude": claude_result.get("confidence", 0.0),
            "rationale_claude": claude_result.get("rationale", ""),
        })

        time.sleep(batch_delay)

    results_df = pd.DataFrame(records)
    results_df = compute_agreement(results_df)
    return results_df


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dual-LLM clean vehicle policy classification pipeline."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to raw policy CSV (must include 'policy_id' and 'policy_text' columns)."
    )
    parser.add_argument(
        "--output", default="../../outputs/classified_policies.csv",
        help="Path to save classification results."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of policies to classify (for testing)."
    )
    args = parser.parse_args()

    print(f"\nLoading policies from: {args.input}")
    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)
    print(f"Policies to classify: {len(df)}")

    results_df = run_classification(df)
    summarize_classifications(results_df)

    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
