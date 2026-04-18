"""
prompt_templates.py
-------------------
Structured prompts for dual-LLM policy classification.
Each policy is assigned to exactly one of six mechanism categories.
"""

MECHANISM_LABELS = {
    "upfront_cost_reduction":      "Upfront Cost Reduction",
    "operating_cost_incentives":   "Operating Cost Incentives",
    "access_convenience":          "Access & Convenience Benefits",
    "weight_capacity_advantages":  "Weight & Capacity Advantages",
    "regulatory_relief":           "Regulatory Relief",
    "restrictions_penalties":      "Restrictions & Penalties",
}

MECHANISM_DEFINITIONS = {
    "upfront_cost_reduction": (
        "Policies that reduce the purchase price or upfront acquisition cost "
        "of alternative fuel vehicles (AFVs), including tax credits, rebates, "
        "vouchers, grants, and preferential fleet procurement rules."
    ),
    "operating_cost_incentives": (
        "Policies that lower the ongoing costs of owning or operating an AFV, "
        "including fuel tax exemptions, reduced registration/inspection fees, "
        "tolling discounts, and preferential utility rates for EV charging."
    ),
    "access_convenience": (
        "Policies that expand the usability or convenience of AFVs, including "
        "HOV/HOT lane access, preferential parking, charging infrastructure "
        "mandates or incentives, and fueling station requirements."
    ),
    "weight_capacity_advantages": (
        "Policies that permit AFVs to exceed standard vehicle weight, axle load, "
        "or size limits, providing operational advantages especially for commercial "
        "fleets and freight vehicles."
    ),
    "regulatory_relief": (
        "Policies that exempt AFVs from regulations that apply to conventional "
        "vehicles, such as emissions testing, inspection requirements, idling "
        "restrictions, or certain licensing rules."
    ),
    "restrictions_penalties": (
        "Policies that impose restrictions, fees, or penalties on conventional "
        "ICE vehicles or high-emission vehicles to incentivize AFV adoption, "
        "including congestion pricing, emission fees, or ICE vehicle bans."
    ),
}


SYSTEM_CONTEXT = """\
You are an expert transportation policy analyst specializing in alternative fuel vehicle (AFV) policy classification. 
Your task is to classify a U.S. state or local government policy into exactly one of six mechanism categories.
You must respond with valid JSON only — no preamble, no markdown fences, no explanation outside the JSON.
"""

CLASSIFICATION_SCHEMA = """\
{
  "mechanism": "<one of the six mechanism keys>",
  "confidence": <float between 0.0 and 1.0>,
  "rationale": "<1-2 sentence justification referencing specific policy language>"
}
"""

def build_classification_prompt(policy_text: str) -> str:
    """
    Build a structured classification prompt for a single policy.

    Parameters
    ----------
    policy_text : str
        Raw policy description text from AFDC.

    Returns
    -------
    str : Full prompt string.
    """
    mechanism_block = "\n".join([
        f'- "{key}": {definition}'
        for key, definition in MECHANISM_DEFINITIONS.items()
    ])

    prompt = f"""{SYSTEM_CONTEXT}

## Policy Mechanism Definitions

{mechanism_block}

## Policy Text to Classify

\"\"\"{policy_text.strip()}\"\"\"

## Instructions

1. Read the policy text carefully.
2. Select the SINGLE best-fitting mechanism from the six categories above.
3. Assign a confidence score (0.0 = very uncertain, 1.0 = very confident).
4. Provide a brief rationale that references specific language in the policy text.

## Response Format

Respond with ONLY a valid JSON object matching this schema:
{CLASSIFICATION_SCHEMA}

Valid mechanism keys: {list(MECHANISM_DEFINITIONS.keys())}
"""
    return prompt
