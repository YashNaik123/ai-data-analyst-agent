import json
from llm.llm_client import LLMClient

INSIGHT_SYSTEM_PROMPT = """You are a business analyst. You will be given analysis results as JSON.
Respond ONLY with a JSON object with this exact structure, no other text, no markdown formatting:
{
  "observation": "what the numbers literally show, citing exact figures from the input",
  "interpretation": "what this might mean for the business, clearly speculative",
  "recommendation": "concrete, actionable suggestion for what the business should do based on this finding",
  "supporting_numbers": ["list", "of", "exact", "figures", "used", "as strings"]
}
Never state a number that isn't in the input data."""

def _json_safe(obj):
    return str(obj)

def _all_keys(obj, keys=None):
    if keys is None:
        keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            _all_keys(v, keys)
    elif isinstance(obj, list):
        for item in obj:
            _all_keys(item, keys)
    return keys

def generate_insight(analysis_result: dict, llm: LLMClient) -> dict:
    relevant_keys = _all_keys(analysis_result.get("result", {}))

    if relevant_keys and len(relevant_keys) <= 12:
        keys_hint = f"\n\nAttributes that MUST all be mentioned in your observation: {sorted(relevant_keys)}"
    elif relevant_keys:
        keys_hint = "\n\nThis is a broad multi-attribute analysis. Give a high-level synthesis highlighting the most important patterns rather than listing every single number."
    else:
        keys_hint = ""

    raw = llm.generate(
        prompt=json.dumps(analysis_result, default=_json_safe) + keys_hint,
        system=INSIGHT_SYSTEM_PROMPT
    )
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    try:
        insight = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "observation": "Could not parse a structured insight from the model output.",
            "interpretation": "",
            "recommendation": "",
            "supporting_numbers": [],
            "verified": False,
            "unverified_numbers": [],
            "raw_output": raw,
        }

    raw_result_str = json.dumps(analysis_result, default=_json_safe)
    verified_numbers = [n for n in insight.get("supporting_numbers", []) if str(n) in raw_result_str]
    insight["verified"] = len(verified_numbers) == len(insight.get("supporting_numbers", []))
    insight["unverified_numbers"] = list(set(insight.get("supporting_numbers", [])) - set(verified_numbers))

    if len(relevant_keys) > 12:
        insight["coverage_complete"] = True
        insight["missing_attributes"] = []
    else:
        observation_lower = insight.get("observation", "").lower()
        missing_attrs = [k for k in relevant_keys if k.lower() not in observation_lower]
        insight["coverage_complete"] = len(missing_attrs) == 0
        insight["missing_attributes"] = missing_attrs

    return insight