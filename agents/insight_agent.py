import json
from llm.llm_client import LLMClient

INSIGHT_SYSTEM_PROMPT = """You are a business analyst. You will be given analysis results as JSON.
Respond ONLY with a JSON object with this exact structure, no other text, no markdown formatting:
{
  "observation": "what the numbers literally show, citing exact figures from the input",
  "interpretation": "what this might mean for the business, clearly speculative",
  "supporting_numbers": ["list", "of", "exact", "figures", "used", "as strings"]
}
Never state a number that isn't in the input data."""

def generate_insight(analysis_result: dict, llm: LLMClient) -> dict:
    raw = llm.generate(prompt=json.dumps(analysis_result), system=INSIGHT_SYSTEM_PROMPT)
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    try:
        insight = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "observation": "Could not parse a structured insight from the model output.",
            "interpretation": "",
            "supporting_numbers": [],
            "verified": False,
            "unverified_numbers": [],
            "raw_output": raw,
        }

    raw_result_str = json.dumps(analysis_result)
    verified_numbers = [n for n in insight.get("supporting_numbers", []) if str(n) in raw_result_str]
    insight["verified"] = len(verified_numbers) == len(insight.get("supporting_numbers", []))
    insight["unverified_numbers"] = list(set(insight.get("supporting_numbers", [])) - set(verified_numbers))
    return insight