import pandas as pd
import json
from tools.stats_tools import profile_dataset
from tools.cleaning_tools import clean_dataset
from llm.llm_client import LLMClient

def _json_safe(obj):
    return str(obj)

def _clean_summary_output(text: str) -> str:
    """Strip common hallucinated meta-text artifacts from small local models."""
    cutoff_markers = ["Instruction", "###", "Note:", "System:", "User:"]
    for marker in cutoff_markers:
        idx = text.find(marker)
        if idx > 50:  # only cut if it appears after real content, not at the very start
            text = text[:idx].strip()
    return text

def run_profiling_agent(filepath: str, llm: LLMClient) -> dict:
    raw_df = pd.read_csv(filepath)
    clean_result = clean_dataset(raw_df)
    df = clean_result["cleaned_df"]
    profile = profile_dataset(df)

    cleaning_notes = "\n".join(clean_result["report"]) if clean_result["report"] else "No cleaning was necessary."

    raw_summary = llm.generate(
        prompt=f"Summarize this dataset profile in plain English, 3-4 sentences:\n{json.dumps(profile, default=_json_safe)}\n\nCleaning performed:\n{cleaning_notes}",
        system="You are a data analyst. Output ONLY the plain-English summary paragraph. Do not include any headers, instructions, meta-commentary, or repeated text. Only describe what's in the JSON and cleaning notes. Never invent numbers."
    )
    summary = _clean_summary_output(raw_summary)

    return {"profile": profile, "summary": summary, "dataframe": df, "cleaning_report": clean_result["report"]}