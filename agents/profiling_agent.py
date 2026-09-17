import pandas as pd
import json
from tools.stats_tools import profile_dataset
from tools.cleaning_tools import clean_dataset
from llm.llm_client import LLMClient

def _json_safe(obj):
    return str(obj)

def run_profiling_agent(filepath: str, llm: LLMClient) -> dict:
    raw_df = pd.read_csv(filepath)
    clean_result = clean_dataset(raw_df)
    df = clean_result["cleaned_df"]
    profile = profile_dataset(df)

    cleaning_notes = "\n".join(clean_result["report"]) if clean_result["report"] else "No cleaning was necessary."

    summary = llm.generate(
        prompt=f"Summarize this dataset profile in plain English, 3-4 sentences:\n{json.dumps(profile, default=_json_safe)}\n\nCleaning performed:\n{cleaning_notes}",
        system="You are a data analyst. Only describe what's in the JSON and cleaning notes. Never invent numbers."
    )
    return {"profile": profile, "summary": summary, "dataframe": df, "cleaning_report": clean_result["report"]}