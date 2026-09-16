import pandas as pd
import json
from tools.stats_tools import profile_dataset
from llm.llm_client import LLMClient

def run_profiling_agent(filepath: str, llm: LLMClient) -> dict:
    df = pd.read_csv(filepath)
    profile = profile_dataset(df)

    summary = llm.generate(
        prompt=f"Summarize this dataset profile in plain English, 3-4 sentences:\n{json.dumps(profile)}",
        system="You are a data analyst. Only describe what's in the JSON. Never invent numbers."
    )
    return {"profile": profile, "summary": summary, "dataframe": df}