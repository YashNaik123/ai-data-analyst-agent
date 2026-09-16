import pandas as pd
import json
from tools.stats_tools import profile_dataset
from llm.llm_client import LLMClient

df = pd.read_csv("data/sample_sales.csv")
profile = profile_dataset(df)

llm = LLMClient()
summary = llm.generate(
    prompt=f"Summarize this dataset profile in plain English, 3-4 sentences:\n{json.dumps(profile)}",
    system="You are a data analyst. Only describe what's in the JSON. Never invent numbers."
)
print(summary)