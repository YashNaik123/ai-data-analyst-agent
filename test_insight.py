import pandas as pd
from tools.ml_tools import attribute_ranking_regression
from agents.insight_agent import generate_insight
from llm.llm_client import LLMClient

df = pd.read_csv("data/sample_sales.csv")
df["units_sold"] = df["units_sold"].fillna(df["units_sold"].mean())

reg_result = attribute_ranking_regression(df, target="units_sold", features=["brand", "price"])
print("RAW ANALYSIS RESULT:", reg_result)

llm = LLMClient()
insight = generate_insight(reg_result, llm)

print("\nOBSERVATION:", insight.get("observation"))
print("INTERPRETATION:", insight.get("interpretation"))
print("VERIFIED:", insight.get("verified"))
if not insight.get("verified"):
    print("UNVERIFIED NUMBERS FLAGGED:", insight.get("unverified_numbers"))