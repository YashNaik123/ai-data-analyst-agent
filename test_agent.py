from agents.profiling_agent import run_profiling_agent
from llm.llm_client import LLMClient

llm = LLMClient()
result = run_profiling_agent("data/sample_sales.csv", llm)

print("SUMMARY:")
print(result["summary"])
print("\nPROFILE KEYS:", list(result["profile"].keys()))