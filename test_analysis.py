import pandas as pd
from tools.ml_tools import correlation_analysis, attribute_ranking_regression

df = pd.read_csv("data/sample_sales.csv")
df["units_sold"] = df["units_sold"].fillna(df["units_sold"].mean())  # handle the missing value for this test

corr_result = correlation_analysis(df, ["price", "units_sold"])
print("CORRELATION:", corr_result)

reg_result = attribute_ranking_regression(df, target="units_sold", features=["brand", "price"])
print("\nREGRESSION RANKING:", reg_result)