import pandas as pd
from tools.ml_tools import attribute_ranking_regression
from tools.chart_tools import select_and_render_chart

df = pd.read_csv("data/sample_sales.csv")
df["units_sold"] = df["units_sold"].fillna(df["units_sold"].mean())

reg_result = attribute_ranking_regression(df, target="units_sold", features=["brand", "price"])
chart_info = select_and_render_chart(reg_result["type"], reg_result["result"])

print("Chart type:", chart_info["chart_type"])
print("Rationale:", chart_info["rationale"])
chart_info["chart"].write_html("test_chart.html")
print("Chart saved to test_chart.html — open it in your browser to view")