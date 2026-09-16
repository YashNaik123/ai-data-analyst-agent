import pandas as pd
from tools.stats_tools import profile_dataset
import json

df = pd.read_csv("data/sample_sales.csv")
profile = profile_dataset(df)
print(json.dumps(profile, indent=2))