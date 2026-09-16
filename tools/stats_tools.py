import pandas as pd

def profile_dataset(df: pd.DataFrame) -> dict:
    profile = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": {},
        "duplicates": int(df.duplicated().sum()),
    }
    for col in df.columns:
        col_data = df[col]
        info = {
            "dtype": str(col_data.dtype),
            "missing_pct": round(col_data.isna().mean() * 100, 2),
            "n_unique": int(col_data.nunique()),
        }
        if pd.api.types.is_numeric_dtype(col_data):
            q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            iqr = q3 - q1
            outliers = col_data[(col_data < q1 - 1.5*iqr) | (col_data > q3 + 1.5*iqr)]
            info.update({
                "mean": round(col_data.mean(), 2),
                "median": round(col_data.median(), 2),
                "std": round(col_data.std(), 2),
                "outlier_count": int(outliers.count()),
            })
        else:
            info["top_values"] = col_data.value_counts().head(3).to_dict()
        profile["columns"][col] = info
    return profile
