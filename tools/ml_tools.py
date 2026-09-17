import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy import stats

def descriptive_stats(df: pd.DataFrame) -> dict:
    return {"result": df.describe().to_dict(), "type": "descriptive_stats"}

def correlation_analysis(df: pd.DataFrame, cols: list) -> dict:
    corr = df[cols].corr()
    return {"result": corr.to_dict(), "type": "correlation_matrix"}

def attribute_ranking_regression(df: pd.DataFrame, target: str, features: list) -> dict:
    data = df[[target] + features].dropna(subset=[target])
    y = data[target]

    feature_df = pd.DataFrame(index=data.index)
    skipped_columns = []

    for col in features:
        col_data = data[col]

        if pd.api.types.is_datetime64_any_dtype(col_data):
            feature_df[col] = (col_data - col_data.min()).dt.days

        elif pd.api.types.is_bool_dtype(col_data):
            feature_df[col] = col_data.astype(int)

        elif pd.api.types.is_numeric_dtype(col_data):
            feature_df[col] = col_data

        else:
            n_unique = col_data.nunique()
            if n_unique > 50 or n_unique >= 0.9 * len(col_data):
                skipped_columns.append(f"{col} (too many unique values to use as a category — likely an identifier)")
                continue
            dummies = pd.get_dummies(col_data, prefix=col, drop_first=True)
            feature_df = pd.concat([feature_df, dummies], axis=1)

    feature_df = feature_df.dropna()
    y = y.loc[feature_df.index]

    if feature_df.shape[1] == 0 or len(feature_df) == 0:
        raise ValueError(f"No usable features remain after filtering. Skipped: {skipped_columns}")

    model = LinearRegression().fit(feature_df, y)
    ranking = dict(sorted(zip(feature_df.columns, model.coef_), key=lambda x: abs(x[1]), reverse=True))
    return {
        "result": ranking,
        "r_squared": model.score(feature_df, y),
        "type": "regression_ranking",
        "n_rows_used": len(feature_df),
        "skipped_columns": skipped_columns,
    }

def segmentation(df: pd.DataFrame, features: list, n_clusters: int = 3) -> dict:
    X = df[features].dropna()
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
    return {"result": {"labels": km.labels_.tolist(), "centers": km.cluster_centers_.tolist()}, "type": "segmentation"}

def anomaly_detection(df: pd.DataFrame, col: str) -> dict:
    iso = IsolationForest(contamination=0.05, random_state=42)
    values = df[[col]].dropna()
    preds = iso.fit_predict(values)
    anomalies = values[preds == -1]
    return {"result": anomalies.to_dict(), "count": len(anomalies), "type": "anomaly_detection"}

def hypothesis_testing(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    groups = [g[value_col].dropna() for _, g in df.groupby(group_col)]
    t_stat, p_val = stats.ttest_ind(*groups[:2])
    return {"result": {"t_stat": t_stat, "p_value": p_val, "significant": p_val < 0.05}, "type": "hypothesis_test"}