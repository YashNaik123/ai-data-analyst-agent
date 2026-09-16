import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy import stats

def correlation_analysis(df: pd.DataFrame, cols: list) -> dict:
    corr = df[cols].corr()
    return {"result": corr.to_dict(), "type": "correlation_matrix"}

def attribute_ranking_regression(df: pd.DataFrame, target: str, features: list) -> dict:
    X = pd.get_dummies(df[features], drop_first=True)
    y = df[target]
    model = LinearRegression().fit(X, y)
    ranking = dict(sorted(zip(X.columns, model.coef_), key=lambda x: abs(x[1]), reverse=True))
    return {"result": ranking, "r_squared": model.score(X, y), "type": "regression_ranking"}

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