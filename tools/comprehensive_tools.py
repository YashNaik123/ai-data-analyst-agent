import pandas as pd
from sklearn.linear_model import LinearRegression

def comprehensive_analysis(df: pd.DataFrame) -> dict:
    numeric_cols = list(df.select_dtypes(include="number").columns)
    categorical_cols = list(df.select_dtypes(exclude="number").columns)

    result = {"summary_stats": {}, "correlations": {}, "outliers": {}}

    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) == 0:
            continue
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        outliers = data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)]
        result["summary_stats"][col] = {
            "mean": round(data.mean(), 2),
            "median": round(data.median(), 2),
            "min": round(data.min(), 2),
            "max": round(data.max(), 2),
        }
        result["outliers"][col] = len(outliers)

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(2)
        result["correlations"] = corr.to_dict()

    regression_summary = None
    if len(numeric_cols) >= 2:
        target = numeric_cols[0]
        features = [c for c in numeric_cols[1:] + categorical_cols if c != target]
        try:
            data = df[[target] + features].dropna()
            X = pd.get_dummies(data[features], drop_first=True)
            y = data[target]
            if len(X) > 0 and X.shape[1] > 0:
                model = LinearRegression().fit(X, y)
                ranking = dict(sorted(zip(X.columns, model.coef_), key=lambda x: abs(x[1]), reverse=True))
                regression_summary = {"target": target, "top_driver": list(ranking.keys())[0], "r_squared": round(model.score(X, y), 3)}
        except Exception:
            pass

    return {
        "result": result,
        "regression_highlight": regression_summary,
        "type": "comprehensive_analysis",
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }