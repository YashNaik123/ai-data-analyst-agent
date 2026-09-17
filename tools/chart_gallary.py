import pandas as pd
import plotly.express as px

def generate_attribute_charts(df: pd.DataFrame, target: str = None) -> list:
    """
    Returns a list of {title, chart_type, figure} dicts covering:
    - individual distribution of every column
    - each column's relationship to `target`, if provided
    """
    charts = []
    numeric_cols = list(df.select_dtypes(include="number").columns)
    categorical_cols = list(df.select_dtypes(exclude="number").columns)
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical_cols = [c for c in categorical_cols if c not in datetime_cols]

    # --- Individual distributions ---
    for col in numeric_cols:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}", nbins=30)
        charts.append({"title": f"Distribution of {col}", "chart_type": "histogram", "figure": fig})

    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique > 30:
            continue  # skip identifier-like columns, too many bars to be useful
        counts = df[col].value_counts(dropna=True).reset_index()
        counts.columns = [col, "count"]
        if n_unique <= 6:
            fig = px.pie(counts, names=col, values="count", title=f"Breakdown of {col}", hole=0.4)  # donut chart
        else:
            fig = px.bar(counts, x=col, y="count", title=f"Count by {col}")
        charts.append({"title": f"Breakdown of {col}", "chart_type": "pie_or_bar", "figure": fig})

    for col in datetime_cols:
        counts = df[col].dt.to_period("M").value_counts().sort_index()
        fig = px.line(x=counts.index.astype(str), y=counts.values, title=f"{col} over time")
        charts.append({"title": f"{col} over time", "chart_type": "line", "figure": fig})

    # --- Relationship to target, if specified ---
    if target and target in df.columns:
        for col in numeric_cols:
            if col == target:
                continue
            fig = px.scatter(df, x=col, y=target, title=f"{target} vs {col}", trendline="ols")
            charts.append({"title": f"{target} vs {col}", "chart_type": "scatter", "figure": fig})

        for col in categorical_cols:
            n_unique = df[col].nunique()
            if n_unique > 15:
                continue
            fig = px.box(df, x=col, y=target, title=f"{target} by {col}")
            charts.append({"title": f"{target} by {col}", "chart_type": "box", "figure": fig})

    return charts