import plotly.express as px
import plotly.graph_objects as go

CHART_DECISION_TABLE = {
    "correlation_matrix": ("heatmap", "A heatmap shows relationship strength across many variable pairs at once"),
    "regression_ranking": ("bar_horizontal", "A sorted horizontal bar chart ranks attributes by influence, easiest to scan top-to-bottom"),
    "segmentation": ("scatter_colored", "A colored scatter plot shows how records group by cluster"),
    "anomaly_detection": ("scatter_highlighted", "A scatter plot with anomalies highlighted shows outliers in context"),
    "hypothesis_test": ("box", "A box plot compares distributions between groups being tested"),
}

def select_and_render_chart(analysis_type: str, data: dict):
    chart_type, rationale = CHART_DECISION_TABLE.get(analysis_type, ("table", "No specific chart rule matched; showing raw table"))

    fig = None
    if chart_type == "heatmap":
        labels = list(data.keys())
        z_values = [[data[col][row] for col in labels] for row in labels]
        fig = go.Figure(data=go.Heatmap(z=z_values, x=labels, y=labels))
    elif chart_type == "bar_horizontal":
        fig = px.bar(x=list(data.values()), y=list(data.keys()), orientation="h")

    return {"chart": fig, "chart_type": chart_type, "rationale": rationale}