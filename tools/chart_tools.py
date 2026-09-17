import plotly.express as px
import plotly.graph_objects as go

CHART_DECISION_TABLE = {
    "descriptive_stats": ("table", "Descriptive statistics are best viewed as a table, not a chart"),
    "correlation_matrix": ("heatmap", "A heatmap shows relationship strength across many variable pairs at once"),
    "regression_ranking": ("bar_horizontal", "A sorted horizontal bar chart ranks attributes by influence, easiest to scan top-to-bottom"),
    "segmentation": ("scatter_colored", "A colored scatter plot shows how records group by cluster"),
    "anomaly_detection": ("scatter_highlighted", "A scatter plot with anomalies highlighted shows outliers in context"),
    "hypothesis_test": ("bar_comparison", "A bar comparison shows the statistic and significance threshold clearly"),
    "markov_transition": ("heatmap", "A heatmap shows transition probabilities between all product pairs"),
}

def select_and_render_chart(analysis_type: str, data: dict):
    chart_type, rationale = CHART_DECISION_TABLE.get(analysis_type, ("table", "No specific chart rule matched; showing raw table"))
    fig = None

    try:
        if chart_type == "heatmap":
            labels = list(data.keys())
            z_values = [[data[col].get(row, None) for col in labels] for row in labels]
            fig = go.Figure(data=go.Heatmap(z=z_values, x=labels, y=labels, colorscale="Blues"))
            fig.update_layout(title="Correlation / Transition Heatmap")

        elif chart_type == "bar_horizontal":
            fig = px.bar(x=list(data.values()), y=list(data.keys()), orientation="h",
                         title="Attribute Ranking")
            fig.update_layout(yaxis_title="", xaxis_title="Influence")

        elif chart_type == "scatter_colored":
            labels = data.get("labels", [])
            fig = px.scatter(x=list(range(len(labels))), y=labels, color=[str(l) for l in labels],
                             title="Segmentation Clusters")
            fig.update_layout(xaxis_title="Record Index", yaxis_title="Cluster")

        elif chart_type == "scatter_highlighted":
            first_val = list(data.values())[0] if data else None
            if isinstance(first_val, dict):
                values = first_val
            else:
                values = data
            keys = list(values.keys()) if isinstance(values, dict) else list(range(len(values)))
            vals = list(values.values()) if isinstance(values, dict) else values
            fig = px.scatter(x=keys, y=vals, title="Anomalies Detected")
            fig.update_traces(marker=dict(color="red", size=10))

        elif chart_type == "bar_comparison":
            fig = px.bar(x=["t-statistic", "p-value"], y=[data.get("t_stat", 0), data.get("p_value", 0)],
                        title="Hypothesis Test Result")

        elif chart_type == "table":
            fig = go.Figure(data=[go.Table(
                header=dict(values=["Metric", "Value"]),
                cells=dict(values=[list(data.keys()), [str(v) for v in data.values()]])
            )])
    except Exception as e:
        return {"chart": None, "chart_type": chart_type, "rationale": rationale, "render_error": str(e)}

    return {"chart": fig, "chart_type": chart_type, "rationale": rationale}