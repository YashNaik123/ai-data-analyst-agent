import inspect
from agents.profiling_agent import run_profiling_agent
from agents.router_agent import load_pack, route_question
from tools import ml_tools, chart_tools
from agents.insight_agent import generate_insight
from llm.llm_client import LLMClient

METHOD_MAP = {
    "descriptive_stats": ml_tools.descriptive_stats,
    "correlation_analysis": ml_tools.correlation_analysis,
    "attribute_ranking_regression": ml_tools.attribute_ranking_regression,
    "segmentation": ml_tools.segmentation,
    "anomaly_detection": ml_tools.anomaly_detection,
    "hypothesis_testing": ml_tools.hypothesis_testing,
}

class Controller:
    def __init__(self, pack_path="packs/default.yaml", backend="ollama"):
        self.llm = LLMClient(backend=backend)
        self.pack = load_pack(pack_path)
        self.state = {}

    def load_data(self, filepath: str):
        result = run_profiling_agent(filepath, self.llm)
        self.state["df"] = result["dataframe"]
        self.state["profile"] = result["profile"]
        return result["summary"]

    def ask(self, question: str, method_kwargs: dict):
        route = route_question(question, self.pack)
        method_fn = METHOD_MAP.get(route["method"])
        if not method_fn:
            return {"error": f"Method '{route['method']}' not implemented yet"}

        # Only pass kwargs that this specific method actually accepts
        accepted_params = set(inspect.signature(method_fn).parameters.keys())
        filtered_kwargs = {k: v for k, v in method_kwargs.items() if k in accepted_params}

        try:
            analysis_result = method_fn(self.state["df"], **filtered_kwargs)
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}

        chart_info = chart_tools.select_and_render_chart(analysis_result["type"], analysis_result["result"])
        insight = generate_insight(analysis_result, self.llm)

        return {"route": route, "analysis": analysis_result, "chart": chart_info, "insight": insight}