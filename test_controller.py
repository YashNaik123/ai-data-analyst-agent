from agents.controller import Controller

controller = Controller()
summary = controller.load_data("data/sample_sales.csv")
print("PROFILE SUMMARY:", summary)

result = controller.ask(
    "What drives units sold the most?",
    method_kwargs={"target": "units_sold", "features": ["brand", "price"]}
)
print("\nMETHOD USED:", result["route"]["method"], "-", result["route"]["rationale"])
print("OBSERVATION:", result["insight"].get("observation"))
print("INTERPRETATION:", result["insight"].get("interpretation"))