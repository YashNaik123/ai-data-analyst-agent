import yaml

def load_pack(pack_path: str) -> dict:
    with open(pack_path) as f:
        return yaml.safe_load(f)

def route_question(question: str, pack: dict) -> dict:
    question_lower = question.lower()
    for route in pack["routes"]:
        if any(pattern in question_lower for pattern in route["trigger_patterns"]):
            return route
    return {"method": "descriptive_stats", "rationale": "No specific pattern matched; defaulting to summary statistics"}