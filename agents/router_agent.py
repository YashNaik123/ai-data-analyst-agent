import yaml
from difflib import SequenceMatcher

def load_pack(pack_path: str) -> dict:
    with open(pack_path) as f:
        return yaml.safe_load(f)

def _fuzzy_contains(text: str, pattern: str, threshold: float = 0.8) -> bool:
    """Checks substrings of text against pattern using similarity ratio, to catch typos."""
    if pattern in text:
        return True
    words = text.split()
    pattern_len = len(pattern.split())
    for i in range(len(words) - pattern_len + 1):
        candidate = " ".join(words[i:i + pattern_len])
        if SequenceMatcher(None, candidate, pattern).ratio() >= threshold:
            return True
    return False

def route_question(question: str, pack: dict) -> dict:
    question_lower = question.lower().strip()
    for route in pack["routes"]:
        if any(_fuzzy_contains(question_lower, pattern) for pattern in route["trigger_patterns"]):
            return route
    return {"method": "descriptive_stats", "rationale": "No specific pattern matched; defaulting to summary statistics"}