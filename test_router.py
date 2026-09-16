from agents.router_agent import load_pack, route_question

pack = load_pack("packs/default.yaml")

questions = [
    "What drives sales the most?",
    "Is there a trend over time?",
    "Can you segment my customers?",
    "asdkjfh random gibberish question",
]

for q in questions:
    route = route_question(q, pack)
    print(f"Q: {q}\n → method: {route['method']} | why: {route['rationale']}\n")