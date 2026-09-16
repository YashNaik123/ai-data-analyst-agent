import ollama

class LLMClient:
    def __init__(self, backend="ollama", model="phi3:mini"):
        self.backend = backend
        self.model = model

    def generate(self, prompt: str, system: str = "") -> str:
        if self.backend == "ollama":
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
            )
            return response["message"]["content"]
        else:
            raise ValueError(f"Unknown backend: {self.backend}")