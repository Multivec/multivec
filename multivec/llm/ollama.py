from .base import BaseLLM
import requests


class ChatOllama(BaseLLM):
    def __init__(self, model: str, api_url: str = "http://localhost:11434"):
        self.model = model
        self.api_url = api_url

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.api_url}/api/generate", json={"model": self.model, "prompt": prompt}
        )
        return response.json()["response"]
