from typing import Optional
from .base import AnthropicModel, BaseLLM
import anthropic


class ChatAnthropic(BaseLLM):
    """
    A class that integrates with the Anthropic API to generate text completions 
    using the specified Anthropic model. This class provides an interface for 
    interacting with Anthropic language models and handles the authentication 
    process.

    Attributes:
    - model: The Anthropic model to be used for generating completions.
    - api_key: (Optional) The API key for Anthropic authentication. If not provided, 
      it retrieves the key from the authentication module.

    Methods:
    - generate(prompt: str) -> str: Sends the prompt to the Anthropic model and 
      returns the generated completion.
    """
    def __init__(self, model: AnthropicModel, api_key: Optional[str] = None):
        super().__init__("anthropic", api_key)
        self.model = model
        self.client = anthropic.Client(api_key=self.auth.get_key("anthropic"))

    def generate(self, prompt: str) -> str:
        response = self.client.completion(
            model=self.model, prompt=prompt, max_tokens_to_sample=300
        )
        return response.completion
