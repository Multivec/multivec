from typing import Optional
from ..base import BaseLLM, OpenAIModel
import openai


class ChatOpenAI(BaseLLM):
    """
    A class that integrates with the OpenAI API to generate text completions
    based on user prompts using a specified OpenAI model. It serves as an
    interface to interact with the OpenAI language models and handles
    authentication and response generation.

    Attributes:
    - model: The OpenAI model to be used for generating completions.
    - api_key: (Optional) The API key used to authenticate requests. If not provided,
      it retrieves the key from the authentication module.

    Methods:
    - generate(prompt: str) -> str: Generates and returns a text completion
      for the given input prompt using the specified OpenAI model.
    """

    def __init__(self, model: OpenAIModel, api_key: Optional[str] = None):
        super().__init__("openai", api_key)
        self.model = model
        openai.api_key = self.auth.get_key("openai")

    def generate(self, prompt: str) -> str:
        response = openai.Completion.create(
            engine=self.model, prompt=prompt, max_tokens=150
        )
        return response.choices[0].text.strip()
