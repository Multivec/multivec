import functools
from typing import Optional

from multivec.providers.base import BaseLLM, GroqModel, TemperatureValidator
from groq import Groq

class ChatGroq(BaseLLM):
    """
    A class for generating text completions using Groq models, supporting temperature 
    settings for controlling the variability of responses. It manages authentication 
    and client interaction with the Groq API.

    Attributes:
    - model_name: The Groq model name to use for generating completions.
    - temperature: (Optional) Controls randomness in the output, with a default value of 0.7.
    - api_key: (Optional) The API key for authenticating with the Groq API. If not provided, 
      it retrieves the key from the authentication module.

    Methods:
    - generate(prompt: str) -> str: Sends the prompt to the Groq model and returns the 
      generated response, including predefined system messages.
    """
    def __init__(self, model_name: GroqModel, temperature: Optional[TemperatureValidator] = 0.7, api_key: Optional[str] = None):
        super().__init__("groq", api_key)
        self.model_name = model_name
        self.api_key = self.auth.get_key("groq")
        self.client = functools.partial(Groq(api_key=self.api_key).chat.completions.create, model=self.model_name, temperature=temperature)

    def generate(self, prompt: str) -> str:
        client = self.client(
            messages = [
                        {
                    "role": "system",
                    "content": "you are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )
        return client.choices[0].message.content