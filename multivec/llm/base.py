# File: multivec/llm/base.py

from abc import ABC, abstractmethod
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

ProviderType = Literal["ollama", "openai", "anthropic", "bedrock"]


class BaseLLM(ABC):
    def __init__(self, provider: ProviderType, api_key: Optional[str] = None):
        from .auth import Auth

        self.provider = provider
        self.auth = Auth()
        if api_key:
            self.auth.set_key(provider, api_key)
        elif not self.auth.get_key(provider):
            raise ValueError(f"No API key provided or found for {provider}")

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class TemperatureValidator(BaseModel):
    temperature: float = Field(..., ge=0.0, le=1.0, description="Temperature must be between 0 and 1.")

    @field_validator('temperature')
    def validate_temperature(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('Temperature must be between 0 and 1.')
        return v

# Update to Latest models
OpenAIModel = Literal["text-davinci-002", "text-davinci-003", "gpt-3.5-turbo", "gpt-4"]

AnthropicModel = Literal["claude-2", "claude-instant-1"] 

BedrockModel = Literal["anthropic.claude-v2", "ai21.j2-ultra", "amazon.titan-tg1-large"]

GroqModel = Literal["anthropic.claude-v2", "ai21.j2-ultra", "amazon.titan-tg1-large", "llama-3.2-3b-preview"]
