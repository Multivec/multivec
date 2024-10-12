# File: multivec/llm/base.py

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from multivec.providers.auth import Auth
from multivec.providers.embedding import Embedding, EmbeddingProvider
from multivec.utils.base_format import BaseDocument, Vector


class CloudProviderType(Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"


class BaseCloud(ABC):
    def __init__(self, provider: CloudProviderType, api_key: Optional[str] = None):
        self.provider = provider
        self.auth = Auth()  # Using the singleton Auth instance
        if api_key:
            self.auth.set_key(provider.value, api_key)
        elif not self.auth.get_key(provider.value):
            raise ValueError(f"No API key provided or found for {provider}")

    @abstractmethod
    def connect(self):
        pass


LLMProviderType = Literal["ollama", "openai", "anthropic", "bedrock"]


class VectorDBProviderType(Enum):
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    FAISS = "faiss"


class BaseLLM(ABC):
    def __init__(self, provider: LLMProviderType, api_key: Optional[str] = None):
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


class VectorDBProviderType(Enum):
    PINECONE = "pinecone"
    QDRANT = "qdrant"


class BaseVectorDB(ABC):
    def __init__(
        self,
        provider: VectorDBProviderType,
        api_key: Optional[str] = None,
        embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI,
        embedding_model: str = "text-embedding-ada-002",
        embedding_api_key: Optional[str] = None,
    ):
        from .auth import Auth

        self.provider = provider
        self.auth = Auth()
        if api_key:
            self.auth.set_key(provider, api_key)
        elif not self.auth.get_key(provider):
            raise ValueError(f"No API key provided or found for {provider}")

        self.embedding = Embedding(
            embedding_provider, embedding_model, api_key=embedding_api_key
        )

    @abstractmethod
    def create_index(self, name: str, dimension: int, **kwargs) -> None:
        pass

    @abstractmethod
    def list_indexes(self) -> List[str]:
        pass

    @abstractmethod
    def delete_index(self, name: str) -> None:
        pass

    @abstractmethod
    def add_documents(
        self, documents: List[BaseDocument], vectors: Optional[List[Vector]] = None
    ) -> List[str]:
        pass

    @abstractmethod
    def search(
        self,
        query: Union[str, Vector],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> None:
        pass

    @abstractmethod
    def update_document_metadata(
        self, document_id: str, metadata: Dict[str, Any]
    ) -> None:
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def batch_upload(
        self,
        documents: List[BaseDocument],
        vectors: Optional[List[Vector]] = None,
        batch_size: int = 100,
    ) -> List[str]:
        pass


class TemperatureValidator(BaseModel):
    temperature: float = Field(
        ..., ge=0.0, le=1.0, description="Temperature must be between 0 and 1."
    )

    @field_validator("temperature")
    def validate_temperature(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Temperature must be between 0 and 1.")
        return v


# Update to Latest models
OpenAIModel = Literal["text-davinci-002", "text-davinci-003", "gpt-3.5-turbo", "gpt-4"]

AnthropicModel = Literal["claude-2", "claude-instant-1"]

BedrockModel = Literal["anthropic.claude-v2", "ai21.j2-ultra", "amazon.titan-tg1-large"]

GroqModel = Literal[
    "gemma2-9b-it",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.2-1b-preview",
]
