# File: multivec/llm/base.py

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator

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


class BaseVectorDB(ABC):
    def __init__(self, provider: VectorDBProviderType, api_key: Optional[str] = None):
        from .auth import Auth

        self.provider = provider
        self.auth = Auth()
        if api_key:
            self.auth.set_key(provider, api_key)
        elif not self.auth.get_key(provider):
            raise ValueError(f"No API key provided or found for {provider}")

    @abstractmethod
    def add_vectors(
        self, vectors: List[List[float]], metadata: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add vectors to the database with associated metadata.

        :param vectors: List of vector embeddings
        :param metadata: List of metadata dictionaries corresponding to each vector
        :return: List of unique IDs for the added vectors
        """
        pass

    @abstractmethod
    def search_vectors(
        self, query_vector: List[float], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for the most similar vectors to the query vector.

        :param query_vector: The query vector to search for
        :param top_k: Number of results to return
        :return: List of dictionaries containing search results (vector IDs, scores, and metadata)
        """
        pass

    @abstractmethod
    def delete_vectors(self, vector_ids: List[str]) -> bool:
        """
        Delete vectors from the database by their IDs.

        :param vector_ids: List of vector IDs to delete
        :return: True if deletion was successful, False otherwise
        """
        pass

    @abstractmethod
    def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update the metadata for a specific vector.

        :param vector_id: ID of the vector to update
        :param metadata: New metadata to associate with the vector
        :return: True if update was successful, False otherwise
        """
        pass

    @abstractmethod
    def get_vector(self, vector_id: str) -> Dict[str, Any]:
        """
        Retrieve a vector and its metadata by ID.

        :param vector_id: ID of the vector to retrieve
        :return: Dictionary containing the vector and its metadata
        """
        pass

    @abstractmethod
    def create_index(self, index_name: str, dimension: int) -> bool:
        """
        Create a new index in the vector database.

        :param index_name: Name of the index to create
        :param dimension: Dimensionality of the vectors to be stored
        :return: True if index creation was successful, False otherwise
        """
        pass

    @abstractmethod
    def list_indexes(self) -> List[str]:
        """
        List all available indexes in the vector database.

        :return: List of index names
        """
        pass

    @abstractmethod
    def delete_index(self, index_name: str) -> bool:
        """
        Delete an index from the vector database.

        :param index_name: Name of the index to delete
        :return: True if deletion was successful, False otherwise
        """
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
    "anthropic.claude-v2",
    "ai21.j2-ultra",
    "amazon.titan-tg1-large",
    "llama-3.2-3b-preview",
]
