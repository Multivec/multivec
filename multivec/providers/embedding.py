from typing import List, Union, Optional
from enum import Enum
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import torch

class EmbeddingProvider(Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"

class Embedding:
    def __init__(
        self,
        provider: EmbeddingProvider,
        model_name: str,
        api_key: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize the Embedding class.

        :param provider: The embedding provider to use
        :param model_name: The name of the model to use for embeddings
        :param api_key: API key for the provider (if required)
        :param device: The device to use for local embedding (e.g., "cpu" or "cuda")
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        if self.provider == EmbeddingProvider.OPENAI:
            import openai
            openai.api_key = self.api_key
            return None  # OpenAI doesn't require a local model
        elif self.provider == EmbeddingProvider.HUGGINGFACE:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name).to(self.device)
            return (model, tokenizer)
        elif self.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(self.model_name).to(self.device)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for the given text(s).

        :param texts: A single text or a list of texts to embed
        :return: A list of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.provider == EmbeddingProvider.OPENAI:
            import openai
            response = openai.Embedding.create(input=texts, model=self.model_name)
            return [item["embedding"] for item in response["data"]]
        elif self.provider == EmbeddingProvider.HUGGINGFACE:
            model, tokenizer = self.model
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return embeddings.tolist()
        elif self.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return self.model.encode(texts).tolist()

    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the embeddings.

        :return: The dimension of the embeddings
        """
        if self.provider == EmbeddingProvider.OPENAI:
            # For OpenAI, we need to make an API call to get the embedding dimension
            sample_embedding = self.generate("Sample text")[0]
            return len(sample_embedding)
        elif self.provider == EmbeddingProvider.HUGGINGFACE:
            return self.model[0].config.hidden_size
        elif self.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return self.model.get_sentence_embedding_dimension()

# Usage example:
# embedding = Embedding(EmbeddingProvider.OPENAI, "text-embedding-ada-002", api_key="your-api-key")
# vectors = embedding.generate(["Hello, world!", "Another example"])
# embedding_dim = embedding.get_embedding_dim()

# Local embedding example:
# embedding = Embedding(EmbeddingProvider.SENTENCE_TRANSFORMERS, "all-MiniLM-L6-v2", device="cuda")
# vectors = embedding.generate(["Hello, world!", "Another example"])
# embedding_dim = embedding.get_embedding_dim()