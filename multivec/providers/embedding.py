from typing import List, Union, Optional
from enum import Enum
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import torch
from PIL import Image
import clip

class EmbeddingProvider(Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    CLIP = "clip"

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
        elif self.provider == EmbeddingProvider.CLIP:
            model, preprocess = clip.load(self.model_name, device=self.device)
            return (model, preprocess)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, inputs: Union[str, List[str], Image.Image, List[Image.Image]]) -> List[List[float]]:
        """
        Generate embeddings for the given text(s) or image(s).

        :param inputs: A single text/image or a list of texts/images to embed
        :return: A list of embeddings
        """
        if isinstance(inputs, (str, Image.Image)):
            inputs = [inputs]

        if self.provider == EmbeddingProvider.OPENAI:
            import openai
            if isinstance(inputs[0], str):
                response = openai.Embedding.create(input=inputs, model=self.model_name)
                return [item["embedding"] for item in response["data"]]
            else:
                raise ValueError("OpenAI provider does not support image embeddings")
        elif self.provider == EmbeddingProvider.HUGGINGFACE:
            model, tokenizer = self.model
            if isinstance(inputs[0], str):
                inputs_encoded = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = model(**inputs_encoded)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            else:
                raise ValueError("Hugging Face provider does not support image embeddings in this implementation")
            return embeddings.tolist()
        elif self.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            if isinstance(inputs[0], str):
                return self.model.encode(inputs).tolist()
            else:
                raise ValueError("SentenceTransformers provider does not support image embeddings")
        elif self.provider == EmbeddingProvider.CLIP:
            model, preprocess = self.model
            if isinstance(inputs[0], str):
                text = clip.tokenize(inputs).to(self.device)
                with torch.no_grad():
                    text_features = model.encode_text(text)
                return text_features.cpu().numpy().tolist()
            elif isinstance(inputs[0], Image.Image):
                images = [preprocess(img).unsqueeze(0).to(self.device) for img in inputs]
                images = torch.cat(images)
                with torch.no_grad():
                    image_features = model.encode_image(images)
                return image_features.cpu().numpy().tolist()
            else:
                raise ValueError("Unsupported input type for CLIP")

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
        elif self.provider == EmbeddingProvider.CLIP:
            return self.model[0].visual.output_dim

# Usage example:
# text_embedding = Embedding(EmbeddingProvider.OPENAI, "text-embedding-ada-002", api_key="your-api-key")
# text_vectors = text_embedding.generate(["Hello, world!", "Another example"])
# text_embedding_dim = text_embedding.get_embedding_dim()

# Multimodal embedding example:
# clip_embedding = Embedding(EmbeddingProvider.CLIP, "ViT-B/32", device="cuda")
# text_vectors = clip_embedding.generate(["A photo of a cat", "A picture of a dog"])
# image = Image.open("path/to/your/image.jpg")
# image_vector = clip_embedding.generate(image)
# embedding_dim = clip_embedding.get_embedding_dim()