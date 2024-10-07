from typing import List, Dict, Any, Optional, Union
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential

from multivec.providers.base import BaseVectorDB, VectorDBProviderType
from multivec.providers.embedding import Embedding, EmbeddingProvider
from multivec.utils.base_format import BaseDocument, TextDocument, ImageDocument, AudioDocument, VideoDocument, MultimodalDocument, Vector
from multivec.exceptions import QdrantError

class QdrantConnectionType(Enum):
    LOCAL = "local"
    CLOUD = "cloud"

class Qdrant(BaseVectorDB):
    def __init__(
        self,
        connection_type: QdrantConnectionType,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        host: str = "localhost",
        port: int = 6333,
        prefer_grpc: bool = False,
        embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI,
        embedding_model: str = "text-embedding-ada-002",
        embedding_api_key: Optional[str] = None
    ):
        super().__init__(VectorDBProviderType.QDRANT, api_key, embedding_provider, embedding_model, embedding_api_key)
        self.connection_type = connection_type
        self.url = url
        self.host = host
        self.port = port
        self.prefer_grpc = prefer_grpc

        try:
            from qdrant_client import QdrantClient
            if connection_type == QdrantConnectionType.CLOUD:
                if not url or not api_key:
                    raise ValueError("URL and API key are required for cloud connection")
                self.client = QdrantClient(
                    url=self.url,
                    api_key=self.auth.get_key(VectorDBProviderType.QDRANT),
                    prefer_grpc=self.prefer_grpc
                )
            else:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    prefer_grpc=self.prefer_grpc
                )
            self.collection_name = None
        except Exception as e:
            raise QdrantError(f"Failed to initialize Qdrant: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_collection(self, name: str, vector_size: int) -> None:
        """
        Create a new collection in Qdrant.

        :param name: Name of the collection
        :param vector_size: Size of the vectors in the collection
        """
        try:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            self.collection_name = name
        except Exception as e:
            raise QdrantError(f"Failed to create collection: {str(e)}")

    def list_collections(self) -> List[str]:
        """
        List all collections in Qdrant.

        :return: List of collection names
        """
        try:
            collections = self.client.get_collections()
            return [collection.name for collection in collections.collections]
        except Exception as e:
            raise QdrantError(f"Failed to list collections: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def delete_collection(self, name: str) -> None:
        """
        Delete a collection from Qdrant.

        :param name: Name of the collection to delete
        """
        try:
            self.client.delete_collection(collection_name=name)
            if self.collection_name == name:
                self.collection_name = None
        except Exception as e:
            raise QdrantError(f"Failed to delete collection: {str(e)}")

    def _prepare_point(self, document: BaseDocument, vector: Vector) -> PointStruct:
        """
        Prepare a PointStruct from a document and its vector.

        :param document: A BaseDocument object
        :param vector: A Vector object
        :return: A PointStruct object
        """
        metadata = {
            "type": document.data_type.value,
            "page_index": document.page_index,
            **document.metadata
        }

        if isinstance(document, TextDocument):
            metadata["content"] = document.content
        elif isinstance(document, ImageDocument):
            metadata.update({
                "image_path": document.image_path,
                "caption": document.caption,
                "width": document.width,
                "height": document.height
            })
        elif isinstance(document, AudioDocument):
            metadata.update({
                "audio_url": document.audio_url,
                "duration": document.duration,
                "transcript": document.transcript
            })
        elif isinstance(document, VideoDocument):
            metadata.update({
                "video_url": document.video_url,
                "duration": document.duration,
                "thumbnail_url": document.thumbnail_url,
                "caption": document.caption
            })
        elif isinstance(document, MultimodalDocument):
            metadata["components"] = [comp.model_dump() for comp in document.components]

        return PointStruct(
            id=document.id,
            vector=vector.data,
            payload=metadata
        )

    def add_documents(
        self, 
        documents: List[BaseDocument],
        vectors: Optional[List[Vector]] = None
    ) -> List[str]:
        """
        Add documents and their corresponding vectors to the collection.
        If vectors are not provided, they will be generated using the embedding model.

        :param documents: List of BaseDocument objects
        :param vectors: Optional list of Vector objects
        :return: List of added document IDs
        """
        if not self.collection_name:
            raise QdrantError("No collection selected. Please create or select a collection first.")
        
        if vectors is None:
            # Generate embeddings for the documents
            texts = [self._get_document_text(doc) for doc in documents]
            embeddings = self.embedding.generate(texts)
            vectors = [Vector(data=emb, dim=len(emb)) for emb in embeddings]
        
        if len(documents) != len(vectors):
            raise ValueError("Number of documents must match number of vectors")
        
        try:
            points = [self._prepare_point(doc, vec) for doc, vec in zip(documents, vectors)]
            self.client.upsert(collection_name=self.collection_name, points=points)
            return [str(point.id) for point in points]
        except Exception as e:
            raise QdrantError(f"Failed to add documents: {str(e)}")

    def _get_document_text(self, document: BaseDocument) -> str:
        """
        Extract text content from a document for embedding.

        :param document: A BaseDocument object
        :return: Text content of the document
        """
        if isinstance(document, TextDocument):
            return document.content
        elif isinstance(document, ImageDocument):
            return document.caption or ""
        elif isinstance(document, AudioDocument):
            return document.transcript or ""
        elif isinstance(document, VideoDocument):
            return document.caption or ""
        elif isinstance(document, MultimodalDocument):
            return " ".join([self._get_document_text(comp) for comp in document.components])
        else:
            return ""

    def search(
        self, 
        query: Union[str, Vector], 
        top_k: int = 10, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.

        :param query: Query string or Vector to search for
        :param top_k: Number of results to return
        :param filter: Optional filter for the search
        :return: List of search results
        """
        if not self.collection_name:
            raise QdrantError("No collection selected. Please create or select a collection first.")
        
        try:
            if isinstance(query, str):
                query_vector = self.embedding.generate(query)[0]
            else:
                query_vector = query.data

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter
            )
            return [{"id": str(hit.id), "score": hit.score, "metadata": hit.payload} for hit in search_result]
        except Exception as e:
            raise QdrantError(f"Failed to search vectors: {str(e)}")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the collection.

        :param document_ids: List of document IDs to delete
        """
        if not self.collection_name:
            raise QdrantError("No collection selected. Please create or select a collection first.")
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=document_ids)
            )
        except Exception as e:
            raise QdrantError(f"Failed to delete documents: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> None:
        """
        Update the metadata of a document.

        :param document_id: ID of the document to update
        :param metadata: New metadata to set
        """
        if not self.collection_name:
            raise QdrantError("No collection selected. Please create or select a collection first.")
        
        try:
            self.client.update_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[document_id]
            )
        except Exception as e:
            raise QdrantError(f"Failed to update document metadata: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Retrieve a document from the collection.

        :param document_id: ID of the document to retrieve
        :return: Document data including vector and metadata
        """
        if not self.collection_name:
            raise QdrantError("No collection selected. Please create or select a collection first.")
        
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[document_id]
            )
            if result:
                point = result[0]
                return {
                    "id": str(point.id),
                    "vector": Vector(data=point.vector, dim=len(point.vector)),
                    "metadata": point.payload
                }
            else:
                return {}
        except Exception as e:
            raise QdrantError(f"Failed to get document: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def batch_upload(
        self, 
        documents: List[BaseDocument],
        vectors: List[Vector],
        batch_size: int = 100
    ) -> List[str]:
        """
        Upload documents and vectors in batches.

        :param documents: List of BaseDocument objects
        :param vectors: List of Vector objects
        :param batch_size: Number of documents to upload in each batch
        :return: List of uploaded document IDs
        """
        if not self.collection_name:
            raise QdrantError("No collection selected. Please create or select a collection first.")
        
        if len(documents) != len(vectors):
            raise ValueError("Number of documents must match number of vectors")
        
        try:
            points = [self._prepare_point(doc, vec) for doc, vec in zip(documents, vectors)]
            uploaded_ids = []
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(collection_name=self.collection_name, points=batch)
                uploaded_ids.extend([str(point.id) for point in batch])
            return uploaded_ids
        except Exception as e:
            raise QdrantError(f"Failed to batch upload documents: {str(e)}")

    def create_payload_index(self, field_name: str, field_schema: Union[str, Dict[str, Any]]) -> None:
        """
        Create an index on a payload field for faster filtering.

        :param field_name: Name of the field to index
        :param field_schema: Schema of the field
        """
        if not self.collection_name:
            raise QdrantError("No collection selected. Please create or select a collection first.")
        
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema
            )
        except Exception as e:
            raise QdrantError(f"Failed to create payload index: {str(e)}")

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.

        :return: Dictionary containing collection information
        """
        if not self.collection_name:
            raise QdrantError("No collection selected. Please create or select a collection first.")
        
        try:
            return self.client.get_collection(collection_name=self.collection_name).dict()
        except Exception as e:
            raise QdrantError(f"Failed to get collection info: {str(e)}")

