from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from .base import BaseVectorDB, VectorDBProviderType

class Qdrant(BaseVectorDB):
    def __init__(self, 
                 url: Optional[str] = None, 
                 api_key: Optional[str] = None, 
                 host: str = "localhost", 
                 port: int = 6333):
        super().__init__(VectorDBProviderType.QDRANT, api_key)
        
        if url:
            # Cloud setup
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            # Local setup
            self.client = QdrantClient(host=host, port=port)
        
        self.collection_name = None

    def create_index(self, index_name: str, dimension: int) -> bool:
        try:
            self.client.create_collection(
                collection_name=index_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )
            self.collection_name = index_name
            return True
        except Exception as e:
            print(f"Error creating index: {e}")
            return False

    def list_indexes(self) -> List[str]:
        try:
            collections = self.client.get_collections()
            return [collection.name for collection in collections.collections]
        except Exception as e:
            print(f"Error listing indexes: {e}")
            return []

    def delete_index(self, index_name: str) -> bool:
        try:
            self.client.delete_collection(collection_name=index_name)
            if self.collection_name == index_name:
                self.collection_name = None
            return True
        except Exception as e:
            print(f"Error deleting index: {e}")
            return False

    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> List[str]:
        if not self.collection_name:
            raise ValueError("No collection selected. Please create or select a collection first.")
        
        try:
            points = [
                models.PointStruct(
                    id=str(i),
                    vector=vector,
                    payload=metadata[i]
                ) for i, vector in enumerate(vectors)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            return [str(i) for i in range(len(vectors))]
        except Exception as e:
            print(f"Error adding vectors: {e}")
            return []

    def search_vectors(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.collection_name:
            raise ValueError("No collection selected. Please create or select a collection first.")
        
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            return [{"id": str(hit.id), "score": hit.score, "metadata": hit.payload} for hit in search_result]
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []

    def delete_vectors(self, vector_ids: List[str]) -> bool:
        if not self.collection_name:
            raise ValueError("No collection selected. Please create or select a collection first.")
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=vector_ids)
            )
            return True
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False

    def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        if not self.collection_name:
            raise ValueError("No collection selected. Please create or select a collection first.")
        
        try:
            self.client.update_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[vector_id]
            )
            return True
        except Exception as e:
            print(f"Error updating metadata: {e}")
            return False

    def get_vector(self, vector_id: str) -> Dict[str, Any]:
        if not self.collection_name:
            raise ValueError("No collection selected. Please create or select a collection first.")
        
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[vector_id]
            )
            if result:
                point = result[0]
                return {"id": str(point.id), "vector": point.vector, "metadata": point.payload}
            else:
                return {}
        except Exception as e:
            print(f"Error getting vector: {e}")
            return {}