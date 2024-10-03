import pinecone
from typing import List, Dict, Any, Optional
from .base import BaseVectorDB, VectorDBProviderType

class Pinecone(BaseVectorDB):
    def __init__(self, api_key: Optional[str] = None, environment: str = "us-west1-gcp"):
        super().__init__(VectorDBProviderType.PINECONE, api_key)
        self.environment = environment
        pinecone.init(api_key=self.auth.get_key(VectorDBProviderType.PINECONE), environment=self.environment)
        self.index = None

    def create_index(self, index_name: str, dimension: int) -> bool:
        try:
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(name=index_name, dimension=dimension)
            self.index = pinecone.Index(index_name)
            return True
        except Exception as e:
            print(f"Error creating index: {e}")
            return False

    def list_indexes(self) -> List[str]:
        return pinecone.list_indexes()

    def delete_index(self, index_name: str) -> bool:
        try:
            pinecone.delete_index(index_name)
            self.index = None
            return True
        except Exception as e:
            print(f"Error deleting index: {e}")
            return False

    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> List[str]:
        if not self.index:
            raise ValueError("No index selected. Please create or select an index first.")
        
        vector_ids = [f"vec_{i}" for i in range(len(vectors))]
        items_to_upsert = list(zip(vector_ids, vectors, metadata))
        
        try:
            self.index.upsert(items_to_upsert)
            return vector_ids
        except Exception as e:
            print(f"Error adding vectors: {e}")
            return []

    def search_vectors(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.index:
            raise ValueError("No index selected. Please create or select an index first.")
        
        try:
            results = self.index.query(query_vector, top_k=top_k, include_metadata=True)
            return [{"id": match.id, "score": match.score, "metadata": match.metadata} for match in results.matches]
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []

    def delete_vectors(self, vector_ids: List[str]) -> bool:
        if not self.index:
            raise ValueError("No index selected. Please create or select an index first.")
        
        try:
            self.index.delete(ids=vector_ids)
            return True
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False

    def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        if not self.index:
            raise ValueError("No index selected. Please create or select an index first.")
        
        try:
            self.index.update(id=vector_id, set_metadata=metadata)
            return True
        except Exception as e:
            print(f"Error updating metadata: {e}")
            return False

    def get_vector(self, vector_id: str) -> Dict[str, Any]:
        if not self.index:
            raise ValueError("No index selected. Please create or select an index first.")
        
        try:
            result = self.index.fetch([vector_id])
            if vector_id in result.vectors:
                vector = result.vectors[vector_id]
                return {"id": vector_id, "vector": vector.values, "metadata": vector.metadata}
            else:
                return {}
        except Exception as e:
            print(f"Error getting vector: {e}")
            return {}