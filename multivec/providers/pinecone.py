from typing import List, Dict, Any, Optional, Union
import pinecone
from tenacity import retry, stop_after_attempt, wait_exponential

from multivec.utils.base_format import BaseDocument, TextDocument, ImageDocument, AudioDocument, VideoDocument, MultimodalDocument, Vector
from multivec.exceptions import PineconeError

class Pinecone:
    def __init__(
        self,
        api_key: str,
        environment: str,
        project_name: Optional[str] = None
    ):
        """
        Initialize the EnhancedPinecone client.


        :param api_key: Pinecone API key
        :param environment: Pinecone environment
        :param project_name: Optional Pinecone project name

        Usage example:
            pinecone_client = EnhancedPinecone(api_key="your-api-key", environment="your-environment")
            pinecone_client.create_index("my_index", dimension=768)
            text_doc = TextDocument(content="example text", metadata={"source": "web"}, page_index=1)
            image_doc = ImageDocument(image_path="path/to/image.jpg", metadata={"source": "local"}, page_index=1)
            vectors = [Vector(data=[0.1, 0.2, ..., 0.768], dim=768) for _ in range(2)]
            pinecone_client.add_documents([text_doc, image_doc], vectors)
        """
        try:
            pinecone.init(api_key=api_key, environment=environment, project_name=project_name)
            self.index = None
        except Exception as e:
            raise PineconeError(f"Failed to initialize Pinecone: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_index(self, name: str, dimension: int, metric: str = "cosine") -> None:
        """
        Create a new index in Pinecone.

        :param name: Name of the index
        :param dimension: Dimension of the vectors
        :param metric: Distance metric to use (default is cosine)
        """
        try:
            if name not in pinecone.list_indexes():
                pinecone.create_index(name, dimension=dimension, metric=metric)
            self.index = pinecone.Index(name)
        except Exception as e:
            raise PineconeError(f"Failed to create index: {str(e)}")

    def list_indexes(self) -> List[str]:
        """
        List all indexes in Pinecone.

        :return: List of index names
        """
        try:
            return pinecone.list_indexes()
        except Exception as e:
            raise PineconeError(f"Failed to list indexes: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def delete_index(self, name: str) -> None:
        """
        Delete an index from Pinecone.

        :param name: Name of the index to delete
        """
        try:
            pinecone.delete_index(name)
            if self.index and self.index.name == name:
                self.index = None
        except Exception as e:
            raise PineconeError(f"Failed to delete index: {str(e)}")

    def _prepare_metadata(self, document: BaseDocument) -> Dict[str, Any]:
        """
        Prepare metadata for a document.

        :param document: A BaseDocument object
        :return: A dictionary of metadata
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
            metadata["components"] = [comp.dict() for comp in document.components]

        return metadata

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def add_documents(
        self, 
        documents: List[BaseDocument],
        vectors: List[Vector]
    ) -> List[str]:
        """
        Add documents and their corresponding vectors to the index.

        :param documents: List of BaseDocument objects
        :param vectors: List of Vector objects
        :return: List of added document IDs
        """
        if not self.index:
            raise PineconeError("No index selected. Please create or select an index first.")
        
        if len(documents) != len(vectors):
            raise ValueError("Number of documents must match number of vectors")
        
        try:
            items_to_upsert = [
                (doc.id, vec.data, self._prepare_metadata(doc))
                for doc, vec in zip(documents, vectors)
            ]
            self.index.upsert(vectors=items_to_upsert)
            return [doc.id for doc in documents]
        except Exception as e:
            raise PineconeError(f"Failed to add documents: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def search(self, query_vector: Vector, top_k: int = 10, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the index.

        :param query_vector: Vector to search for
        :param top_k: Number of results to return
        :param filter: Optional filter for the search
        :return: List of search results
        """
        if not self.index:
            raise PineconeError("No index selected. Please create or select an index first.")
        
        try:
            results = self.index.query(query_vector.data, top_k=top_k, include_metadata=True, filter=filter)
            return [{"id": match.id, "score": match.score, "metadata": match.metadata} for match in results.matches]
        except Exception as e:
            raise PineconeError(f"Failed to search vectors: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the index.

        :param document_ids: List of document IDs to delete
        """
        if not self.index:
            raise PineconeError("No index selected. Please create or select an index first.")
        
        try:
            self.index.delete(ids=document_ids)
        except Exception as e:
            raise PineconeError(f"Failed to delete documents: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> None:
        """
        Update the metadata of a document.

        :param document_id: ID of the document to update
        :param metadata: New metadata to set
        """
        if not self.index:
            raise PineconeError("No index selected. Please create or select an index first.")
        
        try:
            self.index.update(id=document_id, set_metadata=metadata)
        except Exception as e:
            raise PineconeError(f"Failed to update document metadata: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Retrieve a document from the index.

        :param document_id: ID of the document to retrieve
        :return: Document data including vector and metadata
        """
        if not self.index:
            raise PineconeError("No index selected. Please create or select an index first.")
        
        try:
            result = self.index.fetch([document_id])
            if document_id in result.vectors:
                vector_data = result.vectors[document_id]
                return {
                    "id": document_id,
                    "vector": Vector(data=vector_data.values, dim=len(vector_data.values)),
                    "metadata": vector_data.metadata
                }
            else:
                return {}
        except Exception as e:
            raise PineconeError(f"Failed to get document: {str(e)}")

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
        if not self.index:
            raise PineconeError("No index selected. Please create or select an index first.")
        
        if len(documents) != len(vectors):
            raise ValueError("Number of documents must match number of vectors")
        
        try:
            uploaded_ids = []
            for i in range(0, len(documents), batch_size):
                batch_documents = documents[i:i+batch_size]
                batch_vectors = vectors[i:i+batch_size]
                items_to_upsert = [
                    (doc.id, vec.data, self._prepare_metadata(doc))
                    for doc, vec in zip(batch_documents, batch_vectors)
                ]
                self.index.upsert(vectors=items_to_upsert)
                uploaded_ids.extend([doc.id for doc in batch_documents])
            return uploaded_ids
        except Exception as e:
            raise PineconeError(f"Failed to batch upload documents: {str(e)}")

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.

        :return: Dictionary containing index statistics
        """
        if not self.index:
            raise PineconeError("No index selected. Please create or select an index first.")
        
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            raise PineconeError(f"Failed to get index stats: {str(e)}")
