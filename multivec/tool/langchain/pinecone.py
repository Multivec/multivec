from typing import List, Dict, Any
from multivec.providers.qdrant import Qdrant
from multivec.utils.base_format import BaseDocument, Vector
from multivec.providers.pinecone import Pinecone

class PineconeLangChainAdapter:
    """
    An adapter class to use Pinecone as a LangChain tool without adding LangChain
    as a direct dependency to the multivec library.

    Example of Usage:
        Initialize your Pinecone instance \n
        pinecone = Pinecone(
            api_key="your-api-key",
            environment="your-environment"
        )
        Create the adapter \n
        adapter = PineconeLangChainAdapter(pinecone) \n

        Create the LangChain tool class \n
        PineconeTool = adapter.to_tool(BaseTool) \n

        Instantiate the tool \n
        pinecone_tool = PineconeTool() \n

        Create a list of tools (you can add more tools if needed)
        tools = [pinecone_tool] \n

        Initialize the LLM \n
        llm = OpenAI(temperature=0) \n

        Create the agent \n
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

        Use the agent
        agent.run("Search for a vector in the Pinecone database: 0.1, 0.2, 0.3, 0.4")
    """

    def __init__(self, pinecone_instance: Pinecone):
        self.pinecone = pinecone_instance

    def to_tool(self, BaseTool):
        """
        Convert the Pinecone instance to a LangChain tool.
        
        :param BaseTool: The LangChain BaseTool class (passed by the user)
        :return: A LangChain Tool class that can be instantiated
        """
        
        class PineconeTool(BaseTool):
            name: str = "Pinecone Vector Database"
            description: str = "A tool for interacting with the Pinecone vector database for similarity search and document retrieval."
            pinecone: Pinecone = self.pinecone

            def _run(self, query: str) -> str:
                """Use the tool."""
                try:
                    # Assume query is a comma-separated list of floats representing a vector
                    query_vector = Vector(data=[float(x.strip()) for x in query.split(',')], dim=len(query.split(',')))
                    results = self.pinecone.search(query_vector, top_k=5)
                    return self._format_results(results)
                except Exception as e:
                    return f"Error: {str(e)}"

            async def _arun(self, query: str) -> str:
                """Use the tool asynchronously."""
                # For simplicity, we're just calling the synchronous version here
                return self._run(query)

            def _format_results(self, results: List[Dict[str, Any]]) -> str:
                """Format the search results as a string."""
                formatted = "Search Results:\n"
                for i, result in enumerate(results, 1):
                    formatted += f"{i}. ID: {result['id']}, Score: {result['score']:.4f}\n"
                    formatted += f"   Metadata: {result['metadata']}\n\n"
                return formatted

            def add_documents(self, documents: List[BaseDocument], vectors: List[Vector]) -> List[str]:
                """Add documents to the Pinecone database."""
                return self.pinecone.add_documents(documents, vectors)

            def delete_documents(self, document_ids: List[str]) -> None:
                """Delete documents from the Pinecone database."""
                self.pinecone.delete_documents(document_ids)

            def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> None:
                """Update the metadata of a document in the Pinecone database."""
                self.pinecone.update_document_metadata(document_id, metadata)

        return PineconeTool
