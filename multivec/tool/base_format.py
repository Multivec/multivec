from pydantic import BaseModel, Field


class QdrantToolInput(BaseModel):
    query: str = Field(..., description="A comma-separated list of floats representing a vector to search for in the database.")
