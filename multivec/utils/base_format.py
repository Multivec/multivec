from typing import List, Optional
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator
from enum import Enum
import numpy as np


class DataType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


class BaseDocument(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    metadata: dict = Field(default_factory=dict)
    data_type: DataType
    page_index: Optional[int] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TextDocument(BaseDocument):
    content: str

    def __init__(self, **data):
        super().__init__(data_type=DataType.TEXT, **data)


class ImageDocument(BaseDocument):
    image_path: str
    caption: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

    def __init__(self, **data):
        super().__init__(data_type=DataType.IMAGE, **data)


class AudioDocument(BaseDocument):
    audio_url: str
    duration: Optional[float] = None
    transcript: Optional[str] = None

    def __init__(self, **data):
        super().__init__(data_type=DataType.AUDIO, **data)


class VideoDocument(BaseDocument):
    video_url: str
    duration: Optional[float] = None
    thumbnail_url: Optional[str] = None
    caption: Optional[str] = None

    def __init__(self, **data):
        super().__init__(data_type=DataType.VIDEO, **data)

class MultimodalDocument(BaseDocument):
    components: List[BaseDocument]

    def __init__(self, **data):
        super().__init__(data_type=DataType.MULTIMODAL, **data)


class DocumentCollection(BaseModel):
    documents: List[BaseDocument]

    model_config = ConfigDict(arbitrary_types_allowed=True)

class Vector(BaseDocument):
    """
    A class representing a mathematical vector with various operations.
    
    This class inherits from BaseDocument and provides functionality for vector operations
    such as dot product, magnitude calculation, and normalization.

     Usage example:
        v1 = Vector(data=[1, 2, 3], dim=3, metadata={"source": "example"}) \n
        v2 = Vector(data=[4, 5, 6], dim=3) \n
        dot_product = v1.dot(v2) \n
        normalized_v1 = v1.normalize()\n
        cosine_sim = v1.cosine_similarity(v2) \n
    """

    data: List[float]
    dim: int

    def __init__(self, **data):
        """
        Initialize a Vector object.

        :param data: List of float values representing the vector.
        :param dim: Dimension of the vector.
        :param metadata: Optional metadata dictionary.
        :param page_index: Optional page index.
        """
        super().__init__(data_type=DataType.VECTOR, **data)

    @field_validator("data")
    def check_dimensions(cls, v, values):
        """
        Validate that the vector's dimension matches the specified dimension.

        :param v: The vector data.
        :param values: The values dictionary containing the 'dim' key.
        :return: The validated vector data.
        :raises ValueError: If the vector dimension doesn't match the specified dimension.
        """
        if "dim" in values and len(v) != values["dim"]:
            raise ValueError(
                f"Vector dimension mismatch. Expected {values['dim']}, got {len(v)}"
            )
        return v

    def __len__(self) -> int:
        """
        Get the length (dimension) of the vector.

        :return: The dimension of the vector.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> float:
        """
        Get the value at the specified index in the vector.

        :param index: The index of the desired value.
        :return: The value at the specified index.
        """
        return self.data[index]

    def dot(self, other: "Vector") -> float:
        """
        Calculate the dot product with another vector.

        :param other: Another Vector object.
        :return: The dot product of the two vectors.
        :raises ValueError: If the vectors have different dimensions.
        """
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimension for dot product")
        return np.dot(self.data, other.data)

    def magnitude(self) -> float:
        """
        Calculate the magnitude (Euclidean norm) of the vector.

        :return: The magnitude of the vector.
        """
        return np.linalg.norm(self.data)

    def normalize(self) -> "Vector":
        """
        Normalize the vector (create a unit vector).

        :return: A new Vector object representing the normalized vector.
        """
        mag = self.magnitude()
        return Vector(data=(np.array(self.data) / mag).tolist(), dim=self.dim, metadata=self.metadata.copy())

    def add(self, other: "Vector") -> "Vector":
        """
        Add another vector to this vector.

        :param other: Another Vector object.
        :return: A new Vector object representing the sum of the two vectors.
        :raises ValueError: If the vectors have different dimensions.
        """
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimension for addition")
        return Vector(data=(np.array(self.data) + np.array(other.data)).tolist(), dim=self.dim, metadata=self.metadata.copy())

    def subtract(self, other: "Vector") -> "Vector":
        """
        Subtract another vector from this vector.

        :param other: Another Vector object.
        :return: A new Vector object representing the difference between the two vectors.
        :raises ValueError: If the vectors have different dimensions.
        """
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimension for subtraction")
        return Vector(data=(np.array(self.data) - np.array(other.data)).tolist(), dim=self.dim, metadata=self.metadata.copy())

    def scale(self, scalar: float) -> "Vector":
        """
        Scale the vector by a scalar value.

        :param scalar: The scalar value to multiply the vector by.
        :return: A new Vector object representing the scaled vector.
        """
        return Vector(data=(np.array(self.data) * scalar).tolist(), dim=self.dim, metadata=self.metadata.copy())

    def to_numpy(self) -> np.ndarray:
        """
        Convert the vector to a numpy array.

        :return: A numpy array representation of the vector.
        """
        return np.array(self.data)

    @classmethod
    def from_numpy(cls, array: np.ndarray, metadata: Optional[dict] = None) -> "Vector":
        """
        Create a Vector object from a numpy array.

        :param array: A numpy array.
        :param metadata: Optional metadata dictionary.
        :return: A new Vector object.
        """
        return cls(data=array.tolist(), dim=len(array), metadata=metadata or {})

    def cosine_similarity(self, other: "Vector") -> float:
        """
        Calculate the cosine similarity with another vector.

        :param other: Another Vector object.
        :return: The cosine similarity between the two vectors.
        :raises ValueError: If the vectors have different dimensions.
        """
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimension for cosine similarity")
        return self.dot(other) / (self.magnitude() * other.magnitude())

    # Usage example:
    # v1 = Vector(data=[1, 2, 3], dim=3, metadata={"source": "example"})
    # v2 = Vector(data=[4, 5, 6], dim=3)
    # dot_product = v1.dot(v2)
    # normalized_v1 = v1.normalize()
    # cosine_sim = v1.cosine_similarity(v2)

