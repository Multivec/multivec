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


class Vector(BaseDocument):
    data: List[float]
    dim: int

    @field_validator("data")
    def check_dimensions(cls, v, values):
        if "dim" in values and len(v) != values["dim"]:
            raise ValueError(
                f"Vector dimension mismatch. Expected {values['dim']}, got {len(v)}"
            )
        return v

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def dot(self, other: "Vector") -> float:
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimension for dot product")
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self) -> float:
        return np.sqrt(sum(x**2 for x in self.data))

    def normalize(self) -> "Vector":
        mag = self.magnitude()
        return Vector(data=[x / mag for x in self.data], dim=self.dim)


class MultimodalDocument(BaseDocument):
    components: List[BaseDocument]

    def __init__(self, **data):
        super().__init__(data_type=DataType.MULTIMODAL, **data)


class DocumentCollection(BaseModel):
    documents: List[BaseDocument]

    model_config = ConfigDict(arbitrary_types_allowed=True)
