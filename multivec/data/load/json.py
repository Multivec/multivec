from typing import Union, Dict, List, Any, Optional
from pathlib import Path
import json
from multivec.utils.base_format import (
    TextDocument,
    ImageDocument,
    AudioDocument,
    VideoDocument,
    MultimodalDocument,
    BaseDocument,
    DocumentCollection
)

class JsonLoader:
    def __init__(
        self,
        json_input: Union[str, Path, Dict[str, Any], List[Any]],
        text_content_path: Optional[str] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None
    ):
        """
        Initialize the JsonLoader with either a file path or JSON data.

        Args:
            json_input: Path to JSON file, or JSON data as a dictionary or list.
            text_content_path: Path to the text content within the JSON structure.
            image_path: Path to the image content within the JSON structure.
            audio_path: Path to the audio content within the JSON structure.
            video_path: Path to the video content within the JSON structure.
        """
        self.json_input = json_input
        self.text_content_path = text_content_path
        self.image_path = image_path
        self.audio_path = audio_path
        self.video_path = video_path

    def load(self) -> DocumentCollection:
        """
        Load and process JSON data into a DocumentCollection.

        Returns:
            DocumentCollection containing the processed documents.

        Raises:
            JSONDecodeError: If the JSON is invalid.
            IOError: If there's an error reading the file.
            ValueError: If the json_input is of an invalid type.
        """
        if isinstance(self.json_input, (str, Path)):
            data = self._load_from_file(self.json_input)
        elif isinstance(self.json_input, (dict, list)):
            data = self.json_input
        else:
            raise ValueError(f"Invalid json_input type: {type(self.json_input)}")

        documents = self._process_data(data)
        return DocumentCollection(documents=documents)

    @staticmethod
    def _load_from_file(file_path: Union[str, Path]) -> Union[Dict[str, Any], List[Any]]:
        """Load JSON data from a file."""
        file_path = Path(file_path)
        if not file_path.exists() or file_path.suffix.lower() != '.json':
            raise ValueError(f"Invalid JSON file path: {file_path}")

        try:
            with file_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in {file_path}: {e}", e.doc, e.pos) from e
        except IOError as e:
            raise IOError(f"Error reading file {file_path}: {e}") from e

    def _process_data(self, data: Union[Dict[str, Any], List[Any]], path: str = '') -> List[BaseDocument]:
        """
        Recursively process the JSON data to create Document objects.

        Args:
            data: The JSON data to process.
            path: The current path within the JSON structure.

        Returns:
            List of BaseDocument objects created from the JSON data.
        """
        documents = []

        if isinstance(data, dict):
            documents.extend(self._process_dict(data, path))
        elif isinstance(data, list):
            documents.extend(self._process_list(data, path))
        else:
            document = self._create_document(data, path)
            if document:
                documents.append(document)

        return documents

    def _process_dict(self, data: Dict[str, Any], path: str) -> List[BaseDocument]:
        """Process dictionary data."""
        documents = []
        for key, value in data.items():
            new_path = f"{path}/{key}" if path else key
            if self._is_multimodal(value):
                documents.append(self._create_multimodal_document(value, new_path))
            else:
                documents.extend(self._process_data(value, new_path))
        return documents

    def _process_list(self, data: List[Any], path: str) -> List[BaseDocument]:
        """Process list data."""
        documents = []
        for index, item in enumerate(data):
            new_path = f"{path}/{index}"
            documents.extend(self._process_data(item, new_path))
        return documents

    def _create_document(self, data: Any, path: str) -> Optional[BaseDocument]:
        """Create a Document object based on the data and path."""
        if self._matches_path(path, self.text_content_path):
            return TextDocument(content=str(data), metadata={'path': path})
        elif self._matches_path(path, self.image_path):
            return ImageDocument(image_path=str(data), metadata={'path': path})
        elif self._matches_path(path, self.audio_path):
            return AudioDocument(audio_url=str(data), metadata={'path': path})
        elif self._matches_path(path, self.video_path):
            return VideoDocument(video_url=str(data), metadata={'path': path})
        return None

    def _create_multimodal_document(self, data: Dict[str, Any], path: str) -> MultimodalDocument:
        """Create a MultimodalDocument from a dictionary containing multiple modalities."""
        components = []
        if self.text_content_path and self.text_content_path in data:
            components.append(TextDocument(content=str(data[self.text_content_path]), metadata={'path': f"{path}/{self.text_content_path}"}))
        if self.image_path and self.image_path in data:
            components.append(ImageDocument(image_path=str(data[self.image_path]), metadata={'path': f"{path}/{self.image_path}"}))
        if self.audio_path and self.audio_path in data:
            components.append(AudioDocument(audio_url=str(data[self.audio_path]), metadata={'path': f"{path}/{self.audio_path}"}))
        if self.video_path and self.video_path in data:
            components.append(VideoDocument(video_url=str(data[self.video_path]), metadata={'path': f"{path}/{self.video_path}"}))
        return MultimodalDocument(components=components, metadata={'path': path})

    def _is_multimodal(self, data: Any) -> bool:
        """Check if the data represents a multimodal document."""
        if not isinstance(data, dict):
            return False
        return sum(1 for path in [self.text_content_path, self.image_path, self.audio_path, self.video_path] if path and path in data) > 1

    @staticmethod
    def _matches_path(current_path: str, target_path: Optional[str]) -> bool:
        """Check if the current path matches the target path."""
        return bool(target_path) and current_path.endswith(target_path)