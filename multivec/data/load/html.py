import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
import io
import base64

from multivec.utils.base_format import ImageDocument, TextDocument
from multivec.exceptions import HTMLLoaderError

class HTMLLoader:
    """Load HTML file, extract text and images, and highlight important keywords in images."""

    def __init__(self, html_source: str, is_url: bool = False, output_dir: Optional[str] = None):
        self.html_source = html_source
        self.is_url = is_url
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.soup = self._load_html()

    def _load_html(self) -> BeautifulSoup:
        """Load HTML content from file or URL."""
        try:
            if self.is_url:
                response = requests.get(self.html_source)
                response.raise_for_status()
                return BeautifulSoup(response.content, 'html.parser')
            else:
                with open(self.html_source, 'r', encoding='utf-8') as file:
                    return BeautifulSoup(file, 'html.parser')
        except (requests.RequestException, IOError) as e:
            raise HTMLLoaderError(f"Failed to load HTML: {str(e)}")

    def extract_text(self) -> List[TextDocument]:
        """Extract text content from HTML."""
        texts = []
        for i, paragraph in enumerate(self.soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])):
            text_content = paragraph.get_text().strip()
            if text_content:
                texts.append(
                    TextDocument(
                        content=text_content,
                        metadata={"type": "text", "tag": paragraph.name},
                        page_index=i
                    )
                )
        return texts

    def extract_images(self) -> List[ImageDocument]:
        """Extract images from HTML."""
        images = []
        for i, img in enumerate(self.soup.find_all('img')):
            src = img.get('src')
            if src:
                try:
                    if src.startswith('data:image'):
                        # Handle base64 encoded images
                        image_data = base64.b64decode(src.split(',')[1])
                        image = Image.open(io.BytesIO(image_data))
                    elif src.startswith(('http://', 'https://')):
                        # Handle remote images
                        response = requests.get(src)
                        response.raise_for_status()
                        image = Image.open(io.BytesIO(response.content))
                    else:
                        # Handle local images
                        image_path = Path(self.html_source).parent / src if not self.is_url else Path(src)
                        image = Image.open(image_path)

                    # Save the image
                    image_filename = f"image_{i}.png"
                    image_path = self.output_dir / image_filename
                    image.save(image_path)

                    images.append(
                        ImageDocument(
                            content=str(image_path),
                            metadata={"type": "image", "original_src": src},
                            page_index=i
                        )
                    )
                except Exception as e:
                    print(f"Failed to process image {src}: {str(e)}")
        return images

    def highlight_keywords(self, image_path: str, keywords: List[str]) -> str:
        """Highlight keywords in the image."""
        try:
            with Image.open(image_path) as img:
                draw = ImageDraw.Draw(img)
                font = ImageFont.load_default()

                for i, keyword in enumerate(keywords):
                    position = (10, 10 + i * 20)
                    draw.text(position, keyword, fill="red", font=font)

                highlighted_path = self.output_dir / f"highlighted_{Path(image_path).name}"
                img.save(highlighted_path)
                return str(highlighted_path)
        except Exception as e:
            print(f"Failed to highlight keywords in {image_path}: {str(e)}")
            return image_path

    def process(self, keywords: Optional[List[str]] = None) -> Tuple[List[TextDocument], List[ImageDocument]]:
        """Process the HTML file, extract text and images, and highlight keywords in images."""
        text_docs = self.extract_text()
        image_docs = self.extract_images()

        if keywords:
            highlighted_image_docs = []
            for img_doc in image_docs:
                highlighted_path = self.highlight_keywords(img_doc.content, keywords)
                highlighted_image_docs.append(
                    ImageDocument(
                        content=highlighted_path,
                        metadata={**img_doc.metadata, "highlighted": True, "keywords": keywords},
                        page_index=img_doc.page_index
                    )
                )
            image_docs = highlighted_image_docs

        return text_docs, image_docs

# Usage example:
# loader = HTMLLoader("http://example.com", is_url=True)
# text_docs, image_docs = loader.process(keywords=["example", "important"])