import re
import tempfile
from pathlib import Path
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from multivec.providers.base import BaseLLM
from multivec.utils.base_format import ImageDocument, TextDocument
from multivec.exceptions import HTMLLoaderError


class HTMLLoader:
    """Load HTML file, extract text and images, take image screenshots, and highlight important keywords in images."""

    def __init__(
        self,
        html_source: str,
        llm: Optional[BaseLLM] = None,
        output_dir: Optional[str] = None,
    ):
        self.html_source = html_source
        self.llm = llm
        self.is_url = bool(re.match(r"^https?://", html_source))
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.soup = self._load_html()

    def _load_html(self) -> BeautifulSoup:
        """Load HTML content from file or URL."""
        print("is url is ", self.is_url)
        try:
            if self.is_url:
                response = requests.get(self.html_source)
                response.raise_for_status()
                return BeautifulSoup(response.content, "html.parser")
            else:
                with open(self.html_source, "r", encoding="utf-8") as file:
                    return BeautifulSoup(file, "html.parser")
        except (requests.RequestException, IOError) as e:
            raise HTMLLoaderError(f"Failed to load HTML: {str(e)}")

    def extract(self) -> List[TextDocument | ImageDocument]:
        """Extract text content and images from HTML."""
        documents = []

        # Extract text
        logger.info("Scraped Content", self.soup, self.is_url)
        for i, paragraph in enumerate(
            self.soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        ):
            print(paragraph)
            text_content = paragraph.get_text().strip()
            if text_content:
                documents.append(
                    TextDocument(
                        content=text_content,
                        metadata={"type": "text", "tag": paragraph.name},
                        page_index=i,
                    )
                )

        # Extract images
        for i, img in enumerate(self.soup.find_all("img")):
            src = img.get("src")
            if src:
                try:
                    if src.startswith("data:image"):
                        # Handle base64 encoded images
                        image_data = base64.b64decode(src.split(",")[1])
                        image = Image.open(io.BytesIO(image_data))
                    elif src.startswith(("http://", "https://")):
                        # Handle remote images
                        response = requests.get(src)
                        response.raise_for_status()
                        image = Image.open(io.BytesIO(response.content))
                    else:
                        # Handle local images
                        image_path = (
                            Path(self.html_source).parent / src
                            if not self.is_url
                            else Path(src)
                        )
                        image = Image.open(image_path)

                    # Save the image
                    image_filename = f"image_{i}.png"
                    image_path = self.output_dir / image_filename
                    image.save(image_path)

                    documents.append(
                        ImageDocument(
                            image_path=str(image_path),
                            metadata={"type": "image", "original_src": src},
                            page_index=i,
                        )
                    )
                except Exception as e:
                    print(f"Failed to process image {src}: {str(e)}")

        # Take screenshot if it's a URL
        if self.is_url:
            screenshot_path = self.take_screenshot()
            documents.append(
                ImageDocument(
                    image_path=screenshot_path,
                    metadata={"type": "screenshot", "original_src": self.html_source},
                    page_index=len(documents),
                )
            )

        return documents

    def take_screenshot(self) -> str:
        """Take a screenshot of the web page."""
        if not self.is_url:
            raise HTMLLoaderError("Cannot take screenshot of local HTML file.")

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=options)
        try:
            driver.get(self.html_source)
            screenshot_path = self.output_dir / "screenshot.png"
            driver.save_screenshot(str(screenshot_path))
            return str(screenshot_path)
        finally:
            driver.quit()

    def highlight_keywords(self, image_path: str, keywords: List[str]) -> str:
        """Highlight keywords in the image."""
        try:
            with Image.open(image_path) as img:
                draw = ImageDraw.Draw(img)
                font = ImageFont.load_default()

                for i, keyword in enumerate(keywords):
                    position = (10, 10 + i * 20)
                    draw.text(position, keyword, fill="red", font=font)

                highlighted_path = (
                    self.output_dir / f"highlighted_{Path(image_path).name}"
                )
                img.save(highlighted_path)
                return str(highlighted_path)
        except Exception as e:
            print(f"Failed to highlight keywords in {image_path}: {str(e)}")
            return image_path

    def process(
        self, keywords: Optional[List[str]] = None
    ) -> List[TextDocument | ImageDocument]:
        """Process the HTML file, extract text and images, take screenshot, and highlight keywords in images."""
        documents = self.extract()

        if not keywords and self.llm:
            # Extract keywords using LLM if no keywords are provided
            text_content = " ".join(
                doc.content for doc in documents if isinstance(doc, TextDocument)
            )
            keywords = self.llm.generate(
                f"Extract important keywords from the following text, output are words separated by a comma:\n\n{text_content}"
            ).split(",")
            keywords = [keyword.strip() for keyword in keywords]

        if keywords:
            highlighted_documents = []
            for doc in documents:
                if isinstance(doc, ImageDocument):
                    highlighted_path = self.highlight_keywords(doc.content, keywords)
                    highlighted_documents.append(
                        ImageDocument(
                            image_path=highlighted_path,
                            metadata={
                                **doc.metadata,
                                "highlighted": True,
                                "keywords": keywords,
                            },
                            page_index=doc.page_index,
                            width=doc.width,
                            height=doc.height,
                        )
                    )
                else:
                    highlighted_documents.append(doc)
            documents = highlighted_documents

        return documents


# Usage example:
# from multivec.providers.llm.openai import ChatOpenAI
# llm = ChatOpenAI(args)
# loader = HTMLLoader("http://example.com", llm=llm)
# documents = loader.process()
