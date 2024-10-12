import tempfile
from pathlib import Path
from typing import List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2

from PIL import ImageDraw
import pytesseract

from multivec.providers.base import BaseLLM
from multivec.utils.base_format import ImageDocument, TextDocument
from multivec.exceptions import LLMAndKeywordsNotFound, PDFLoaderError
from multivec.providers.cloud.aws import AWSConnector
from multivec.providers.cloud.azure import AzureConnector


class PDFLoader:
    """
    PDFLoader class for extracting images and text from PDF files.

    This class provides functionality to extract text and images from PDF files,
    with support for local files, AWS S3, and Azure Blob Storage.

    Attributes:
        pdf_path (Union[str, Path]): Path or URL to the PDF file.
        output_dir (Path): Directory where extracted images will be saved.
        pdf_document (fitz.Document): The opened PDF document.
        aws_connector (Optional[AWSConnector]): Connector for AWS S3.
        azure_connector (Optional[AzureConnector]): Connector for Azure Blob Storage.

    Args:
        pdf_path (Union[str, Path]): Path or URL to the PDF file.
        output_dir (Optional[Union[str, Path]]): Directory to save extracted images.
            If None, a temporary directory is created.
        aws_connector (Optional[AWSConnector]): Connector for AWS S3.
        azure_connector (Optional[AzureConnector]): Connector for Azure Blob Storage.

    Raises:
        PDFLoaderError: If there's an error opening or processing the PDF file.
    """

    def __init__(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        llm: Optional[BaseLLM] = None,
        aws_connector: Optional[AWSConnector] = None,
        azure_connector: Optional[AzureConnector] = None,
    ):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.aws_connector = aws_connector
        self.azure_connector = azure_connector
        self.pdf_document = self._open_pdf()

    def _open_pdf(self) -> fitz.Document:
        """
        Opens the PDF file from local storage, AWS S3, or Azure Blob Storage.

        Returns:
            fitz.Document: The opened PDF document.

        Raises:
            PDFLoaderError: If there's an error opening the PDF file.
        """
        try:
            if isinstance(self.pdf_path, str) and self.pdf_path.startswith("s3://"):
                return self._open_from_s3()
            elif isinstance(self.pdf_path, str) and self.pdf_path.startswith(
                "azure://"
            ):
                return self._open_from_azure()
            else:
                return fitz.open(self.pdf_path)
        except Exception as e:
            raise PDFLoaderError(f"Failed to open PDF file: {e}")

    def _open_from_s3(self) -> fitz.Document:
        """
        Opens a PDF file from AWS S3.

        Returns:
            fitz.Document: The opened PDF document.

        Raises:
            PDFLoaderError: If there's an error opening the PDF file from S3.
        """
        if not self.aws_connector:
            raise PDFLoaderError("AWS Connector is not initialized.")

        try:
            bucket_name, object_key = self._parse_s3_path(self.pdf_path)
            file_content = self.aws_connector.pull_data_from_s3(bucket_name, object_key)
            return fitz.open(stream=file_content, filetype="pdf")
        except Exception as e:
            raise PDFLoaderError(f"Failed to open PDF from S3: {e}")

    def _open_from_azure(self) -> fitz.Document:
        """
        Opens a PDF file from Azure Blob Storage.

        Returns:
            fitz.Document: The opened PDF document.

        Raises:
            PDFLoaderError: If there's an error opening the PDF file from Azure.
        """
        if not self.azure_connector:
            raise PDFLoaderError("Azure Connector is not initialized.")

        try:
            container_name, blob_name = self._parse_azure_path(self.pdf_path)
            file_content = self.azure_connector.pull_data_from_blob(
                container_name, blob_name
            )
            return fitz.open(stream=file_content, filetype="pdf")
        except Exception as e:
            raise PDFLoaderError(f"Failed to open PDF from Azure: {e}")

    @staticmethod
    def _parse_s3_path(s3_path: str) -> Tuple[str, str]:
        """Parses an S3 path into bucket name and object key."""
        parts = s3_path.replace("s3://", "").split("/", 1)
        return parts[0], parts[1]

    @staticmethod
    def _parse_azure_path(azure_path: str) -> Tuple[str, str]:
        """Parses an Azure Blob Storage path into container name and blob name."""
        parts = azure_path.replace("azure://", "").split("/", 1)
        return parts[0], parts[1]

    def extract_text(self) -> List[TextDocument]:
        """
        Extracts text from the PDF file.

        Returns:
            List[TextDocument]: A list of TextDocument objects, one for each page.

        Raises:
            PDFLoaderError: If there's an error extracting text from the PDF.
        """
        try:
            return [
                TextDocument(
                    content=page.get_text().encode("utf-8"),
                    metadata={"type": "text", "page": page_num},
                    page_index=page_num,
                )
                for page_num, page in enumerate(self.pdf_document)
            ]
        except Exception as e:
            raise PDFLoaderError(f"Failed to extract text from PDF: {e}")

    def extract_images(self) -> List[ImageDocument]:
        """
        Extracts embedded images and takes screenshots of each page from the PDF file.

        Returns:
            List[ImageDocument]: A list of ImageDocument objects.

        Raises:
            PDFLoaderError: If there's an error extracting images from the PDF.
        """
        image_docs = []
        try:
            for page_num, page in enumerate(self.pdf_document):
                # Extract embedded images
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = self.pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]

                    image_filename = f"embedded_image_p{page_num}_i{img_index}.png"
                    image_path = self.output_dir / image_filename

                    with open(image_path, "wb") as image_file:
                        image_file.write(image_bytes)

                    # Open the image to get its dimensions
                    with Image.open(image_path) as img:
                        width, height = img.size

                    image_docs.append(
                        ImageDocument(
                            image_path=str(image_path),
                            metadata={
                                "type": "embedded_image",
                                "page": page_num,
                                "index": img_index,
                                "format": base_image["ext"],
                            },
                            page_index=page_num,
                            width=width,
                            height=height,
                        )
                    )

                # Take screenshot of the entire page
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                screenshot_filename = f"page_screenshot_{page_num}.png"
                screenshot_path = self.output_dir / screenshot_filename

                img.save(str(screenshot_path))

                image_docs.append(
                    ImageDocument(
                        image_path=str(screenshot_path),
                        metadata={
                            "type": "page_screenshot",
                            "page": page_num,
                            "format": "png",
                        },
                        page_index=page_num,
                        width=pix.width,
                        height=pix.height,
                    )
                )

            return image_docs
        except Exception as e:
            raise PDFLoaderError(f"Error extracting images from PDF: {str(e)}")

    def process(self) -> Tuple[List[TextDocument], List[ImageDocument]]:
        """
        Processes the PDF file, extracting both text and images.

        Returns:
            Tuple[List[TextDocument], List[ImageDocument]]: Lists of extracted text and image documents.

        Raises:
            PDFLoaderError: If there's an error processing the PDF.
        """
        try:
            with ThreadPoolExecutor() as executor:
                text_future = executor.submit(self.extract_text)
                images_future = executor.submit(self.extract_images)

                text_docs = text_future.result()
                image_docs = images_future.result()

            return text_docs, image_docs
        except Exception as e:
            raise PDFLoaderError(f"Failed to process PDF: {e}")

    def augment(
        self,
        docs: Union[List[ImageDocument], List[TextDocument]],
        keywords: Optional[List[str]] = None,
    ) -> List[ImageDocument]:
        """
        Augments images by finding keywords and highlighting them.

        Args:
            keywords (List[str]): List of keywords to search for in the images.
            image_docs (List[ImageDocument]): List of ImageDocument objects to process.

        Returns:
           Union[List[ImageDocument], List[TextDocument]]: List of ImageDocument objects and TextDocument .

        Raises:
            PDFLoaderError: If there's an error processing the images.
        """
        if not self.keywords and self.llm:
            # Extract keywords using LLM if no keywords are provided
            text_content = " ".join(
                doc.content for doc in docs if isinstance(doc, TextDocument)
            )
            keywords = self.llm.generate(
                f"Extract important keywords from the following text, output are words separated by a comma:\n\n{text_content}"
            ).split(",")
            keywords = [keyword.strip() for keyword in keywords]

        if self.keywords:
            try:
                image_docs = [doc for doc in docs if isinstance(doc, ImageDocument)]
                augmented_docs = []
                for img_doc in image_docs:
                    # Open the image
                    img = Image.open(img_doc.image_path)
                    img_array = np.array(img)

                    # Convert to grayscale for OCR
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                    # Perform OCR
                    data = pytesseract.image_to_data(
                        gray, output_type=pytesseract.Output.DICT
                    )

                    # Create a drawing object
                    draw = ImageDraw.Draw(img)

                    # Highlight keywords
                    for i, word in enumerate(data["text"]):
                        if word.lower() in [kw.lower() for kw in keywords]:
                            x, y, w, h = (
                                data["left"][i],
                                data["top"][i],
                                data["width"][i],
                                data["height"][i],
                            )
                            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

                    # Save the augmented image
                    augmented_filename = (
                        f"augmented_{img_doc.image_path.split('/')[-1]}"
                    )
                    augmented_path = self.output_dir / augmented_filename
                    img.save(str(augmented_path))

                    # Create a new ImageDocument for the augmented image
                    augmented_doc = ImageDocument(
                        image_path=str(augmented_path),
                        metadata={**img_doc.metadata, "augmented": True},
                        page_index=img_doc.page_index,
                        width=img_doc.width,
                        height=img_doc.height,
                    )
                    augmented_docs.append(augmented_doc)

                return augmented_docs
            except Exception as e:
                raise PDFLoaderError(f"Error augmenting images: {str(e)}")
        else:
            raise LLMAndKeywordsNotFound(
                "LLM not found, please provide either keywords or llm in PDFLoader to augment images."
            )

    def process_with_augmentation(
        self, keywords: Optional[List[str]] = None
    ) -> Tuple[List[TextDocument], List[ImageDocument]]:
        """
        Processes the PDF file, extracting text and images, and augments the images with keyword highlighting.

        Args:
            keywords (List[str]): List of keywords to search for in the images.

        Returns:
            Tuple[List[TextDocument], List[ImageDocument]]: Lists of extracted text documents and augmented image documents.

        Raises:
            PDFLoaderError: If there's an error processing the PDF or augmenting images.
        """

        try:
            text_docs, image_docs = self.process()
            if not keywords and self.llm:
                # Extract keywords using LLM if no keywords are provided
                text_content = " ".join(doc.content for doc in text_docs)
                keywords = self.llm.generate(
                    f"Extract important keywords from the following text, output are words separated by a comma:\n\n{text_content}"
                ).split(",")
                keywords = [keyword.strip() for keyword in keywords]

            if keywords:
                augmented_image_docs = self.augment(keywords, image_docs)
                return text_docs, augmented_image_docs
            else:
                raise LLMAndKeywordsNotFound(
                    "LLM not found, please provide either keywords or llm in PDFLoader to augment images."
                )

        except Exception as e:
            raise PDFLoaderError(f"Failed to process PDF with augmentation: {e}")


# Usage examples:

# Local file:
# loader = PDFLoader("path/to/local/file.pdf")
# text_docs, image_docs = loader.process()

# S3 file:
# aws_connector = AWSConnector(api_key="your_aws_api_key")
# loader = PDFLoader("s3://your-bucket/path/to/file.pdf", aws_connector=aws_connector)
# text_docs, image_docs = loader.process()

# Azure Blob Storage file:
# azure_connector = AzureConnector(api_key="your_azure_connection_string")
# loader = PDFLoader("azure://your-container/path/to/file.pdf", azure_connector=azure_connector)
# text_docs, image_docs = loader.process()
