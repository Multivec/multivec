import os
import tempfile
from typing import List, Optional
import pymupdf

from multivec.utils.base_format import ImageDocument, TextDocument


class PDFLoader:
    """
    PDFLoader class is used to extract images and text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.
        path_to_output_imgs (str): Path to the directory where images will be saved.
        pdf file: The PDF file to be loaded.
    """

    def __init__(self, pdf_path: str, path_to_output_imgs: Optional[str] = None):
        self.pdf_path = pdf_path
        if path_to_output_imgs is None:
            self.path_to_output_imgs = tempfile.mkdtemp()
        else:
            self.path_to_output_imgs = path_to_output_imgs

        self.pdf = pymupdf.open(self.pdf_path)

    def extract_text(self) -> List[TextDocument]:
        """
        Extracts text from the PDF file.

        Returns:
            List[TextDocument]: A list of TextDocument objects.
        """
        texts = []
        for page_num in range(len(self.pdf)):
            page = self.pdf[page_num]
            text_content = page.get_text().encode("utf-8")
            texts.append(
                TextDocument(
                    content=text_content, metadata={"type": "text"}, page_index=page_num
                )
            )
        return texts

    def extract_images(self) -> List[ImageDocument]:
        """
        Extracts images from the PDF file.

        Returns:
            List[str]: A list of paths to the extracted images.
        """
        image_docs: List[ImageDocument] = []
        for page_num in range(len(self.pdf)):
            images = self.pdf[page_num].get_images()
            for image_index, image in enumerate(images):
                xref = image[0]
                pixmap = pymupdf.Pixmap(self.pdf, xref)

                # Convert CMYK or grayscale images to RGB
                if pixmap.n - pixmap.alpha > 3:
                    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
                
                # Create the output directory if it doesn't exist
                if not os.path.exists(self.path_to_output_imgs):
                    os.makedirs(self.path_to_output_imgs)

                image_path = f"{self.path_to_output_imgs}/image_{page_num}_{image_index}.png"
                pixmap.save(image_path)
                image_docs.append(
                    ImageDocument(
                        image_path=image_path,
                        metadata={"type": "image"},
                        page_index=page_num,
                    )
                )
        return images