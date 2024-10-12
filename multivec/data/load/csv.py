import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import io
import matplotlib.pyplot as plt

from multivec.utils.base_format import ImageDocument, TextDocument
from multivec.providers.cloud.aws import AWSConnector
from multivec.providers.cloud.azure import AzureConnector


class CSVLoader:
    """
    Load CSV or Excel (XLSX) File from local storage, AWS S3, or Azure Blob Storage,
    create visualizations, and highlight important information using pandas.

    Usage example:
    Local file:
    loader = CSVLoader("path/to/local/file.csv") \n
    documents = loader.process() \n

    S3 file:
    aws_connector = AWSConnector(api_key="your_aws_api_key") \n
    loader = CSVLoader("s3://your-bucket/path/to/file.csv", aws_connector=aws_connector) \n
    documents = loader.process() \n

    Azure Blob Storage file:
    azure_connector = AzureConnector(api_key="your_azure_connection_string") \n
    loader = CSVLoader("azure://your-container/path/to/file.csv", azure_connector=azure_connector) \n
    documents = loader.process() \n
    """

    def __init__(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        aws_connector: Optional[AWSConnector] = None,
        azure_connector: Optional[AzureConnector] = None,
    ):
        self.file_path = file_path
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.aws_connector = aws_connector
        self.azure_connector = azure_connector
        self.df = self._load_file()

    def _load_file(self) -> pd.DataFrame:
        if self.file_path.startswith("s3://") and self.aws_connector:
            return self._load_from_s3()
        elif self.file_path.startswith("azure://") and self.azure_connector:
            return self._load_from_azure()
        else:
            return self._load_from_local()

    def _load_from_local(self) -> pd.DataFrame:
        file_path = Path(self.file_path)
        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        else:
            raise ValueError(
                "Unsupported file format. Please use CSV or Excel (XLSX/XLS) files."
            )

    def _load_from_s3(self) -> pd.DataFrame:
        if not self.aws_connector:
            raise ValueError("AWS Connector is not initialized.")

        bucket_name, object_key = self._parse_s3_path(self.file_path)
        file_content = self.aws_connector.pull_data_from_s3(bucket_name, object_key)

        return self._load_from_bytes(file_content, object_key)

    def _load_from_azure(self) -> pd.DataFrame:
        if not self.azure_connector:
            raise ValueError("Azure Connector is not initialized.")

        container_name, blob_name = self._parse_azure_path(self.file_path)
        file_content = self.azure_connector.pull_data_from_blob(
            container_name, blob_name
        )

        return self._load_from_bytes(file_content, blob_name)

    def _load_from_bytes(self, file_content: bytes, file_name: str) -> pd.DataFrame:
        if file_name.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(file_content))
        elif file_name.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(file_content))
        else:
            raise ValueError(
                "Unsupported file format. Please use CSV or Excel (XLSX/XLS) files."
            )

    def _parse_s3_path(self, s3_path: str) -> Tuple[str, str]:
        parts = s3_path.replace("s3://", "").split("/")
        bucket_name = parts[0]
        object_key = "/".join(parts[1:])
        return bucket_name, object_key

    def _parse_azure_path(self, azure_path: str) -> Tuple[str, str]:
        parts = azure_path.replace("azure://", "").split("/")
        container_name = parts[0]
        blob_name = "/".join(parts[1:])
        return container_name, blob_name

    def _data_to_image(self) -> Image.Image:
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Hide axes
        ax.axis("off")

        # Create a table and add it to the axis
        table = ax.table(
            cellText=self.df.values,
            colLabels=self.df.columns,
            cellLoc="center",
            loc="center",
        )

        # Adjust table style
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.2)

        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)

        # Create PIL Image from BytesIO
        img = Image.open(buf)

        plt.close(fig)

        return img

    def _highlight_important_info(self, image: Image.Image) -> Image.Image:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        # Example: Highlight cells with values above the mean
        mean_value = self.df.select_dtypes(include=[np.number]).mean().mean()
        highlight_coords = []

        for row, data in enumerate(self.df.values, start=1):
            for col, value in enumerate(data, start=1):
                if isinstance(value, (int, float)) and value > mean_value:
                    x = col * 100
                    y = row * 30
                    highlight_coords.append((x, y))

        for x, y in highlight_coords:
            draw.rectangle([x, y, x + 90, y + 20], outline="red", width=2)

        return image

    def process(self) -> Tuple[List[TextDocument], List[ImageDocument]]:
        """
        Process the CSV or Excel file, create visualizations, and highlight important information.

        Returns:
            Tuple[List[TextDocument], List[ImageDocument]]: Lists of text and image documents.
        """
        text_content = self.df.to_csv(index=False)
        text_format = "csv"

        if self.file_path.startswith("s3://") or self.file_path.startswith("azure://"):
            file_type = self.file_path.split(".")[-1].lower()
        else:
            file_type = Path(self.file_path).suffix.lower()[1:]

        if file_type in ["xlsx", "xls"]:
            text_format = "csv (converted from xlsx)"

        text_docs = [
            TextDocument(
                content=text_content,
                metadata={"type": "text", "format": text_format},
                page_index=0,
            )
        ]

        data_image = self._data_to_image()
        highlighted_image = self._highlight_important_info(data_image)

        prefix = (
            "s3_"
            if self.file_path.startswith("s3://")
            else "azure_"
            if self.file_path.startswith("azure://")
            else ""
        )
        image_path = self.output_dir / f"{prefix}data_highlighted.png"
        highlighted_image.save(image_path)

        image_docs = [
            ImageDocument(
                content=str(image_path),
                metadata={
                    "type": "image",
                    "format": "png",
                    "description": f"Highlighted {file_type.upper()} visualization",
                },
                page_index=0,
            )
        ]

        return text_docs, image_docs
