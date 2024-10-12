from typing import Optional
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError
from tenacity import retry, stop_after_attempt, wait_fixed

from multivec.providers.base import BaseCloud, CloudProviderType


class AzureConnector(BaseCloud):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(CloudProviderType.AZURE, api_key)
        self.blob_service_client = None

    def connect(self):
        api_key = self.auth.get_key(self.provider.value)
        if not api_key:
            raise ValueError(f"No API key found for {self.provider}")

        self.blob_service_client = BlobServiceClient.from_connection_string(api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def list_containers(self):
        if not self.blob_service_client:
            self.connect()
        try:
            containers = self.blob_service_client.list_containers()
            return [container.name for container in containers]
        except AzureError as e:
            print(f"An error occurred: {e}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def pull_data_from_blob(self, container_name: str, blob_name: str) -> bytes:
        if not self.blob_service_client:
            self.connect()
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )
            download_stream = blob_client.download_blob()
            return download_stream.readall()
        except AzureError as e:
            print(f"An error occurred: {e}")
            return b""
