from typing import Optional
from botocore.exceptions import ClientError
import boto3
from tenacity import retry, stop_after_attempt, wait_fixed

from multivec.providers.base import BaseCloud, CloudProviderType


class AWSConnector(BaseCloud):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(CloudProviderType.AWS, api_key)
        self.s3_client = None

    def connect(self):
        api_key = self.auth.get_key(self.provider.value)
        if not api_key:
            raise ValueError(f"No API key found for {self.provider}")

        self.s3_client = boto3.client(
            "s3", aws_access_key_id=api_key, aws_secret_access_key=api_key
        )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def list_buckets(self):
        if not self.s3_client:
            self.connect()
        try:
            response = self.s3_client.list_buckets()
            return [bucket["Name"] for bucket in response["Buckets"]]
        except ClientError as e:
            print(f"An error occurred: {e}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def pull_data_from_s3(self, bucket_name: str, object_key: str) -> bytes:
        if not self.s3_client:
            self.connect()
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=object_key)
            return response["Body"].read()
        except ClientError as e:
            print(f"An error occurred: {e}")
            return b""
