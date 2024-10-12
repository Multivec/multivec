from typing import Optional
from ..base import BaseLLM, BedrockModel
import boto3
import json


class ChatBedrock(BaseLLM):
    """
    A class for interacting with Amazon Bedrock models to generate completions
    using the AWS Bedrock API. It manages the connection to the AWS Bedrock
    runtime, handles authentication, and invokes models for text generation.

    Attributes:
    - model: The specific Bedrock model to be used.
    - region_name: The AWS region where the Bedrock model is hosted.
    - api_key: (Optional) The API key for authenticating with AWS Bedrock. If not provided,
      it retrieves the key from the authentication module.

    Methods:
    - generate(prompt: str) -> str: Sends the prompt to the Bedrock model and returns the
      generated completion.
    """

    def __init__(
        self, model: BedrockModel, region_name: str, api_key: Optional[str] = None
    ):
        super().__init__("bedrock", api_key)
        self.model = model
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region_name,
            aws_access_key_id=self.auth.get_key("bedrock"),
        )

    def generate(self, prompt: str) -> str:
        response = self.client.invoke_model(
            modelId=self.model, body=json.dumps({"prompt": prompt})
        )
        return json.loads(response["body"].read())["completion"]
