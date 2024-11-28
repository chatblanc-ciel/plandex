import dataclasses
import json
import os
import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import logging
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
import openai
from .common import StreamRequest
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming

logger = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True)
class AzureOpenAIConfig:
    """Azure OpenAI service configurations"""

    endpoint: str
    api_key: str
    deployment_name: str
    api_version: str = "2024-03-01-preview"

    @staticmethod
    def new_from_env() -> "AzureOpenAIConfig":
        endpoint_env = os.getenv("AZURE_OPENAI_ENDPOINT")
        if isinstance(endpoint_env, str) and endpoint_env != "":
            endpoint: str = endpoint_env
        else:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
        api_key_env = os.getenv("AZURE_OPENAI_API_KEY")
        if isinstance(api_key_env, str) and api_key_env != "":
            api_key: str = api_key_env
        else:
            raise ValueError("AZURE_OPENAI_API_KEY is not set")
        deployment_name_env = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if isinstance(deployment_name_env, str) and deployment_name_env != "":
            deployment_name: str = deployment_name_env
        else:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME is not set")

        return AzureOpenAIConfig(
            endpoint=endpoint,
            api_key=api_key,
            deployment_name=deployment_name,
        )


async def send_token(request: CompletionCreateParamsStreaming):
    llm_config = AzureOpenAIConfig.new_from_env()
    client = openai.AzureOpenAI(
        azure_endpoint=llm_config.endpoint,
        api_key=llm_config.api_key,
        api_version=llm_config.api_version,
    )

    response: openai.Stream[ChatCompletionChunk] = client.chat.completions.create(
        model=llm_config.deployment_name,
        temperature=request.get("temperature"),
        top_p=request.get("top_p"),
        messages=request.get("messages"),
        stream=request.get("stream"),
    )
    for chunk in response:
        if len(chunk.choices) > 0:
            data = f"data: {chunk.to_json(indent=None, exclude_unset=False,use_api_names=False)}\n\n"
            logger.debug(data)
            yield data
        else:
            continue
    yield "data: [DONE]"


def proxy_openai(request: CompletionCreateParamsStreaming) -> StreamingResponse:
    logger.debug(
        json.dumps(
            {
                "msg": "proxy_openai",
                "request": request,
            },
            indent=2,
        )
    )
    try:
        return StreamingResponse(
            send_token(request),
            media_type="text/event-stream",
            headers={
                "Transfer-Encoding": "chunked",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
