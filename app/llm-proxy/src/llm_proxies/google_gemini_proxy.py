import json
from typing import Any, Iterable, Literal
import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import logging
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
import openai
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from vertexai.generative_models._generative_models import GenerationResponse, Content
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta


logger = logging.getLogger(__name__)


def transrate_msg(msg) -> list[dict[str, Any]]:
    gemini_msg = []

    for m in msg:
        if m["content"] == "":
            continue
        if m["role"] == "user":
            gemini_msg.append(
                Content(role="user", parts=[Part.from_text(m["content"])])
            )
        elif m["role"] == "assistant":
            gemini_msg.append(
                Content(role="model", parts=[Part.from_text(m["content"])])
            )
    return gemini_msg


def generate(
    generation_config, safety_settings, request: CompletionCreateParamsStreaming
):
    vertexai.init(project="lf-rd-zoo", location="us-central1")
    model = GenerativeModel(
        "gemini-1.5-pro-002",
        system_instruction=[
            """ソフトウェアエンジニアとしてコンテキストを深く理解して、ユーザーに有用なフィードバックを行ってください。"""
        ],
    )
    responses: Iterable[GenerationResponse] = model.generate_content(
        transrate_msg(request["messages"]),
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    for response in responses:
        logger.info(response.text)


def gen(request: CompletionCreateParamsStreaming):
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }

    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
    ]

    generate(generation_config, safety_settings, request)


async def send_token(request: CompletionCreateParamsStreaming):
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }

    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
    ]

    vertexai.init(project="lf-rd-zoo", location="us-central1")
    model = GenerativeModel(
        "gemini-1.5-pro-002",
        system_instruction=[
            """ソフトウェアエンジニアとしてコンテキストを深く理解して、ユーザーに有用なフィードバックを行ってください。"""
        ],
    )
    responses: Iterable[GenerationResponse] = model.generate_content(
        transrate_msg(request["messages"]),
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    for res in responses:
        if res.text == "":
            continue
        chunk = ChatCompletionChunk(
            id="",
            created=0,
            model=f"{model._model_name}/{res._raw_response.model_version}",
            object="chat.completion.chunk",
            choices=[Choice(delta=ChoiceDelta(content=res.text), index=0)],
        )
        data = f"data: {chunk.to_json(indent=None, exclude_unset=False,use_api_names=False)}\n\n"
        logger.debug(data)
        yield data
    yield "data: [DONE]"


def proxy_google_gemini(request: CompletionCreateParamsStreaming) -> StreamingResponse:
    # gen(request)
    # raise HTTPException(status_code=500, detail="Internal Server Error")
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
