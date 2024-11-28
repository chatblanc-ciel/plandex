import json
import logging
from fastapi import FastAPI, status
from dotenv import load_dotenv
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from llm_proxies.openai_proxy import proxy_openai
from llm_proxies.google_gemini_proxy import proxy_google_gemini
from starlette.requests import Request
from llm_proxies.common import StreamRequest
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# ----------------- Exception Handlers -----------------


@app.exception_handler(RequestValidationError)
async def handler(request: StreamRequest, exc: RequestValidationError):
    logger.error(
        json.dumps(
            {
                "error": "RequestValidationError",
                "detail": exc.errors(),
                "body": exc.body,
            },
            indent=2,
        )
    )

    return JSONResponse(content={}, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


# ----------------- basic endpoint -----------------


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# ----------------- llm proxy endpoint -----------------


@app.post("/openai/chat/completions")
async def handle_openai_request_(request: CompletionCreateParamsStreaming):
    logger.info(
        json.dumps(
            {
                "msg": "handle_openai_request_",
                # "request": vars(request),
            },
            # indent=2,
        )
    )
    return proxy_openai(request)


@app.post("/openai/chat/completions/{endpoint}")
async def handle_openai_request(endpoint: str, request: Request):
    return {"status": "ok", "message": "openai endpoint(argumented)"}
    return await proxy_openai(endpoint, request)


@app.post("/gemini/chat/completions")
async def handle_google_gemini_request(request: CompletionCreateParamsStreaming):
    return proxy_google_gemini(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=18000)
