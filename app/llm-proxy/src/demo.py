import logging
from dotenv import load_dotenv
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from llm_proxies.openai_proxy import AzureOpenAIConfig
import openai

load_dotenv()
logging.basicConfig(level=logging.DEBUG)

llm_config = AzureOpenAIConfig.new_from_env()
client = openai.AzureOpenAI(
    azure_endpoint=llm_config.endpoint,
    api_key=llm_config.api_key,
    api_version=llm_config.api_version,
)
client = openai.OpenAI(
    api_key=llm_config.api_key,
    base_url="http://localhost:18001/openai",
)
msg: list[dict[str, str]] = [
    {
        "role": "system",
        "content": "Hello, how can I help you today?",
    },
    {
        "role": "user",
        "content": "I'd like to book a flight to London.",
    },
]

response: openai.Stream[ChatCompletionChunk] = client.chat.completions.create(
    model=llm_config.deployment_name,
    temperature=0.0,
    top_p=0.5,
    messages=[m for m in msg],
    stream=True,
)
print(f"\n\nResponse:\n{response}\n\n")
for chunk in response:
    print(f"\n\nChunk:\n{chunk}\n\n")
    if len(chunk.choices) == 0:
        print("  No response from the model")
    else:
        chunk_message = chunk.choices[0].delta.content
        print(chunk_message)
    # raise ValueError("Unexpected response from the model")
