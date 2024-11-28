from typing import Optional
from pydantic import BaseModel


class StreamRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: Optional[float]
    top_p: Optional[float]
    stream: Optional[bool]
