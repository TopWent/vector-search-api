from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

# embed/upsert text fields share this cap so one request cannot blow up memory
BoundedText = Annotated[str, StringConstraints(min_length=1, max_length=32768)]


class Document(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    text: BoundedText
    metadata: dict[str, str] = Field(default_factory=dict)


class UpsertRequest(BaseModel):
    documents: list[Document] = Field(min_length=1, max_length=1000)


class UpsertResponse(BaseModel):
    upserted: int
    total: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)


class SearchHit(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict[str, str]


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class EmbedRequest(BaseModel):
    texts: list[BoundedText] = Field(min_length=1, max_length=256)


class EmbedResponse(BaseModel):
    dim: int
    vectors: list[list[float]]
