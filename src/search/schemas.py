from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1)
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
    texts: list[str] = Field(min_length=1, max_length=256)


class EmbedResponse(BaseModel):
    dim: int
    vectors: list[list[float]]
