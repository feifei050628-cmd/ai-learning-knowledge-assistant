from pydantic import BaseModel, Field, field_validator
from rag_project.config import (
    DEFAULT_MAX_NEW_TOKENS,     
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TOP_K,  
)

class AskRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description="用户提出的问题",
    )
    top_k: int = Field( #top
        default=DEFAULT_TOP_K,
        ge=1,
        le=10,
        description="最多检索的文本块数量",
    )
    min_similarity: float = Field(
        default=DEFAULT_MIN_SIMILARITY,
        ge=-1.0,
        le=1.0,
        description="最低相关性门槛",
    )
    max_new_tokens: int = Field(
        default=DEFAULT_MAX_NEW_TOKENS, 
        ge=1,
        le=512,
        description="最多生成的新Token数量",
    )

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("问题不能为空")

        return value


class SourceItem(BaseModel):
    rank: int
    title: str
    chunk_id: str
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


class AskResponse(BaseModel):
    query: str
    answer: str
    passed: bool
    max_score: float
    gate_score: float | None = None
    gate_status: str | None = None
    answerable: bool | None = None
    verification: dict | None = None
    sources: list[SourceItem]


class DocumentItem(BaseModel):
    id: str
    name: str
    type: str
    size_bytes: int
    updated_at: str
    status: str
    chunks: int
    error: str | None = None
    category: str
    read_only: bool


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]
    total: int
    ready: int
    chunks: int
