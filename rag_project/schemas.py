from pydantic import BaseModel, Field, field_validator

class AskRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description="用户提出的问题",
    )
    top_k: int = Field( #top
        default=3,
        ge=1,
        le=10,
        description="最多检索的文本块数量",
    )
    min_similarity: float = Field(
        default=0.40,
        ge=-1.0,
        le=1.0,
        description="最低相关性门槛",
    )
    max_new_tokens: int = Field(
        default=200,
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


class AskResponse(BaseModel):
    query: str
    answer: str
    passed: bool
    max_score: float
    sources: list[SourceItem]