import pytest
from fastapi.testclient import TestClient

from rag_project.api import app


class FakePipeline:
    def __init__(self):
        self.last_call = None

    def answer(
        self,
        query: str,
        top_k: int,
        min_similarity: float,
        max_new_tokens: int,
    ) -> dict:
        self.last_call = {
            "query": query,
            "top_k": top_k,
            "min_similarity": min_similarity,
            "max_new_tokens": max_new_tokens,
        }

        if query == "触发异常":
            raise RuntimeError("模拟模型推理失败")

        if query == "世界最高峰是什么？":
            return {
                "query": query,
                "answer": "现有资料不足",
                "passed": False,
                "max_score": 0.24,
                "sources": [],
            }

        return {
            "query": query,
            "answer": "RAG由检索、增强和生成三个阶段组成。",
            "passed": True,
            "max_score": 0.88,
            "sources": [
                {
                    "rank": 1,
                    "title": "RAG基础",
                    "chunk_id": "doc_001_chunk_1",
                    "score": 0.88,
                }
            ],
            "internal_debug": "该字段不应该返回给客户端",
        }


@pytest.fixture
def client():
    test_client = TestClient(app)

    yield test_client

    test_client.close()


@pytest.fixture
def fake_pipeline():
    previous_pipeline = getattr(
        app.state,
        "pipeline",
        None,
    )

    pipeline = FakePipeline()
    app.state.pipeline = pipeline

    yield pipeline

    app.state.pipeline = previous_pipeline