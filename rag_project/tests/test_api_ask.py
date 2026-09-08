def test_health_reports_loaded_pipeline(
    client,
    fake_pipeline,
):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["pipeline_loaded"] is True


def test_ask_returns_answer_and_sources(
    client,
    fake_pipeline,
):
    response = client.post(
        "/ask",
        json={
            "query": "RAG由哪三个阶段组成？",
            "top_k": 4,
            "min_similarity": 0.5,
            "max_new_tokens": 256,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "RAG由哪三个阶段组成？"
    assert data["passed"] is True
    assert data["max_score"] == 0.88
    assert data["answer"] == "RAG由检索、增强和生成三个阶段组成。"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["title"] == "RAG基础"
    assert data["sources"][0]["chunk_id"] == "doc_001_chunk_1"

    assert fake_pipeline.last_call == {
        "query": "RAG由哪三个阶段组成？",
        "top_k": 4,
        "min_similarity": 0.5,
        "max_new_tokens": 256,
    }

    assert "internal_debug" not in data


def test_ask_returns_insufficient_information(
    client,
    fake_pipeline,
):
    response = client.post(
        "/ask",
        json={
            "query": "世界最高峰是什么？",
            "min_similarity": 0.6,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["passed"] is False
    assert data["answer"] == "现有资料不足"
    assert data["max_score"] == 0.24
    assert data["sources"] == []


def test_ask_converts_pipeline_error_to_500(
    client,
    fake_pipeline,
):
    response = client.post(
        "/ask",
        json={
            "query": "触发异常",
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "RAG问答处理失败，请查看服务器日志",
    }