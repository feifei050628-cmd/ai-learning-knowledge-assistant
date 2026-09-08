def test_root_returns_api_information(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "RAG API 已启动",
        "docs": "/docs",
    }


def test_validate_accepts_valid_request(client):
    response = client.post(
        "/validate",
        json={
            "query": "什么是 RAG？",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "什么是 RAG？"
    assert data["top_k"] == 3
    assert data["min_similarity"] == 0.40
    assert data["max_new_tokens"] == 200


def test_validate_rejects_invalid_top_k(client):
    response = client.post(
        "/validate",
        json={
            "query": "什么是 RAG？",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_ask_returns_503_without_pipeline(client):
    response = client.post(
        "/ask",
        json={
            "query": "什么是 RAG？",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "RAG Pipeline 尚未加载完成",
    }