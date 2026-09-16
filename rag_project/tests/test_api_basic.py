def test_root_returns_api_information(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "RAG API 已启动",
        "docs": "/docs",
        "chat": "/chat",
    }


def test_chat_page_and_static_assets_are_available(client):
    page = client.get("/chat")
    stylesheet = client.get("/web/styles.css")
    script = client.get("/web/app.js")

    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "向知识库提问" in page.text
    assert stylesheet.status_code == 200
    assert "text/css" in stylesheet.headers["content-type"]
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]


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
    assert data["top_k"] == 5
    assert data["min_similarity"] == 0.48
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
