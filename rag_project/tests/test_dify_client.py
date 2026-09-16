import httpx
import pytest

from rag_project.dify_client import (
    DifyAPIError,
    DifyClient,
    DifyConfigurationError,
)


def make_client() -> DifyClient:
    return DifyClient(
        base_url="https://api.dify.example/v1/",
        api_key="app-test-key",
        user="test-user",
    )


def test_requires_server_side_api_key():
    with pytest.raises(DifyConfigurationError, match="DIFY_API_KEY"):
        DifyClient(
            base_url="https://api.dify.example/v1",
            api_key="",
            user="test-user",
        )


def test_generate_calls_chat_messages(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(
            200,
            json={
                "answer": "Dify 生成的回答",
                "conversation_id": "conversation-1",
                "message_id": "message-1",
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("rag_project.dify_client.httpx.post", fake_post)

    result = make_client().generate("请根据资料回答")

    assert result.answer == "Dify 生成的回答"
    assert result.conversation_id == "conversation-1"
    assert captured["url"] == "https://api.dify.example/v1/chat-messages"
    assert captured["headers"]["Authorization"] == "Bearer app-test-key"
    assert captured["json"]["response_mode"] == "blocking"
    assert captured["json"]["query"] == "请根据资料回答"


def test_generate_reports_dify_http_error(monkeypatch):
    def fake_post(url, **kwargs):
        return httpx.Response(
            401,
            text="invalid token",
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("rag_project.dify_client.httpx.post", fake_post)

    with pytest.raises(DifyAPIError, match="HTTP 401"):
        make_client().generate("测试")


def test_generate_rejects_invalid_payload(monkeypatch):
    def fake_post(url, **kwargs):
        return httpx.Response(
            200,
            json={"conversation_id": "conversation-1"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("rag_project.dify_client.httpx.post", fake_post)

    with pytest.raises(DifyAPIError, match="answer"):
        make_client().generate("测试")
