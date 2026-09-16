"""Dify Chat/Chatflow 应用的服务端客户端。"""

from dataclasses import dataclass

import httpx


class DifyConfigurationError(RuntimeError):
    """Dify 环境变量缺失或无效。"""


class DifyAPIError(RuntimeError):
    """Dify API 请求失败。"""


@dataclass(frozen=True)
class DifyResult:
    answer: str
    conversation_id: str = ""
    message_id: str = ""


class DifyClient:
    """调用 Dify `/chat-messages` 的阻塞模式客户端。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        user: str,
        timeout_seconds: float = 60,
        verify_ssl: bool = True,
    ) -> None:
        if not base_url.strip():
            raise DifyConfigurationError("DIFY_API_BASE_URL 不能为空")
        if not api_key.strip():
            raise DifyConfigurationError(
                "GENERATION_PROVIDER=dify 时必须设置 DIFY_API_KEY"
            )
        if not user.strip():
            raise DifyConfigurationError("DIFY_USER 不能为空")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.user = user
        self.timeout_seconds = timeout_seconds
        self.verify_ssl = verify_ssl

    def generate(self, prompt: str) -> DifyResult:
        if not prompt.strip():
            raise ValueError("发送给 Dify 的提示词不能为空")

        try:
            response = httpx.post(
                f"{self.base_url}/chat-messages",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": {},
                    "query": prompt,
                    "response_mode": "blocking",
                    "conversation_id": "",
                    "user": self.user,
                    "files": [],
                },
                timeout=self.timeout_seconds,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise DifyAPIError("Dify 请求超时") from error
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:500]
            raise DifyAPIError(
                f"Dify 返回 HTTP {error.response.status_code}: {detail}"
            ) from error
        except httpx.HTTPError as error:
            raise DifyAPIError(f"无法连接 Dify: {error}") from error

        try:
            payload = response.json()
            answer = str(payload["answer"]).strip()
        except (KeyError, TypeError, ValueError) as error:
            raise DifyAPIError("Dify 响应缺少有效的 answer 字段") from error

        if not answer:
            raise DifyAPIError("Dify 返回了空回答")

        return DifyResult(
            answer=answer,
            conversation_id=str(payload.get("conversation_id", "")),
            message_id=str(payload.get("message_id", "")),
        )
