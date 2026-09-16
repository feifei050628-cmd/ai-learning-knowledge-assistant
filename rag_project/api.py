import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag_project.config import PROJECT_NAME
from rag_project.rag_pipeline import RAGPipeline
from rag_project.schemas import AskRequest, AskResponse

from rag_project.update_knowledge_base import (
    main as update_knowledge_base,
)

logger = logging.getLogger(__name__)    #logger 用于记录日志信息，__name__ 表示当前模块的名称
WEB_DIR = Path(__file__).resolve().parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("正在检查知识库是否需要更新……")
    update_knowledge_base()

    print("\n正在加载 RAG Pipeline，请稍候……")
    app.state.pipeline = RAGPipeline()
    print("RAG Pipeline 加载完成")

    yield

    app.state.pipeline = None
    print("RAG Pipeline 已释放")

app = FastAPI(
    title=PROJECT_NAME,
    version="0.1.0",
    description="一个基于本地知识库的中文 RAG 问答 API",
    lifespan=lifespan,
)

app.mount(
    "/web",
    StaticFiles(directory=WEB_DIR),
    name="web",
)

@app.middleware("http")
async def add_utf8_charset(
    request: Request,
    call_next,
):
    response = await call_next(request)

    content_type = response.headers.get(
        "content-type",
        "",
    )

    if (
        content_type.startswith("application/json")
        and "charset=" not in content_type
    ):
        response.headers["content-type"] = (
            f"{content_type}; charset=utf-8"
        )

    return response


@app.get("/")
def read_root() -> dict:
    return {
        "message": "RAG API 已启动",
        "docs": "/docs",
        "chat": "/chat",
    }


@app.get("/chat", include_in_schema=False)
def read_chat_page() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health_check(request: Request) -> dict:     #request: Request: FastAPI 的请求对象，包含请求的所有信息
    return {
        "status": "ok",
        "project": PROJECT_NAME,
        "pipeline_loaded": request.app.state.pipeline is not None,  #app.state.pipeline 是否已加载
    }


@app.post("/validate", response_model=AskRequest)
def validate_request(request: AskRequest) -> AskRequest:
    return request


@app.post("/ask", response_model=AskResponse)
def ask_question(
    request_data: AskRequest,
    request: Request,
) -> dict:
    pipeline = getattr(
        request.app.state,
        "pipeline",
        None,
    )

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG Pipeline 尚未加载完成",
        )

    try:
        return pipeline.answer(
            query=request_data.query,
            top_k=request_data.top_k,
            min_similarity=request_data.min_similarity,
            max_new_tokens=request_data.max_new_tokens,
        )
    except Exception as error:
        logger.exception("处理RAG问答请求时发生错误")

        raise HTTPException(
            status_code=500,
            detail="RAG问答处理失败，请查看服务器日志",
        ) from error


if __name__ == "__main__":
    print(
        "请在项目父目录运行："
        "python -m uvicorn rag_project.api:app --reload"
    )
