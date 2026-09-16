import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag_project.config import GENERATION_PROVIDER, PROJECT_NAME
from rag_project.rag_pipeline import RAGPipeline
from rag_project.schemas import AskRequest, AskResponse
from rag_project.schemas import DocumentListResponse
from rag_project.document_service import (
    DOCUMENT_OPERATION_LOCK,
    DocumentConflictError,
    DocumentNotFoundError,
    delete_document,
    list_documents,
    mark_failed,
    resolve_document,
    save_upload,
    validate_document,
)

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
        "generation_provider": GENERATION_PROVIDER,
    }


def refresh_runtime_knowledge_base(request: Request) -> None:
    """重建向量数据并让当前检索 Pipeline 立即读取新版本。"""
    update_knowledge_base()
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError("RAG Pipeline 尚未加载完成")
    pipeline.reload_knowledge_base()


@app.get("/documents", response_model=DocumentListResponse)
def get_documents() -> dict:
    documents = list_documents()
    return {
        "documents": documents,
        "total": len(documents),
        "ready": sum(item["status"] == "ready" for item in documents),
        "chunks": sum(item["chunks"] for item in documents),
    }


@app.post("/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(request: Request, file: UploadFile = File(...)) -> dict:
    if not DOCUMENT_OPERATION_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="知识库正在处理其他文档，请稍后重试")

    path = None
    try:
        content = await file.read()
        path = await run_in_threadpool(save_upload, file.filename or "", content)
        await run_in_threadpool(refresh_runtime_knowledge_base, request)
        return {"message": "文档已上传并加入知识库", "document_id": resolve_document_id(path)}
    except DocumentConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        if path is not None:
            mark_failed(path, error)
        logger.exception("上传文档并重建知识库失败")
        raise HTTPException(status_code=500, detail=f"文档处理失败：{error}") from error
    finally:
        DOCUMENT_OPERATION_LOCK.release()
        await file.close()


def resolve_document_id(path: Path) -> str:
    from rag_project.document_processor import create_document_id
    from rag_project.config import SOURCE_DOCUMENTS_DIR
    return create_document_id(path.relative_to(SOURCE_DOCUMENTS_DIR).as_posix())


@app.post("/documents/{document_id}/reparse")
async def reparse_document(document_id: str, request: Request) -> dict:
    if not DOCUMENT_OPERATION_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="知识库正在处理其他文档，请稍后重试")
    try:
        await run_in_threadpool(validate_document, document_id)
        await run_in_threadpool(refresh_runtime_knowledge_base, request)
        return {"message": "文档已重新解析"}
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("重新解析文档失败")
        raise HTTPException(status_code=500, detail=f"重新解析失败：{error}") from error
    finally:
        DOCUMENT_OPERATION_LOCK.release()


@app.delete("/documents/{document_id}")
async def remove_document(document_id: str, request: Request) -> dict:
    if not DOCUMENT_OPERATION_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="知识库正在处理其他文档，请稍后重试")
    try:
        path = resolve_document(document_id)
        backup = path.read_bytes()
        await run_in_threadpool(delete_document, document_id)
        try:
            await run_in_threadpool(refresh_runtime_knowledge_base, request)
        except Exception:
            path.write_bytes(backup)
            await run_in_threadpool(refresh_runtime_knowledge_base, request)
            raise
        return {"message": "文档已删除，知识库已更新"}
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        logger.exception("删除文档并重建知识库失败")
        raise HTTPException(status_code=500, detail=f"删除失败：{error}") from error
    finally:
        DOCUMENT_OPERATION_LOCK.release()


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
