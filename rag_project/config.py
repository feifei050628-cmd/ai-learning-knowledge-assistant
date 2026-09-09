from pathlib import Path

import torch


PROJECT_NAME = "本地中文 RAG 问答系统"

# __file__ 是当前 config.py 的路径，parent 得到 rag_project 文件夹。
PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PROJECT_DIR / "knowledge_base"
SOURCE_DOCUMENTS_DIR = PROJECT_DIR / "source_documents"
CHUNK_PREVIEW_PATH = KNOWLEDGE_BASE_DIR / "chunk_preview.json"

METADATA_PATH = (
    KNOWLEDGE_BASE_DIR
    / "day16_knowledge_base.json"
)

EMBEDDINGS_PATH = (
    KNOWLEDGE_BASE_DIR
    / "day16_chunk_embeddings.pth"
)

GENERATION_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

GENERATION_DTYPE = (
    torch.float16
    if DEVICE.type == "cuda"
    else torch.float32
)

DEFAULT_TOP_K = 3
DEFAULT_MIN_SIMILARITY = 0.40
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80

if __name__ == "__main__":          #表示只有直接运行 config.py 这个模块时，下面的检查代码才执行；其他模块导入 config.py 时，不会自动打印这些内容。
    print("项目名称：", PROJECT_NAME)
    print("项目目录：", PROJECT_DIR)
    print("知识库目录：", KNOWLEDGE_BASE_DIR)
    print("元数据文件存在：", METADATA_PATH.exists())
    print("向量文件存在：", EMBEDDINGS_PATH.exists())
    print("生成模型：", GENERATION_MODEL_ID)
    print("运行设备：", DEVICE)
    print("模型数据类型：", GENERATION_DTYPE)


RETRIEVAL_QUERY_INSTRUCTION = (
    "为这个句子生成表示以用于检索相关文章："
)

RETRIEVAL_MAX_LENGTH = 512