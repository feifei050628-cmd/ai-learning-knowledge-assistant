# AI 技术学习知识库与智能问答系统

这是一个面向中文学习资料的本地 RAG 项目。它把 Markdown 文档和 Day 1–29 学习笔记处理为可检索知识库，使用 BGE 完成中文语义检索，再由 Qwen 根据召回资料生成带来源的回答。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![PyTorch](https://img.shields.io/badge/PyTorch-Embeddings-EE4C2C)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC)

## 项目亮点

- 完整 RAG 链路：文档导入、智能切块、向量化、Top-K 检索、相关性门控和生成。
- 可追溯回答：接口和 Web 页同时展示来源、文本块编号与相似度。
- 质量评估：包含检索评估、门槛校准、生成质量评估和拒答测试。
- 增量更新：启动 API 时检查学习笔记变化，必要时重建知识库。
- 可交互界面：提供响应式中文问答页、服务状态、参数调节与错误恢复。
- 工程化保障：FastAPI/Pydantic 数据校验、pytest 测试与 GitHub Actions。

## 技术架构

```text
原始文档 / 学习笔记
        ↓
文档清洗与切块 → BGE 向量化 → 本地知识库
                                      ↑
用户问题 → 语义检索 → 相关性门控 → Qwen 生成
                                      ↓
                           回答 + 分数 + 引用来源
```

## 快速开始

Python 3.10–3.12 均可，建议使用全新虚拟环境。首次启动需下载 Hugging Face 模型。

```powershell
git clone <your-repository-url>
cd "python-learning"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\rag_project\requirements-dev.txt
```

向量文件不纳入 Git。克隆后先构建知识库：

```powershell
python -m rag_project.note_importer
python -m rag_project.knowledge_base_builder
```

启动 Web 与 API：

```powershell
python -m uvicorn rag_project.api:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000/chat`；OpenAPI 文档位于 `http://127.0.0.1:8000/docs`。

## 测试与评估

```powershell
# 离线单元测试（不下载模型）
python -m pytest .\rag_project\tests -q

# 真实检索评估与门槛校准
python -m rag_project.evaluate_retrieval
python -m rag_project.calibrate_threshold

# 完整生成评估（会加载生成模型）
python -m rag_project.evaluate_generation
```

## 仓库导览

- `rag_project/api.py`：FastAPI 入口与模型生命周期。
- `rag_project/rag_pipeline.py`：RAG 主流程。
- `rag_project/retrieval.py` / `generation.py`：检索与生成。
- `rag_project/document_processor.py` / `note_importer.py`：资料导入与切块。
- `rag_project/knowledge_base_builder.py`：向量知识库构建。
- `rag_project/web/`：无前端框架的聊天工作台。
- `rag_project/tests/`：模式、检索、构建、API 与更新测试。
- `day01笔记.txt`–`day29笔记.txt`：项目学习轨迹与知识源。

更完整的模块、数据处理和评估说明见 [`rag_project/README.md`](rag_project/README.md)。

## 已知边界

- 默认生成模型较小，结果仍需人工核验。
- 当前评估集规模有限，指标只作为迭代基线。
- 本项目为本地单用户学习系统，未实现登录、多租户和云端历史同步。
