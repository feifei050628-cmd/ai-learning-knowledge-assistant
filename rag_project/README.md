# 模块化中文 RAG 问答系统

## 项目简介

这是一个使用 Python、PyTorch、Transformers、BGE 向量模型和 Qwen 生成模型实现的中文 RAG 问答项目。

项目能够从本地知识库中检索与用户问题相关的文本块，将检索资料加入提示词，再调用生成模型生成回答。

## 主要功能

- 加载本地 JSON 知识库和 PTH 向量矩阵
- 使用 BGE 模型生成用户问题向量
- 计算查询向量与知识库向量的余弦相似度
- 使用 Top-K 找出最相关的文本块
- 使用相关性门槛拒绝无关问题
- 将检索资料、来源和问题组合成 RAG 提示词
- 使用 Qwen 模型生成中文回答
- 返回回答、检索分数和引用来源
- 支持在命令行中连续提问

## 项目结构

rag_project/
├── __init__.py
├── config.py
├── model_manager.py
├── retrieval.py
├── generation.py
├── rag_pipeline.py
├── schemas.py
├── api.py
├── main.py
├── requirements.txt
├── README.md
└── knowledge_base/
    ├── day16_knowledge_base.json
    └── day16_chunk_embeddings.pth

## 模块说明

### config.py

集中保存项目名称、模型名称、文件路径、设备、Top-K、相关性门槛和最大生成长度等配置。

### model_manager.py

负责加载检索模型、生成模型以及它们配套的 Tokenizer。

### retrieval.py

负责加载知识库、生成查询向量、计算相似度，并返回最相关的文本块。

### generation.py

负责创建 Chat Template，并使用生成模型生成自然语言回答。

### rag_pipeline.py

负责连接 Retrieval、Augmentation 和 Generation，形成完整的 RAG 流程。

### schemas.py

使用 Pydantic 定义 API 请求和响应的数据结构，负责类型检查、参数限制和问题文本清洗。

### api.py

使用 FastAPI 提供健康检查、参数验证和 RAG 问答接口。服务器启动时加载一次 RAGPipeline，并让后续请求复用同一套模型和知识库。

### main.py

项目的命令行入口，负责接收用户输入、调用 RAGPipeline 并展示回答和引用来源。

## 使用的模型

检索模型：

BAAI/bge-small-zh-v1.5

生成模型：

Qwen/Qwen2.5-0.5B-Instruct

## 运行方法

首先进入 python learning 文件夹：

powershell

- cd "C:\Users\21855\Documents\python learning"

- .\.venv\Scripts\Activate.ps1

- python -m rag_project.main


输入以下内容可以退出程序：
exit
quit
退出

## RAG 工作流程
- 接收用户问题
- 为问题添加 BGE 查询指令
- 使用 BGE 生成查询向量
- 对查询向量进行归一化
- 与知识库向量矩阵计算相似度
- 使用 Top-K 找出相关文本块
- 检查最高相似度是否达到门槛
- 将资料、来源和问题组合成增强提示词
- 使用 Qwen 生成回答
- 返回回答、分数和引用来源


## 当前局限
- 知识库规模较小
- 文本切块方式比较简单
- 相关性门槛尚未经过大规模测试和校准
- 生成模型规模较小
- 当前系统主要用于学习和演示
- 回答结果仍需要人工检查


## API 运行方法

进入项目父目录并激活虚拟环境：

```powershell
cd "C:\Users\21855\Documents\python learning"
.\.venv\Scripts\Activate.ps1

python -m uvicorn rag_project.api:app --reload --host 127.0.0.1 --port 8000
```

## 运行自动化测试

安装开发依赖：

```powershell
python -m pip install -r .\rag_project\requirements-dev.txt
```
运行全部测试：
```
python -m pytest .\rag_project\tests -v
```
运行测试并查看覆盖率：
```
python -m pytest .\rag_project\tests -v --cov=rag_project.schemas --cov=rag_project.api --cov-report=term-missing
```


## 运行真实检索评估

执行：

```text
python -m rag_project.evaluate_retrieval
```

## 校准相关性门槛

先运行真实检索评估：

```powershell
python -m rag_project.evaluate_retrieval
```

再使用评估报告校准相关性门槛：

```powershell
python -m rag_project.calibrate_threshold
```

校准脚本会比较 0.20～0.70 之间的候选门槛，并计算：

- 混淆矩阵
- Precision
- Recall
- Specificity
- F1
- Balanced Accuracy

结果保存在：

```text
rag_project/threshold_calibration_report.json
```

当前评估集规模较小，校准结果只作为基线，不应直接代表生产环境效果。


## 运行完整 RAG 生成质量评估

生成质量评估会加载真实检索模型和生成模型：

```powershell
python -m rag_project.evaluate_generation
```

评估内容包括：

- 相关性门槛判断
- 正确来源命中
- 关键词覆盖
- 引用出现和引用编号合法性
- 无关问题拒答

报告保存在：

```text
rag_project/generation_evaluation_report.json
```

关键词匹配和引用检查属于规则型代理指标，不能完全代替人工评审或语义评估。


## 文档导入与智能切块

原始知识文档保存在：

```text
rag_project/source_documents/
```

支持的文档格式：

- `.txt`
- `.md`

运行文档处理程序：

```powershell
python -m rag_project.document_processor
```

程序会完成：

1. 递归读取原始文档。
2. 统一换行并清理多余空行。
3. 优先按照段落和完整句子切块。
4. 尽量保留完整的 Markdown 代码块。
5. 为文档和文本块生成稳定编号。
6. 把结果保存到 `knowledge_base/chunk_preview.json`。

当前切块参数：

- 最大文本块长度：500字符
- 相邻文本块重叠：80字符

`chunk_preview.json` 只用于检查切块质量，目前不包含向量，也不会替换正式检索知识库。


## 扩展知识库：导入学习笔记

除了 `source_documents/` 中的基础资料，系统还可以导入项目根目录下符合以下规则的学习笔记：

```text
day[0-9][0-9]笔记.txt
```


运行学习笔记导入程序：

    python -m rag_project.note_importer

程序会把基础资料与学习笔记合并，并生成：

    rag_project/knowledge_base/expanded_chunk_preview.json

该文件是切块预览，不包含向量。新增学习笔记后，需要重新运行导入程序。

当前扩展知识库包含：

- 基础资料：4篇
- 学习笔记：25篇
- 知识源总数：29
- 总字符数：338100
- 文本块总数：810


## 构建正式向量知识库

运行：

    python -m rag_project.knowledge_base_builder

程序会：

1. 加载并验证扩展切块预览。
2. 检查字段、字符数和 chunk_id。
3. 使用 BAAI/bge-small-zh-v1.5 分批编码文本。
4. 对文本向量进行 L2 归一化。
5. 保存正式元数据和向量矩阵。

生成文件：

    rag_project/knowledge_base/day26_knowledge_base.json
    rag_project/knowledge_base/day26_chunk_embeddings.pth

当前正式知识库规模：

- 知识源：29
- 文本块：810
- 向量矩阵形状：(810, 512)
- 元数据文件约：927 KB
- 向量文件约：1.66 MB

`.pth` 向量文件被 Git 忽略，不会上传 GitHub。克隆项目后，可以按照上述命令重新生成向量。

当前 `config.py` 已将正式检索路径切换到 Day 26 知识库。扩充知识库后，应重新执行检索评估并校准相关性门槛。


## 扩展知识库检索评估与门槛校准

扩展知识库建立后，需要重新评估检索质量，不能直接沿用小知识库上的指标和门槛。

运行检索评估：

```powershell
python -m rag_project.evaluate_retrieval
```

## 扩展知识库检索与生成质量评估

当前系统使用扩展知识库执行中文语义检索：

- 检索模型：`BAAI/bge-small-zh-v1.5`
- 默认 Top-K：5
- 默认相关性门槛：0.48
- 知识源数量：29
- 文本块数量：810
- 向量维度：512

运行生成质量评估：

```powershell
python -m rag_project.evaluate_generation
```