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