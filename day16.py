import json
from pathlib import Path
print("===== Day 16：RAG中文语义检索 =====")

documents = [
    {
        "id": "doc_001",
        "title": "RAG基础",
        "content": (
            "RAG是检索增强生成技术。"
            "它会先从外部知识库中找到与用户问题相关的资料，"
            "再把检索结果加入提示词，最后交给大模型生成回答。"
            "RAG可以补充模型没有见过的企业内部知识，"
            "也能在一定程度上降低大模型幻觉。"
        ),
    },
    {
        "id": "doc_002",
        "title": "Transformer基础",
        "content": (
            "Transformer通过自注意力机制处理Token之间的关系。"
            "多头注意力可以从多个角度学习不同Token之间的联系。"
            "Transformer还包含残差连接、LayerNorm和前馈神经网络。"
        ),
    },
    {
        "id": "doc_003",
        "title": "PyTorch训练",
        "content": (
            "PyTorch训练循环通常包括前向传播、计算损失、"
            "清空梯度、反向传播和更新参数。"
            "预测时应该调用model.eval并关闭梯度计算。"
        ),
    },
    {
        "id": "doc_004",
        "title": "Docker部署",
        "content": (
            "Docker可以把应用代码、依赖和运行环境打包成镜像。"
            "使用容器可以减少不同电脑之间的环境差异，"
            "方便机器学习服务的部署和交付。"
        ),
    },
]


def split_text(
    text: str,
    chunk_size: int = 45,
    overlap: int = 10,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap必须满足0 <= overlap < chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # 已经到达文本末尾，必须结束循环。
        if end == len(text):
            break

        # 保留一部分上一块结尾，减少边界信息丢失。
        start = end - overlap

    return chunks


document_chunks = []

for document in documents:
    chunks = split_text(document["content"])

    for chunk_index, chunk_text in enumerate(chunks, start=1):
        document_chunks.append(
            {
                "chunk_id": (
                    f"{document['id']}_chunk_{chunk_index}"
                ),
                "document_id": document["id"],
                "title": document["title"],
                "text": chunk_text,
            }
        )


print("原始文档数量：", len(documents))
print("切块后的数量：", len(document_chunks))

print("\n===== 查看所有文本块 =====")

for chunk in document_chunks:
    print("\n文本块编号：", chunk["chunk_id"])
    print("来源标题：", chunk["title"])
    print("文本内容：", chunk["text"])
    print("字符数量：", len(chunk["text"]))



#加载中文向量模型
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


print("\n===== 加载中文向量模型 =====")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model_id = "BAAI/bge-small-zh-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_id)
embedding_model = AutoModel.from_pretrained(model_id).to(device)
embedding_model.eval()

print("当前设备：", device)


sample_texts = [
    "RAG可以从外部知识库中检索相关资料。",
    "Docker可以帮助开发者部署人工智能应用。",
]

model_inputs = tokenizer(
    sample_texts,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt",
)

# 字典中的input_ids、attention_mask等Tensor都必须和模型位于相同设备。
model_inputs = {
    name: tensor.to(device)
    for name, tensor in model_inputs.items()
}

with torch.no_grad():
    model_outputs = embedding_model(**model_inputs)

last_hidden_state = model_outputs.last_hidden_state

# BGE官方用法：取每句话第一个Token，也就是CLS位置的上下文向量。
sentence_embeddings = last_hidden_state[:, 0, :]

# 将每个句子向量的模长归一化为1，之后点积就等于余弦相似度。
sentence_embeddings = F.normalize(
    sentence_embeddings,
    p=2,
    dim=1,
)

print("input_ids形状：", model_inputs["input_ids"].shape)
print("Token上下文向量形状：", last_hidden_state.shape)
print("句子向量形状：", sentence_embeddings.shape)
print("句子向量模长：", torch.linalg.vector_norm(
    sentence_embeddings,
    dim=1,
))


#为所有文本块生成向量
def encode_passages(
    texts: list[str],
    batch_size: int = 8,
) -> torch.Tensor:
    """分批生成文档文本的归一化句子向量。"""

    if not texts:
        return torch.empty((0, 512))

    embedding_batches = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        batch_inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        batch_inputs = {
            name: tensor.to(device)
            for name, tensor in batch_inputs.items()
        }

        with torch.no_grad():
            batch_outputs = embedding_model(**batch_inputs)

        batch_embeddings = batch_outputs.last_hidden_state[:, 0, :]
        batch_embeddings = F.normalize(
            batch_embeddings,
            p=2,
            dim=1,
        )

        # GPU负责计算，完成后的向量移回CPU长期保存，减少显存占用。
        embedding_batches.append(batch_embeddings.cpu())

    return torch.cat(embedding_batches, dim=0)


print("\n===== 构建文本块向量索引 =====")

chunk_texts = [
    chunk["text"]
    for chunk in document_chunks
]

chunk_embeddings = encode_passages(
    chunk_texts,
    batch_size=4,
)

print("文本块数量：", len(document_chunks))
print("文本块向量形状：", chunk_embeddings.shape)
print("文本块向量设备：", chunk_embeddings.device)

for index, chunk in enumerate(document_chunks):
    print(
        f"索引={index}，"
        f"标题={chunk['title']}，"
        f"文本={chunk['text']}"
    )


#查询编码与 Top-K 检索
def encode_queries(queries: list[str]) -> torch.Tensor:
    """为检索问题生成归一化查询向量。"""

    query_instruction = "为这个句子生成表示以用于检索相关文章："

    instructed_queries = [
        query_instruction + query
        for query in queries
    ]

    query_inputs = tokenizer(
        instructed_queries,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    query_inputs = {
        name: tensor.to(device)
        for name, tensor in query_inputs.items()
    }

    with torch.no_grad():
        query_outputs = embedding_model(**query_inputs)

    query_embeddings = query_outputs.last_hidden_state[:, 0, :]
    query_embeddings = F.normalize(
        query_embeddings,
        p=2,
        dim=1,
    )

    return query_embeddings.cpu()


print("\n===== Top-K语义检索 =====")

query = "如何通过外部知识减少大模型回答中的幻觉？"

query_embedding = encode_queries([query])

similarity_scores = (
    query_embedding
    @ chunk_embeddings.T
).squeeze(0)

top_k = min(3, len(document_chunks))

top_scores, top_indices = torch.topk(
    similarity_scores,
    k=top_k,
)

print("用户问题：", query)
print("查询向量形状：", query_embedding.shape)
print("文档向量形状：", chunk_embeddings.shape)
print("全部相似度形状：", similarity_scores.shape)

for rank, (score_tensor, index_tensor) in enumerate(
    zip(top_scores, top_indices),
    start=1,
):
    score = score_tensor.item()
    index = index_tensor.item()
    chunk = document_chunks[index]

    print(f"\n第{rank}名")
    print("相似度：", round(score, 4))
    print("来源标题：", chunk["title"])
    print("文档编号：", chunk["document_id"])
    print("文本块编号：", chunk["chunk_id"])
    print("文本内容：", chunk["text"])


#把检索结果加入提示词
print("\n===== 构造RAG上下文 =====")

retrieved_chunks = []

for rank, (score_tensor, index_tensor) in enumerate(
    zip(top_scores, top_indices),
    start=1,
):
    index = index_tensor.item()

    retrieved_chunks.append({
        "rank": rank,
        "score": score_tensor.item(),
        "chunk": document_chunks[index],
    })


context_parts = []

for result in retrieved_chunks:
    chunk = result["chunk"]

    # 保留资料编号和来源，方便大模型引用，也方便检查回答依据。
    context_part = (
        f"[资料{result['rank']}]\n"
        f"来源：{chunk['title']}\n"
        f"内容：{chunk['text']}"
    )

    context_parts.append(context_part)

retrieved_context = "\n\n".join(context_parts)


rag_prompt = f"""你是一名严谨的人工智能学习助手。

请仅根据下面提供的资料回答用户问题。
如果资料不足以回答，请明确说明“现有资料不足”，不要编造信息。
回答时请标注使用了哪一条资料，例如“根据资料1”。

检索资料：
{retrieved_context}

用户问题：
{query}

回答：
"""


print("检索上下文：")
print(retrieved_context)

print("\n===== 最终RAG提示词 =====")
print(rag_prompt)



#保存和重新加载向量知识库
print("\n===== 保存本地向量知识库 =====")

current_directory = Path(__file__).resolve().parent

metadata_path = current_directory / "day16_knowledge_base.json"
embeddings_path = current_directory / "day16_chunk_embeddings.pth"


knowledge_base_metadata = {
    "model_id": model_id,
    "embedding_dimension": chunk_embeddings.shape[1],
    "chunk_count": len(document_chunks),
    "chunks": document_chunks,
}

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(
        knowledge_base_metadata,
        file,
        ensure_ascii=False,
        indent=4,
    )

torch.save(
    chunk_embeddings,
    embeddings_path,
)

print("文本块信息保存位置：", metadata_path)
print("文本块向量保存位置：", embeddings_path)


print("\n===== 重新加载向量知识库 =====")

with open(metadata_path, "r", encoding="utf-8") as file:
    loaded_metadata = json.load(file)

loaded_chunks = loaded_metadata["chunks"]

# weights_only=True限制torch.load只加载Tensor等安全数据类型。
loaded_embeddings = torch.load(
    embeddings_path,
    map_location="cpu",
    weights_only=True,
)

print("加载的模型名称：", loaded_metadata["model_id"])
print("加载的文本块数量：", len(loaded_chunks))
print("加载的向量形状：", loaded_embeddings.shape)
print("加载的向量设备：", loaded_embeddings.device)


if len(loaded_chunks) == loaded_embeddings.shape[0]:
    print("检查通过：文本块数量与向量行数一致")
else:
    print("检查失败：文本块和向量无法正确对应")