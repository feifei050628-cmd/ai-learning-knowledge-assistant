import torch
import transformers

print("===== Day 15 环境检查 =====")
print("PyTorch版本：", torch.__version__)
print("Transformers版本：", transformers.__version__)
print("CUDA是否可用：", torch.cuda.is_available())

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("本次使用设备：", device)


#使用预训练 Tokenizer
from transformers import AutoTokenizer

print("\n===== 加载预训练Tokenizer =====")

model_name = "sentence-transformers/all-MiniLM-L6-v2"

# 根据模型名称下载并加载与模型配套的Tokenizer。
tokenizer = AutoTokenizer.from_pretrained(model_name)

texts = [
    "Python PyTorch Transformer",
    "Machine learning engineer",
    "Sales marketing customer",
]

encoded_inputs = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=12,
    return_tensors="pt",
)

print("Tokenizer类型：", type(tokenizer))
print("编码结果包含：", encoded_inputs.keys())

print("\ninput_ids：")
print(encoded_inputs["input_ids"])
print("input_ids形状：", encoded_inputs["input_ids"].shape)

print("\nattention_mask：")
print(encoded_inputs["attention_mask"])
print("attention_mask形状：", encoded_inputs["attention_mask"].shape)

first_tokens = tokenizer.convert_ids_to_tokens(
    encoded_inputs["input_ids"][0]
)

print("\n第一句话：", texts[0])
print("第一句话的Token：", first_tokens)


#加载预训练 Transformer
from transformers import AutoModel

print("\n===== 加载预训练Transformer =====")

# 下载并加载与Tokenizer配套的预训练模型。
model = AutoModel.from_pretrained(model_name)
model = model.to(device)
model.eval()

# Tokenizer目前生成的是CPU Tensor，需要移动到模型所在设备。
model_inputs = {
    name: tensor.to(device)
    for name, tensor in encoded_inputs.items()
}

with torch.no_grad():
    # **会把字典拆成input_ids、attention_mask等关键字参数。
    model_outputs = model(**model_inputs)

last_hidden_state = model_outputs.last_hidden_state

print("模型类型：", type(model))
print("模型隐藏层维度：", model.config.hidden_size)
print("模型输出类型：", type(model_outputs))

print("\nlast_hidden_state形状：", last_hidden_state.shape)
print("last_hidden_state设备：", last_hidden_state.device)

print("\n第一句话第一个Token的向量：")
print(last_hidden_state[0, 0])

print(
    "第一句话第一个Token的向量形状：",
    last_hidden_state[0, 0].shape,
)


#Masked Mean Pooling
import torch.nn.functional as F


def mean_pooling(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    # 在末尾增加向量维度，使Mask能够和Token向量逐元素相乘。
    expanded_mask = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
   
    # PAD位置的Mask是0，相乘后对应Token向量会被清零。
    valid_embedding_sum = torch.sum(
        token_embeddings * expanded_mask,
        dim=1,
    )

    # 统计每句话中参与平均计算的有效Token数量。
    valid_token_count = expanded_mask.sum(dim=1).clamp(min=1e-9)

    return valid_embedding_sum / valid_token_count


print("\n===== 生成句子向量 =====")

sentence_embeddings = mean_pooling(
    last_hidden_state,
    model_inputs["attention_mask"],
)

print("池化前形状：", last_hidden_state.shape)
print("池化后形状：", sentence_embeddings.shape)

print("\n归一化前的句子向量模长：")
print(torch.linalg.vector_norm(sentence_embeddings, dim=1))

# 将每个句子向量的模长归一化为1。
sentence_embeddings = F.normalize(
    sentence_embeddings,
    p=2,
    dim=1,
)

print("\n归一化后的句子向量模长：")
print(torch.linalg.vector_norm(sentence_embeddings, dim=1))

print("\n第一句话的句子向量：")
print(sentence_embeddings[0])


#预训练模型岗位匹配排序
def encode_sentences(sentences: list[str]) -> torch.Tensor:
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    inputs = {
        name: tensor.to(device)
        for name, tensor in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = mean_pooling(
        outputs.last_hidden_state,
        inputs["attention_mask"],
    )

    # 归一化后，两个向量的点积就等于余弦相似度。
    return F.normalize(embeddings, p=2, dim=1)


print("\n===== 预训练模型岗位匹配 =====")

my_profile = (
    "Python PyTorch transformer machine learning "
    "natural language processing"
)

job_descriptions = [
    (
        "Machine learning engineer requiring Python, "
        "PyTorch and transformer experience"
    ),
    (
        "Frontend developer requiring JavaScript, "
        "React, HTML and CSS"
    ),
    (
        "Data analyst requiring SQL, Excel and "
        "data visualization"
    ),
    (
        "AI engineer building natural language processing, "
        "RAG and large language model applications"
    ),
]

profile_embedding = encode_sentences([my_profile])
job_embeddings = encode_sentences(job_descriptions)

print("个人技能向量形状：", profile_embedding.shape)
print("岗位向量形状：", job_embeddings.shape)

# (1, 384) @ (384, 4) → (1, 4)
similarities = profile_embedding @ job_embeddings.T
similarities = similarities.squeeze(0)

print("相似度形状：", similarities.shape)
print("所有相似度：", similarities)

ranked_indices = torch.argsort(
    similarities,
    descending=True,
)

print("\n岗位匹配排序：")

for rank, index_tensor in enumerate(ranked_indices, start=1):
    index = index_tensor.item()
    score = similarities[index].item()

    print(f"\n第{rank}名")
    print("岗位描述：", job_descriptions[index])
    print(f"余弦相似度：{score:.4f}")


#语义相似度与必备技能检查
print("\n===== 语义相似度与必备技能检查 =====")

my_skills = {
    "python",
    "pytorch",
    "transformer",
    "nlp",
}

jobs = [
    {
        "title": "机器学习工程师",
        "description": job_descriptions[0],
        "required_skills": {
            "python",
            "pytorch",
            "transformer",
        },
    },
    {
        "title": "前端开发工程师",
        "description": job_descriptions[1],
        "required_skills": {
            "javascript",
            "react",
            "css",
        },
    },
    {
        "title": "数据分析师",
        "description": job_descriptions[2],
        "required_skills": {
            "sql",
            "excel",
            "visualization",
        },
    },
    {
        "title": "大模型应用工程师",
        "description": job_descriptions[3],
        "required_skills": {
            "python",
            "nlp",
            "rag",
            "llm",
        },
    },
]

results = []

for index, job in enumerate(jobs):
    required_skills = job["required_skills"]

    matched_skills = my_skills & required_skills
    missing_skills = required_skills - my_skills

    if required_skills:
        coverage = len(matched_skills) / len(required_skills)
    else:
        coverage = 1.0

    results.append(
        {
            "title": job["title"],
            "semantic_score": similarities[index].item(),
            "coverage": coverage,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
        }
    )

# 优先按照必备技能覆盖率排序，覆盖率相同时再比较语义相似度。
results.sort(
    key=lambda result: (
        result["coverage"],
        result["semantic_score"],
    ),
    reverse=True,
)

for rank, result in enumerate(results, start=1):
    print(f"\n第{rank}名：{result['title']}")
    print(f"语义相似度：{result['semantic_score']:.4f}")
    print(f"必备技能覆盖率：{result['coverage']:.2%}")
    print("已掌握技能：", result["matched_skills"])
    print("缺失技能：", result["missing_skills"])