import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


print("===== Day 17：本地大模型与RAG生成 =====")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

generation_model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# GPU使用float16可以减少模型占用的显存；CPU则使用兼容性更好的float32。
generation_dtype = (
    torch.float16
    if device.type == "cuda"
    else torch.float32
)

print("当前设备：", device)
print("生成模型：", generation_model_id)
print("加载数据类型：", generation_dtype)


print("\n===== 加载Tokenizer =====")

generation_tokenizer = AutoTokenizer.from_pretrained(
    generation_model_id
)


print("\n===== 加载生成模型 =====")

generation_model = AutoModelForCausalLM.from_pretrained(
    generation_model_id,
    torch_dtype=generation_dtype,
)

generation_model = generation_model.to(device)
generation_model.eval()


parameter_count = sum(
    parameter.numel()
    for parameter in generation_model.parameters()
)

print("模型加载完成")
print("模型所在设备：", next(generation_model.parameters()).device)
print("模型参数类型：", next(generation_model.parameters()).dtype)
print("模型参数量：", f"{parameter_count / 1_000_000:.2f}M")
print("隐藏层向量维度：", generation_model.config.hidden_size)
print("Transformer层数：", generation_model.config.num_hidden_layers)
print("词表大小：", generation_model.config.vocab_size)


#Chat Template 与首次生成
print("\n===== 构造对话消息 =====")

messages = [
    {
        "role": "system",
        "content": (
            "你是一名人工智能学习助手。"
            "请使用简洁、准确的中文回答问题。"
        ),
    },
    {
        "role": "user",
        "content": "请用一句话解释什么是RAG。",
    },
]


formatted_prompt = generation_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("应用Chat Template后的提示词：")
print(formatted_prompt)


print("\n===== 将对话转换成模型输入 =====")

generation_inputs = generation_tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

generation_inputs = generation_inputs.to(device)

input_token_count = generation_inputs["input_ids"].shape[1]

print("input_ids形状：", generation_inputs["input_ids"].shape)
print("输入Token数量：", input_token_count)


print("\n===== 生成回答 =====")

with torch.inference_mode():
    generated_token_ids = generation_model.generate(
        **generation_inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=generation_tokenizer.eos_token_id,
    )


# generate()返回“原始输入Token + 新生成Token”，这里只保留新生成部分。
new_token_ids = generated_token_ids[
    0,
    input_token_count:,
]

answer = generation_tokenizer.decode(
    new_token_ids,
    skip_special_tokens=True,
)

print("完整生成结果形状：", generated_token_ids.shape)
print("新生成Token数量：", new_token_ids.shape[0])
print("模型回答：", answer)


#封装生成函数与生成参数
def generate_answer(
    user_question: str,
    do_sample: bool = False,
    max_new_tokens: int = 150,
) -> str:
    """接收用户问题，调用本地生成模型，返回回答字符串。"""

    messages = [
        {
            "role": "system",
            "content": "你是一名人工智能学习助手，请使用简洁、准确的中文回答。",
        },
        {
            "role": "user",
            "content": user_question,
        },
    ]

    model_inputs = generation_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    prompt_token_count = model_inputs["input_ids"].shape[1]

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": generation_tokenizer.eos_token_id,
    }

    # 只有开启随机采样时，temperature 和 top_p 才会生效。
    if do_sample:
        generation_kwargs["temperature"] = 0.7
        generation_kwargs["top_p"] = 0.9

    with torch.inference_mode():
        output_token_ids = generation_model.generate(
            **model_inputs,
            **generation_kwargs,
        )

    # generate()返回“原提示词+新回答”，这里切掉原提示词。
    new_token_ids = output_token_ids[0, prompt_token_count:]

    answer = generation_tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True,
    )

    return answer


question = "请用简单的语言解释RAG有什么作用。"

stable_answer = generate_answer(
    user_question=question,
    do_sample=False,
)

sampled_answer = generate_answer(
    user_question=question,
    do_sample=True,
)

print("\n===== 稳定生成结果 =====")
print(stable_answer)

print("\n===== 随机采样结果 =====")
print(sampled_answer)


#接入 Day 16 知识库，完成完整 RAG
# 加载Day 16保存的向量知识库
print("\n===== 加载Day 16向量知识库 =====")

current_directory = Path(__file__).resolve().parent

metadata_path = current_directory / "day16_knowledge_base.json"
embeddings_path = current_directory / "day16_chunk_embeddings.pth"

with open(metadata_path, "r", encoding="utf-8") as file:
    knowledge_base_metadata = json.load(file)

knowledge_chunks = knowledge_base_metadata["chunks"]

knowledge_embeddings = torch.load(
    embeddings_path,
    map_location="cpu",
    weights_only=True,
)

# 文本块列表和向量矩阵必须按照同一顺序一一对应。
if len(knowledge_chunks) != knowledge_embeddings.shape[0]:
    raise ValueError("文本块数量与向量矩阵行数不一致")

if knowledge_embeddings.shape[1] != knowledge_base_metadata["embedding_dimension"]:
    raise ValueError("加载的向量维度与知识库记录的维度不一致")

print("向量模型：", knowledge_base_metadata["model_id"])
print("文本块数量：", len(knowledge_chunks))
print("文本块向量形状：", knowledge_embeddings.shape)
print("文本块向量设备：", knowledge_embeddings.device)


# 加载Day 16使用的中文向量模型
print("\n===== 加载查询向量模型 =====")

retrieval_model_id = knowledge_base_metadata["model_id"]

retrieval_tokenizer = AutoTokenizer.from_pretrained(
    retrieval_model_id
)

retrieval_model = AutoModel.from_pretrained(
    retrieval_model_id
).to(device)

retrieval_model.eval()

print("查询向量模型：", retrieval_model_id)
print("查询模型设备：", next(retrieval_model.parameters()).device)


def encode_rag_query(query: str) -> torch.Tensor:
    """把一个用户问题转换成归一化查询向量。"""

    query_instruction = "为这个句子生成表示以用于检索相关文章："
    instructed_query = query_instruction + query

    query_inputs = retrieval_tokenizer(
        [instructed_query],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    query_inputs = {
        name: tensor.to(device)
        for name, tensor in query_inputs.items()
    }

    with torch.inference_mode():
        query_outputs = retrieval_model(**query_inputs)

    # BGE使用第一个Token的上下文向量作为句子向量。
    query_embedding = query_outputs.last_hidden_state[:, 0, :]

    query_embedding = F.normalize(
        query_embedding,
        p=2,
        dim=1,
    )

    # 文档向量保存在CPU，所以查询向量也移回CPU进行相似度计算。
    return query_embedding.cpu()


def retrieve_chunks(
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """检索与用户问题最相似的Top-K文本块。"""

    query_embedding = encode_rag_query(query)

    similarity_scores = (
        query_embedding
        @ knowledge_embeddings.T
    ).squeeze(0)

    actual_top_k = min(
        top_k,
        len(knowledge_chunks),
    )

    top_scores, top_indices = torch.topk(
        similarity_scores,
        k=actual_top_k,
    )

    retrieval_results = []

    for rank, (score_tensor, index_tensor) in enumerate(
        zip(top_scores, top_indices),
        start=1,
    ):
        index = index_tensor.item()

        retrieval_results.append({
            "rank": rank,
            "score": score_tensor.item(),
            "chunk": knowledge_chunks[index],
        })

    return retrieval_results


def build_rag_prompt(
    query: str,
    retrieval_results: list[dict],
) -> str:
    """把检索资料和用户问题组合成RAG提示词。"""

    context_parts = []

    for result in retrieval_results:
        chunk = result["chunk"]

        context_part = (
            f"[资料{result['rank']}]\n"
            f"来源标题：{chunk['title']}\n"
            f"文档编号：{chunk['document_id']}\n"
            f"文本块编号：{chunk['chunk_id']}\n"
            f"内容：{chunk['text']}"
        )

        context_parts.append(context_part)

    retrieved_context = "\n\n".join(context_parts)

    rag_prompt = f"""请完成一次严格依据资料的问答。

要求：
1. 只能根据下面提供的检索资料回答。
2. 不要使用资料之外的信息进行补充或编造。
3. 如果资料不足，请明确回答“现有资料不足”。
4. 回答时标注资料编号，例如“根据资料1”。

检索资料：
{retrieved_context}

用户问题：
{query}
"""

    return rag_prompt


print("\n===== 执行完整RAG流程 =====")

rag_query = "RAG如何减少大模型幻觉？"

retrieval_results = retrieve_chunks(
    query=rag_query,
    top_k=3,
)

print("用户问题：", rag_query)

for result in retrieval_results:
    chunk = result["chunk"]

    print(f"\n第{result['rank']}名检索结果")
    print("相似度：", f"{result['score']:.4f}")
    print("来源：", chunk["title"])
    print("文本块：", chunk["text"])


rag_prompt = build_rag_prompt(
    query=rag_query,
    retrieval_results=retrieval_results,
)

print("\n===== RAG增强提示词 =====")
print(rag_prompt)


rag_answer = generate_answer(
    user_question=rag_prompt,
    do_sample=False,
    max_new_tokens=200,
)

print("\n===== RAG最终回答 =====")
print(rag_answer)


#增加相关性门槛与完整RAG问答函数
def answer_with_rag(
    query: str,
    top_k: int = 3,
    min_similarity: float = 0.60,
) -> dict:
    """执行检索、相关性判断、提示词增强和答案生成。"""

    retrieval_results = retrieve_chunks(
        query=query,
        top_k=top_k,
    )

    if not retrieval_results:
        return {
            "query": query,
            "answer": "现有资料不足",
            "best_similarity": None,
            "passed_threshold": False,
            "sources": [],
        }

    best_similarity = retrieval_results[0]["score"]

    # Top-K一定会返回结果，因此还要判断最高分是否真正达到相关性门槛。
    if best_similarity < min_similarity:
        return {
            "query": query,
            "answer": "现有资料不足",
            "best_similarity": best_similarity,
            "passed_threshold": False,
            "sources": [],
        }

    rag_prompt = build_rag_prompt(
        query=query,
        retrieval_results=retrieval_results,
    )

    answer = generate_answer(
        user_question=rag_prompt,
        do_sample=False,
        max_new_tokens=200,
    )

    sources = []

    for result in retrieval_results:
        chunk = result["chunk"]

        sources.append({
            "rank": result["rank"],
            "score": result["score"],
            "title": chunk["title"],
            "document_id": chunk["document_id"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
        })

    return {
        "query": query,
        "answer": answer,
        "best_similarity": best_similarity,
        "passed_threshold": True,
        "sources": sources,
    }


def print_rag_result(result: dict) -> None:
    """打印结构化RAG问答结果。"""

    print("\n用户问题：", result["query"])

    if result["best_similarity"] is None:
        print("最高相似度：无")
    else:
        print(
            "最高相似度：",
            f"{result['best_similarity']:.4f}",
        )

    print("通过相关性门槛：", result["passed_threshold"])
    print("模型回答：", result["answer"])

    if result["sources"]:
        print("\n引用来源：")

        for source in result["sources"]:
            print(
                f"- 资料{source['rank']}："
                f"{source['title']}，"
                f"相似度={source['score']:.4f}，"
                f"文本块={source['chunk_id']}"
            )
    else:
        print("引用来源：无")


print("\n===== 测试相关问题 =====")

relevant_result = answer_with_rag(
    query="Transformer除了多头注意力，还包含哪些重要结构？",
    top_k=3,
    min_similarity=0.60,
)

print_rag_result(relevant_result)


print("\n===== 测试知识库之外的问题 =====")

irrelevant_result = answer_with_rag(
    query="怎样制作番茄炒蛋？",
    top_k=3,
    min_similarity=0.60,
)

print_rag_result(irrelevant_result)



#对比无RAG与RAG，并评估检索效果
# 对比无RAG回答与RAG回答
print("\n===== 无RAG与RAG回答对比 =====")

comparison_question = "Transformer除了多头注意力，还包含哪些重要结构？"

direct_answer = generate_answer(
    user_question=comparison_question,
    do_sample=False,
    max_new_tokens=150,
)

rag_comparison_result = answer_with_rag(
    query=comparison_question,
    top_k=3,
    min_similarity=0.60,
)

print("\n用户问题：", comparison_question)

print("\n----- 无RAG回答 -----")
print(direct_answer)

print("\n----- RAG回答 -----")
print(rag_comparison_result["answer"])

print("\n----- RAG引用来源 -----")

for source in rag_comparison_result["sources"]:
    print(
        f"资料{source['rank']}："
        f"{source['title']}，"
        f"相似度={source['score']:.4f}，"
        f"文本块={source['chunk_id']}"
    )


# 建立一个小型检索测试集
retrieval_test_cases = [
    {
        "query": "RAG如何减少大模型幻觉？",
        "expected_title": "RAG基础",
    },
    {
        "query": "Transformer除了多头注意力还包含哪些结构？",
        "expected_title": "Transformer基础",
    },
    {
        "query": "PyTorch模型训练循环包含哪些步骤？",
        "expected_title": "PyTorch训练",
    },
    {
        "query": "Docker为什么方便机器学习服务部署？",
        "expected_title": "Docker部署",
    },
    {
        "query": "怎样制作番茄炒蛋？",
        "expected_title": None,
    },
]


def evaluate_retrieval_system(
    test_cases: list[dict],
    top_k: int = 3,
    min_similarity: float = 0.60,
) -> float:
    """检查相关资料能否被检索，无关问题能否被门槛拒绝。"""

    passed_case_count = 0

    print("\n===== 开始评估RAG检索系统 =====")

    for case_number, case in enumerate(
        test_cases,
        start=1,
    ):
        query = case["query"]
        expected_title = case["expected_title"]

        results = retrieve_chunks(
            query=query,
            top_k=top_k,
        )

        best_similarity = results[0]["score"]

        retrieved_titles = [
            result["chunk"]["title"]
            for result in results
        ]

        passed_threshold = (
            best_similarity >= min_similarity
        )

        if expected_title is None:
            # 知识库之外的问题应该被相关性门槛拒绝。
            case_passed = not passed_threshold
        else:
            source_hit = expected_title in retrieved_titles

            case_passed = (
                source_hit
                and passed_threshold
            )

        if case_passed:
            passed_case_count += 1
            evaluation_result = "通过"
        else:
            evaluation_result = "失败"

        print(f"\n测试问题{case_number}：{query}")
        print("期望来源：", expected_title)
        print("检索来源：", retrieved_titles)
        print("最高相似度：", f"{best_similarity:.4f}")
        print("通过相关性门槛：", passed_threshold)
        print("本题评估结果：", evaluation_result)

    retrieval_accuracy = (
        passed_case_count
        / len(test_cases)
    )

    print("\n===== 检索评估结果 =====")
    print("测试问题数量：", len(test_cases))
    print("通过问题数量：", passed_case_count)
    print("检索测试准确率：", f"{retrieval_accuracy:.2%}")

    return retrieval_accuracy


retrieval_accuracy = evaluate_retrieval_system(
    test_cases=retrieval_test_cases,
    top_k=3,
    min_similarity=0.60,
)