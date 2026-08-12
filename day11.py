import torch
import torch.nn as nn


print("===== Day 11：Self-Attention =====")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)

tokens = [
    "python",
    "pytorch",
    "rag",
]

# 1个句子、3个Token，每个Token由4个数表示。
input_vectors = torch.tensor(
    [
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    ],
    dtype=torch.float32,
    device=device,
)

print("Token列表：", tokens)
print("输入向量：")
print(input_vectors)
print("输入形状：", input_vectors.shape)


embedding_dim = 4
query_dim = 4

# 三个Linear层会分别学习不同的转换参数。
query_layer = nn.Linear(
    embedding_dim,
    query_dim,
    bias=False,
).to(device)

key_layer = nn.Linear(
    embedding_dim,
    query_dim,
    bias=False,
).to(device)

value_layer = nn.Linear(
    embedding_dim,
    query_dim,
    bias=False,
).to(device)


query = query_layer(input_vectors)
key = key_layer(input_vectors)
value = value_layer(input_vectors)

print("\n===== Q、K、V =====")
print("Query：")
print(query)

print("\nKey：")
print(key)

print("\nValue：")
print(value)

print("\nQuery形状：", query.shape)
print("Key形状：", key.shape)
print("Value形状：", value.shape)


#计算注意力分数：现在我们要让每个 Token 的 Query 与所有 Token 的 Key 进行比较。
print("\n===== 计算注意力分数 =====")

# 交换Key最后两个维度，方便Query与所有Key进行批量点积。
key_transposed = key.transpose(-2, -1)

print("转置前的Key形状：", key.shape)
print("转置后的Key形状：", key_transposed.shape)

# 每个Token的Query分别与所有Token的Key计算点积。
raw_scores = query @ key_transposed

print("\n缩放前的注意力分数：")
print(raw_scores)
print("注意力分数形状：", raw_scores.shape)


# d_k是Query和Key的向量维度。
d_k = query.shape[-1]

# 除以d_k的平方根，避免点积过大。
scaled_scores = raw_scores / (d_k ** 0.5)

print("\nd_k：", d_k)
print("d_k的平方根：", d_k ** 0.5)

print("\n缩放后的注意力分数：")
print(scaled_scores)
print("缩放后的形状：", scaled_scores.shape)


print("\n===== 逐行理解注意力分数 =====")

for query_index, token in enumerate(tokens):
    print(
        f"{token}的Query对所有Token的分数：",
        scaled_scores[0, query_index],
    )


#Softmax 与 Value 加权求和
print("\n===== Softmax注意力权重 =====")

attention_weights = torch.softmax(
    scaled_scores,
    dim=-1,
)

print("注意力权重：")
print(attention_weights)

print("注意力权重形状：", attention_weights.shape)

# 检查每个Query对应的一行权重之和。
row_sums = attention_weights.sum(dim=-1)
print("每行权重之和：")
print(row_sums)


print("\n===== Value加权求和 =====")

context_vectors = torch.matmul(
    attention_weights,
    value,
)

print("上下文向量：")
print(context_vectors)

print("上下文向量形状：", context_vectors.shape)


#使用 Mask 屏蔽 Padding
print("\n===== Padding Mask实验 =====")

padded_tokens = [
    "python",
    "pytorch",
    "rag",
    "<PAD>",
]

# 在原来的3个Token后面增加一个全0的PAD向量。
pad_vector = torch.zeros(
    1,
    1,
    embedding_dim,
    dtype=torch.float32,
    device=device,
)

padded_input_vectors = torch.cat(
    [input_vectors, pad_vector],
    dim=1,
)

attention_mask = torch.tensor(
    [[1, 1, 1, 0]],
    dtype=torch.bool,
    device=device,
)

padded_query = query_layer(padded_input_vectors)
padded_key = key_layer(padded_input_vectors)
padded_value = value_layer(padded_input_vectors)

padded_scores = torch.matmul(
    padded_query,
    padded_key.transpose(-2, -1),
)

padded_scaled_scores = padded_scores / (
    padded_query.shape[-1] ** 0.5
)

print("未屏蔽的缩放分数：")
print(padded_scaled_scores)

unmasked_weights = torch.softmax(
    padded_scaled_scores,
    dim=-1,
)

print("\n未屏蔽PAD的注意力权重：")
print(unmasked_weights)


# (1,4)增加维度后变成(1,1,4)，从而对所有Query行屏蔽同一个PAD列。
key_mask = attention_mask.unsqueeze(1)

masked_scores = padded_scaled_scores.masked_fill(
    key_mask == 0,
    float("-inf"),
)

print("\n屏蔽PAD后的注意力分数：")
print(masked_scores)

masked_weights = torch.softmax(
    masked_scores,
    dim=-1,
)

print("\n屏蔽PAD后的注意力权重：")
print(masked_weights)

print("\nPAD列的权重：")
print(masked_weights[:, :, -1])

print("\n每一行权重之和：")
print(masked_weights.sum(dim=-1))


masked_context_vectors = torch.matmul(
    masked_weights,
    padded_value,
)

print("\n加入Mask后的上下文向量：")
print(masked_context_vectors)

print("上下文向量形状：", masked_context_vectors.shape)


#封装完整的单头注意力函数
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: #表示这个函数返回两个张量
    """计算缩放点积注意力。"""

    d_k = query.shape[-1]

    scores = torch.matmul(
        query,
        key.transpose(-2, -1),
    )
    scaled_scores = scores / (d_k ** 0.5)

    if attention_mask is not None:
        # (batch, tokens)变成(batch, 1, tokens)，屏蔽PAD对应的Key列。
        key_mask = attention_mask.unsqueeze(1).bool()

        scaled_scores = scaled_scores.masked_fill(
            ~key_mask,
            float("-inf"),
        )

    attention_weights = torch.softmax(
        scaled_scores,
        dim=-1,
    )

    context_vectors = torch.matmul(
        attention_weights,
        value,
    )

    if attention_mask is not None:
        # PAD作为Query的输出也不需要保留，因此将PAD行的上下文向量清零。
        query_mask = attention_mask.unsqueeze(-1).to(
            context_vectors.dtype
        )
        context_vectors = context_vectors * query_mask

    return context_vectors, attention_weights

#调用函数
print("\n===== 调用完整注意力函数 =====")

final_context, final_weights = scaled_dot_product_attention(
    padded_query,
    padded_key,
    padded_value,
    attention_mask,
)

print("最终注意力权重：")
print(final_weights)

print("\n最终上下文向量：")
print(final_context)

print("\n注意力权重形状：", final_weights.shape)
print("上下文向量形状：", final_context.shape)

#逐个查看 Token 的注意力
print("\n===== 逐个查看注意力 =====")

for query_index, query_token in enumerate(padded_tokens):
    print(f"\n{query_token}作为Query：")

    for key_index, key_token in enumerate(padded_tokens):
        weight = final_weights[
            0,
            query_index,
            key_index,
        ].item()

        print(
            f"  关注{key_token}: "
            f"{weight:.2%}"
        )