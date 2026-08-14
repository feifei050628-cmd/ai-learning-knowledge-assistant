import torch
import torch.nn as nn


print("===== Day 12：Multi-Head Attention =====")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)


batch_size = 1
token_count = 3
embed_dim = 8
num_heads = 2

assert embed_dim % num_heads == 0

head_dim = embed_dim // num_heads

print("当前设备：", device)
print("完整向量维度：", embed_dim)
print("注意力头数量：", num_heads)
print("每个头的向量维度：", head_dim)


# 1个句子、3个Token，每个Token由8个数表示。
input_vectors = torch.tensor(
    [
        [
            [1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 0.0],
            [0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5],
        ]
    ],
    dtype=torch.float32,
    device=device,
)

print("\n输入向量：")
print(input_vectors)
print("输入形状：", input_vectors.shape)


query_layer = nn.Linear(
    embed_dim,
    embed_dim,
    bias=False,
).to(device)

key_layer = nn.Linear(
    embed_dim,
    embed_dim,
    bias=False,
).to(device)

value_layer = nn.Linear(
    embed_dim,
    embed_dim,
    bias=False,
).to(device)


query = query_layer(input_vectors)
key = key_layer(input_vectors)
value = value_layer(input_vectors)

print("\n===== 拆分前的Q、K、V =====")
print("Q形状：", query.shape)
print("K形状：", key.shape)
print("V形状：", value.shape)


# (batch, tokens, embed_dim)
# → (batch, tokens, num_heads, head_dim)
query_reshaped = query.reshape(
    batch_size,
    token_count,
    num_heads,
    head_dim,
)

print("\nQ经过reshape后的形状：")
print(query_reshaped.shape)


# 把头维度移到Token维度前，方便每个头独立计算注意力。
# (batch, tokens, heads, head_dim)
# → (batch, heads, tokens, head_dim)
query_heads = query_reshaped.transpose(1, 2)

key_heads = key.reshape(
    batch_size,
    token_count,
    num_heads,
    head_dim,
).transpose(1, 2)

value_heads = value.reshape(
    batch_size,
    token_count,
    num_heads,
    head_dim,
).transpose(1, 2)

print("\n===== 拆分后的Q、K、V =====")
print("Q多头形状：", query_heads.shape)
print("K多头形状：", key_heads.shape)
print("V多头形状：", value_heads.shape)

print("\n第1个头的Query：")
print(query_heads[0, 0])

print("\n第2个头的Query：")
print(query_heads[0, 1,])


#print("\n===== 每个头独立计算注意力 =====")

# (batch, heads, tokens, head_dim)
# @
# (batch, heads, head_dim, tokens)
# →
# (batch, heads, query_count, key_count)
raw_scores = torch.matmul(
    query_heads,
    key_heads.transpose(-2, -1),
)

scaled_scores = raw_scores / (head_dim ** 0.5)

attention_weights = torch.softmax(
    scaled_scores,
    dim=-1,
)

print("原始注意力分数形状：", raw_scores.shape)
print("缩放后分数形状：", scaled_scores.shape)
print("注意力权重形状：", attention_weights.shape)

print("\n第1个头的注意力权重：")
print(attention_weights[0, 0])

print("\n第2个头的注意力权重：")
print(attention_weights[0, 1])

print("\n每个头的每行权重之和：")
print(attention_weights.sum(dim=-1))


# 每个头分别根据自己的权重汇总Value。
context_heads = torch.matmul(
    attention_weights,
    value_heads,
)

print("\n每个头的上下文向量形状：")
print(context_heads.shape)

print("\n第1个头的上下文向量：")
print(context_heads[0, 0])

print("\n第2个头的上下文向量：")
print(context_heads[0, 1])


#拼接多个头
print("\n===== 拼接多个注意力头 =====")

# (batch, heads, tokens, head_dim)
# → (batch, tokens, heads, head_dim)
context_transposed = context_heads.transpose(1, 2)

print("交换维度后的形状：", context_transposed.shape)


# 把heads和head_dim重新合并成embed_dim。
merged_context = context_transposed.contiguous().reshape(       #contiguous()会生成内存排列连续的 Tensor，方便后续安全地重新改变形状。
    batch_size,
    token_count,
    embed_dim,
)

print("拼接所有头后的形状：", merged_context.shape)
print("拼接所有头后的向量：")
print(merged_context)


# W_O负责混合多个头的信息，并保持模型维度不变。
output_layer = nn.Linear(   #混合不同注意力头的信息。学习怎样组合各个头的结果。保持输出维度仍为 embed_dim。让结果可以继续交给 Transformer 后续结构。
    embed_dim,
    embed_dim,
    bias=False,
).to(device)

multi_head_output = output_layer(merged_context)

print("\n经过输出Linear层后的形状：")
print(multi_head_output.shape)

print("多头注意力最终输出：")
print(multi_head_output)


#多头注意力中的 Padding Mask
print("\n===== 多头注意力的Padding Mask =====")

padded_tokens = [
    "python",
    "pytorch",
    "rag",
    "<PAD>",
]

pad_vector = torch.zeros(   #创造一个全0的向量
    batch_size,
    1,
    embed_dim,
    dtype=torch.float32,
    device=device,
)

padded_input = torch.cat(       #按第一维拼接
    [input_vectors, pad_vector],
    dim=1,
)

attention_mask = torch.tensor(
    [[1, 1, 1, 0]],
    dtype=torch.bool,
    device=device,
)

padded_token_count = padded_input.shape[1]

print("补齐后的输入形状：", padded_input.shape)
print("Attention Mask：", attention_mask)


#生成并拆分多头 Q、K、V
padded_query = query_layer(padded_input)
padded_key = key_layer(padded_input)
padded_value = value_layer(padded_input)


padded_query_heads = padded_query.reshape(
    batch_size,
    padded_token_count,
    num_heads,
    head_dim,
).transpose(1, 2)

padded_key_heads = padded_key.reshape(
    batch_size,
    padded_token_count,
    num_heads,
    head_dim,
).transpose(1, 2)

padded_value_heads = padded_value.reshape(
    batch_size,
    padded_token_count,
    num_heads,
    head_dim,
).transpose(1, 2)

print("\n加入PAD后的Q形状：", padded_query_heads.shape)
print("加入PAD后的K形状：", padded_key_heads.shape)
print("加入PAD后的V形状：", padded_value_heads.shape)


#生成多头 Key Mask
padded_scores = torch.matmul(
    padded_query_heads,
    padded_key_heads.transpose(-2, -1),
)

padded_scaled_scores = padded_scores / (
    head_dim ** 0.5
)

print("\n屏蔽前的分数形状：", padded_scaled_scores.shape)


# (batch, tokens)
# → (batch, 1, 1, tokens)
# 两个长度为1的维度会分别广播到所有头和所有Query。
key_mask = attention_mask.unsqueeze(1).unsqueeze(1)

print("原始Mask形状：", attention_mask.shape)
print("Key Mask形状：", key_mask.shape)


masked_scores = padded_scaled_scores.masked_fill(
    ~key_mask,
    float("-inf"),
)

masked_weights = torch.softmax(
    masked_scores,
    dim=-1,
)

print("\n屏蔽PAD后的权重形状：", masked_weights.shape)

print("所有头中PAD列的权重：")
print(masked_weights[:, :, :, -1])

print("\n所有头中每一行的权重和：")
print(masked_weights.sum(dim=-1))


#计算并拼接多头上下文向量
padded_context_heads = torch.matmul(
    masked_weights,
    padded_value_heads,
)

padded_context = padded_context_heads.transpose(
    1,
    2,
).contiguous().reshape(
    batch_size,
    padded_token_count,
    embed_dim,
)

padded_output = output_layer(padded_context)


# (batch, tokens) → (batch, tokens, 1)
# 将PAD对应的最终输出行清零。
query_mask = attention_mask.unsqueeze(-1).to(
    padded_output.dtype
)

padded_output = padded_output * query_mask

print("\n拼接后的上下文形状：", padded_context.shape)
print("最终输出形状：", padded_output.shape)

print("\n最终输出：")
print(padded_output)

print("\nPAD位置的最终输出：")
print(padded_output[:, -1, :])


#封装完整的多头自注意力层
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim必须能够被num_heads整除"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_layer = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )
        self.key_layer = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )
        self.value_layer = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )
        self.output_layer = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

    def split_heads(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, token_count, _ = tensor.shape

        # 把完整特征维度拆成多个头，再把heads移动到tokens前面。
        return tensor.reshape(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        input_vectors: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, _ = input_vectors.shape

        query = self.query_layer(input_vectors)
        key = self.key_layer(input_vectors)
        value = self.value_layer(input_vectors)

        query_heads = self.split_heads(query)
        key_heads = self.split_heads(key)
        value_heads = self.split_heads(value)

        scores = torch.matmul(
            query_heads,
            key_heads.transpose(-2, -1),
        )
        scaled_scores = scores / (self.head_dim ** 0.5)

        if attention_mask is not None:
            # 广播到所有注意力头和所有Query行。
            key_mask = (
                attention_mask
                .unsqueeze(1)
                .unsqueeze(1)
                .bool()
            )

            scaled_scores = scaled_scores.masked_fill(
                ~key_mask,
                float("-inf"),
            )

        attention_weights = torch.softmax(
            scaled_scores,
            dim=-1,
        )

        context_heads = torch.matmul(
            attention_weights,
            value_heads,
        )

        # 把(batch,heads,tokens,head_dim)恢复为(batch,tokens,embed_dim)。
        merged_context = context_heads.transpose(
            1,
            2,
        ).contiguous().reshape(
            batch_size,
            token_count,
            self.embed_dim,
        )

        output = self.output_layer(merged_context)

        if attention_mask is not None:
            query_mask = attention_mask.unsqueeze(-1).to(
                output.dtype
            )
            output = output * query_mask

        return output, attention_weights

#测试完整模型层
print("\n===== 测试完整多头注意力层 =====")

multi_head_attention = MultiHeadSelfAttention(
    embed_dim=8,
    num_heads=2,
).to(device)

final_output, final_attention_weights = (
    multi_head_attention(
        padded_input,
        attention_mask,
    )
)

print("输入形状：", padded_input.shape)
print("最终输出形状：", final_output.shape)
print(
    "注意力权重形状：",
    final_attention_weights.shape,
)

print("\n最终输出：")
print(final_output)

print("\nPAD位置的最终输出：")
print(final_output[:, -1, :])

print("\n所有头中PAD列的权重：")
print(final_attention_weights[:, :, :, -1])

print("\n每个头中每行权重之和：")
print(final_attention_weights.sum(dim=-1))