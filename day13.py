import torch
from torch import nn


torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

batch_size = 2
token_count = 5
embed_dim = 8
max_length = 10


#位置编码
# 使用相同向量模拟同一个 token 出现在不同位置
same_token_vector = torch.tensor(
    [0.2, 0.5, 0.1, 0.8, 0.3, 0.6, 0.4, 0.7],
    dtype=torch.float32,
    device=device,
)

token_embeddings = same_token_vector.reshape(1, 1, embed_dim).repeat(
    batch_size,
    token_count,
    1,
)

print("===== 原始词向量 =====")
print("词向量形状：", token_embeddings.shape)
print("位置0的词向量：", token_embeddings[0, 0])
print("位置1的词向量：", token_embeddings[0, 1])


# 为每个 token 生成位置编号：0、1、2、3、4
position_ids = torch.arange(
    token_count,
    device=device,
).unsqueeze(0).expand(batch_size, -1)

print("\n===== 位置编号 =====")
print(position_ids)
print("位置编号形状：", position_ids.shape)


# max_length 表示最多支持10个位置，每个位置用8维向量表示
position_embedding_layer = nn.Embedding(
    max_length,
    embed_dim,
).to(device)

position_embeddings = position_embedding_layer(position_ids)

print("\n===== 位置向量 =====")
print("位置向量形状：", position_embeddings.shape)
print("位置0的位置向量：", position_embeddings[0, 0])
print("位置1的位置向量：", position_embeddings[0, 1])


# 词向量和位置向量形状相同，可以对应位置直接相加
transformer_input = token_embeddings + position_embeddings

print("\n===== 加入位置信息后的向量 =====")
print("最终形状：", transformer_input.shape)
print("位置0的最终向量：", transformer_input[0, 0])
print("位置1的最终向量：", transformer_input[0, 1])

print(
    "两个位置的最终向量是否相同：",
    torch.equal(
        transformer_input[0, 0],
        transformer_input[0, 1],
    ),
)


#残差连接
print("\n===== 残差连接 =====")

residual_input = transformer_input

# 暂时用 Linear 模拟注意力子层，重点观察残差连接
simulated_attention_layer = nn.Linear(
    embed_dim,
    embed_dim,
).to(device)

sublayer_output = simulated_attention_layer(residual_input)

print("原始输入形状：", residual_input.shape)
print("子层输出形状：", sublayer_output.shape)

residual_output = residual_input + sublayer_output

print("残差输出形状：", residual_output.shape)

print("\n第一个token的原始输入：")
print(residual_input[0, 0])

print("\n第一个token的子层输出：")
print(sublayer_output[0, 0])

print("\n第一个token的残差输出：")
print(residual_output[0, 0])

print(
    "\n是否满足原始输入+子层输出：",
    torch.allclose(
        residual_output,
        residual_input + sublayer_output,
    ),
)


#LayerNorm
#残差相加后，不同特征的数值范围可能差别较大。LayerNorm 会把每个 token 的特征调整到相对稳定的范围，帮助模型训练。
#它不会把不同句子或不同 token 混在一起计算。
print("\n===== LayerNorm =====")

layer_norm = nn.LayerNorm(embed_dim).to(device)

normalized_output = layer_norm(residual_output)

print("标准化前形状：", residual_output.shape)
print("标准化后形状：", normalized_output.shape)

before_token = residual_output[0, 0]
after_token = normalized_output[0, 0]

print("\n第一个token标准化前：")
print(before_token)

print("标准化前的均值：", before_token.mean().item())
print(
    "标准化前的标准差：",
    before_token.std(unbiased=False).item(),
)

print("\n第一个token标准化后：")
print(after_token)

print("标准化后的均值：", after_token.mean().item())
print(
    "标准化后的标准差：",
    after_token.std(unbiased=False).item(),
)

print("\nLayerNorm可学习缩放参数：")
print(layer_norm.weight)

print("LayerNorm可学习偏移参数：")
print(layer_norm.bias)


#前馈神经网络FFN
print("\n===== 前馈神经网络FFN =====")

ffn_hidden_dim = embed_dim * 4

feed_forward = nn.Sequential(
    nn.Linear(embed_dim, ffn_hidden_dim),
    nn.GELU(),
    nn.Linear(ffn_hidden_dim, embed_dim),
).to(device)

expanded_features = feed_forward[0](normalized_output)
activated_features = feed_forward[1](expanded_features)
ffn_output = feed_forward[2](activated_features)

print("FFN输入形状：", normalized_output.shape)
print("扩大特征后的形状：", expanded_features.shape)
print("经过GELU后的形状：", activated_features.shape)
print("恢复特征后的形状：", ffn_output.shape)

print("\n第一个token的FFN输入：")
print(normalized_output[0, 0])

print("\n第一个token的FFN输出：")
print(ffn_output[0, 0])

#FFN后的第二次Add & Norm
print("\n===== FFN后的Add & Norm =====")

ffn_residual_output = normalized_output + ffn_output

ffn_layer_norm = nn.LayerNorm(embed_dim).to(device)

encoder_stage_output = ffn_layer_norm(
    ffn_residual_output
)

print("FFN输入形状：", normalized_output.shape)
print("FFN输出形状：", ffn_output.shape)
print("第二次残差输出形状：", ffn_residual_output.shape)
print("第二次LayerNorm输出形状：", encoder_stage_output.shape)

final_token = encoder_stage_output[0, 0]

print(
    "最终第一个token的均值：",
    final_token.mean().item(),
)
print(
    "最终第一个token的标准差：",
    final_token.std(unbiased=False).item(),
)


#组装Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm_after_attention = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, embed_dim),
        )

        self.norm_after_ffn = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_vectors: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        # PyTorch中True表示该Key需要被屏蔽，与我们的mask含义相反
        key_padding_mask = attention_mask == 0

        attention_output, attention_weights = self.attention(
            query=input_vectors,
            key=input_vectors,
            value=input_vectors,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        after_attention = self.norm_after_attention(
            input_vectors + attention_output
        )

        ffn_output = self.feed_forward(after_attention)

        encoder_output = self.norm_after_ffn(
            after_attention + ffn_output
        )

        # nn.MultiheadAttention只屏蔽PAD列，这里继续清零PAD输出行
        query_mask = attention_mask.unsqueeze(-1)
        encoder_output = encoder_output * query_mask

        return encoder_output, attention_weights


print("\n===== 完整Transformer Encoder Block =====")

num_heads = 2
ffn_hidden_dim = embed_dim * 4

encoder_block = TransformerEncoderBlock(
    embed_dim=embed_dim,
    num_heads=num_heads,
    ffn_hidden_dim=ffn_hidden_dim,
).to(device)

encoder_input = transformer_input.clone()

attention_mask = torch.tensor(
    [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
    ],
    dtype=torch.float32,
    device=device,
)

# 第二句话最后两个位置是PAD
encoder_input[1, 3:] = 0.0

encoder_output, attention_weights = encoder_block(
    encoder_input,
    attention_mask,
)

print("Encoder输入形状：", encoder_input.shape)
print("Attention Mask形状：", attention_mask.shape)
print("注意力权重形状：", attention_weights.shape)
print("Encoder输出形状：", encoder_output.shape)

print("\n第二句话的Attention Mask：")
print(attention_mask[1])

print("\n第二句话两个PAD位置的最终输出：")
print(encoder_output[1, 3:])

print(
    "\nPAD输出是否全部为0：",
    torch.allclose(
        encoder_output[1, 3:],
        torch.zeros_like(encoder_output[1, 3:]),
    ),
)