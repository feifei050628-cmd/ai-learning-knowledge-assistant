from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


random.seed(42)
torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备：", device)


samples = [
    {"text": "python pytorch transformer", "label": 1.0},
    {"text": "python rag embedding", "label": 1.0},
    {"text": "pytorch cuda deep learning", "label": 1.0},
    {"text": "python nlp attention", "label": 1.0},
    {"text": "transformer agent docker", "label": 1.0},
    {"text": "python llm fine tuning", "label": 1.0},
    {"text": "rag vector database python", "label": 1.0},
    {"text": "machine learning pytorch python", "label": 1.0},

    {"text": "excel word powerpoint", "label": 0.0},
    {"text": "sales marketing customer", "label": 0.0},
    {"text": "accounting finance tax", "label": 0.0},
    {"text": "human resources recruitment", "label": 0.0},
    {"text": "java spring mysql", "label": 0.0},
    {"text": "frontend html css javascript", "label": 0.0},
    {"text": "product operation communication", "label": 0.0},
    {"text": "administration office management", "label": 0.0},
]


texts = [
    sample["text"]
    for sample in samples
]

labels = torch.tensor(
    [
        sample["label"]
        for sample in samples
    ],
    dtype=torch.float32,
).unsqueeze(1)


print("\n===== 原始训练数据 =====")
print("样本总数：", len(samples))
print("文本数量：", len(texts))
print("标签形状：", labels.shape)
print("标签数据类型：", labels.dtype)

positive_count = int(labels.sum().item())
negative_count = len(labels) - positive_count

print("匹配样本数量：", positive_count)
print("不匹配样本数量：", negative_count)


print("\n===== 逐条查看数据 =====")

for index, sample in enumerate(
    samples,
    start=1,
):
    result_text = (
        "匹配"
        if sample["label"] == 1.0
        else "不匹配"
    )

    print(
        f"样本{index:02d}："
        f"{sample['text']:<35}"
        f"标签={sample['label']:.0f}，"
        f"含义={result_text}"
    )


#分词、词表、Padding和Attention Mask
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
UNK_ID = 1

max_length = 6


def tokenize(text: str) -> list[str]:
    return text.lower().strip().split()


vocab = {
    PAD_TOKEN: PAD_ID,
    UNK_TOKEN: UNK_ID,
}

for text in texts:
    tokens = tokenize(text)

    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)


print("\n===== 词表 =====")
print("词表大小：", len(vocab))

for token, token_id in vocab.items():
    print(f"{token:<15} → {token_id}")


def encode_text(
    text: str,
    vocab: dict[str, int],
    max_length: int,
) -> tuple[list[int], list[int]]:
    tokens = tokenize(text)

    token_ids = [
        vocab.get(token, UNK_ID)
        for token in tokens
    ]

    # 超过最大长度时，只保留前面的token
    token_ids = token_ids[:max_length]

    attention_mask = [1] * len(token_ids)

    padding_count = max_length - len(token_ids)

    token_ids += [PAD_ID] * padding_count
    attention_mask += [0] * padding_count

    return token_ids, attention_mask


all_token_ids = []
all_attention_masks = []

for text in texts:
    token_ids, attention_mask = encode_text(
        text,
        vocab,
        max_length,
    )

    all_token_ids.append(token_ids)
    all_attention_masks.append(attention_mask)


input_ids = torch.tensor(
    all_token_ids,
    dtype=torch.long,
)

attention_masks = torch.tensor(
    all_attention_masks,
    dtype=torch.float32,
)


print("\n===== 编码后的数据 =====")
print("Token ID形状：", input_ids.shape)
print("Attention Mask形状：", attention_masks.shape)
print("标签形状：", labels.shape)


print("\n===== 查看前3个样本 =====")

for index in range(3):
    print(f"\n样本{index + 1}")
    print("原始文本：", texts[index])
    print("分词结果：", tokenize(texts[index]))
    print("Token ID：", input_ids[index])
    print("Attention Mask：", attention_masks[index])
    print("标签：", labels[index])


#TensorDataset与DataLoader
print("\n===== 创建Dataset和DataLoader =====")

dataset = TensorDataset(
    input_ids,
    attention_masks,
    labels,
)

batch_size = 4

data_generator = torch.Generator()  #这行代码创建了一个 PyTorch 随机数生成器对象。
data_generator.manual_seed(42)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=data_generator,
)

print("Dataset样本数量：", len(dataset))
print("每个batch最多包含：", batch_size)
print("batch总数：", len(train_loader))


print("\n===== 查看每个batch =====")

for batch_index, (
    batch_input_ids,
    batch_attention_masks,
    batch_labels,
) in enumerate(
    train_loader,
    start=1,
):
    print(f"\n第{batch_index}批")

    print(
        "Token ID形状：",
        batch_input_ids.shape,
    )
    print(
        "Attention Mask形状：",
        batch_attention_masks.shape,
    )
    print(
        "标签形状：",
        batch_labels.shape,
    )

    print("本批标签：")
    print(batch_labels.squeeze(1))


#搭建Transformer文本分类模型
print("\n===== 创建Transformer文本分类模型 =====")


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
    ):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, embed_dim),
        )

        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # True表示该位置是PAD，MultiheadAttention会屏蔽对应的Key列。
        key_padding_mask = attention_mask == 0

        attention_output, _ = self.self_attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # 第一次残差连接与标准化。
        x = self.norm1(x + attention_output)

        ffn_output = self.ffn(x)

        # 第二次残差连接与标准化。
        x = self.norm2(x + ffn_output)

        # 将PAD对应的整行输出清零。
        query_mask = attention_mask.unsqueeze(-1)
        x = x * query_mask

        return x


class TransformerTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embed_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=PAD_ID,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embed_dim,
        )

        # ModuleList可以正确登记并管理多个Encoder层的参数。
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Linear(
            in_features=embed_dim,
            out_features=1,
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, token_count = input_ids.shape

        position_ids = torch.arange(
            token_count,
            device=input_ids.device,
        ).unsqueeze(0).expand(batch_size, -1)

        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(position_ids)

        x = token_vectors + position_vectors

        # 输入Encoder以前也将PAD位置清零。
        expanded_mask = attention_mask.unsqueeze(-1)
        x = x * expanded_mask

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, attention_mask)

        # 只对真实Token求平均，PAD不参与句子向量计算。
        vector_sum = (x * expanded_mask).sum(dim=1)
        valid_token_count = expanded_mask.sum(dim=1).clamp(min=1.0)
        sentence_vectors = vector_sum / valid_token_count

        return sentence_vectors

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        sentence_vectors = self.encode(
            input_ids,
            attention_mask,
        )

        logits = self.classifier(sentence_vectors)

        return logits


embed_dim = 16
num_heads = 4
ffn_hidden_dim = 64
num_layers = 2

model = TransformerTextClassifier(
    vocab_size=len(vocab),
    max_length=max_length,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ffn_hidden_dim=ffn_hidden_dim,
    num_layers=num_layers,
).to(device)

print(model)

print("\n===== 测试模型前向传播 =====")

sample_input_ids = input_ids[:4].to(device)
sample_attention_masks = attention_masks[:4].to(device)

model.eval()

with torch.no_grad():
    sample_sentence_vectors = model.encode(
        sample_input_ids,
        sample_attention_masks,
    )

    sample_logits = model(
        sample_input_ids,
        sample_attention_masks,
    )

print("Token ID形状：", sample_input_ids.shape)
print("句子向量形状：", sample_sentence_vectors.shape)
print("分类logits形状：", sample_logits.shape)
print("模型初始logits：")
print(sample_logits)


#训练Transformer分类模型
print("\n===== 训练Transformer文本分类模型 =====")

# BCEWithLogitsLoss内部已经包含Sigmoid，训练时不要提前手动调用Sigmoid。
loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.003,
)

epochs = 300

for epoch in range(epochs):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for (
        batch_input_ids,
        batch_attention_masks,
        batch_labels,
    ) in train_loader:
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_masks = batch_attention_masks.to(device)
        batch_labels = batch_labels.to(device)

        # 前向传播，输出形状和标签形状都是(batch_size, 1)。
        logits = model(
            batch_input_ids,
            batch_attention_masks,
        )

        loss = loss_fn(
            logits,
            batch_labels,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_batch_size = batch_labels.size(0)

        # loss.item()是当前batch的平均损失，乘以样本数后再累加。
        total_loss += loss.item() * current_batch_size
        total_samples += current_batch_size

        probabilities = torch.sigmoid(logits.detach())
        predictions = (probabilities >= 0.5).float()

        total_correct += (
            predictions == batch_labels
        ).sum().item()

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    if (epoch + 1) % 50 == 0:
        print(
            f"第{epoch + 1:3d}轮，"
            f"平均损失={average_loss:.6f}，"
            f"训练准确率={accuracy:.2%}"
        )

#使用训练好的模型预测新文本
print("\n===== 使用训练好的模型预测新文本 =====")

test_texts = [
    "python pytorch rag",
    "excel finance office",
    "python transformer attention",
    "sales marketing customer",
]

test_token_ids = []
test_attention_masks = []

for text in test_texts:
    token_ids, attention_mask = encode_text(
    text,
    vocab,
    max_length,
)

    test_token_ids.append(token_ids)
    test_attention_masks.append(attention_mask)

test_input_ids = torch.tensor(
    test_token_ids,
    dtype=torch.long,
)

test_masks = torch.tensor(
    test_attention_masks,
    dtype=torch.float32,
)

print("测试Token ID形状：", test_input_ids.shape)
print("测试Mask形状：", test_masks.shape)

test_input_ids = test_input_ids.to(device)
test_masks = test_masks.to(device)

model.eval()

with torch.no_grad():
    test_logits = model(
        test_input_ids,
        test_masks,
    )

    test_probabilities = torch.sigmoid(test_logits)
    test_predictions = (
        test_probabilities >= 0.5
    ).long()

# 转到CPU后再转换成普通Python列表，便于逐条输出。
logit_values = test_logits.cpu().squeeze(1).tolist()
probability_values = (
    test_probabilities.cpu().squeeze(1).tolist()
)
prediction_values = (
    test_predictions.cpu().squeeze(1).tolist()
)

print("\n===== 逐条输出预测结果 =====")

for (
    text,
    logit,
    probability,
    prediction,
) in zip(
    test_texts,
    logit_values,
    probability_values,
    prediction_values,
):
    result = "匹配" if prediction == 1 else "不匹配"

    print(f"\n文本：{text}")
    print(f"logit：{logit:.4f}")
    print(f"匹配概率：{probability:.2%}")
    print(f"预测结果：{result}")


#保存并重新加载模型
print("\n===== 保存模型Checkpoint =====")

checkpoint_path = (
    Path(__file__).resolve().parent
    / "day14_transformer_classifier.pth"
)

model_config = {
    "embed_dim": embed_dim,
    "num_heads": num_heads,
    "ffn_hidden_dim": ffn_hidden_dim,
    "num_layers": num_layers,
}

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "vocab": vocab,
    "max_length": max_length,
    "model_config": model_config,
    "epochs": epochs,
}

torch.save(
    checkpoint,
    checkpoint_path,
)

print("Checkpoint保存位置：", checkpoint_path)

#重新读取Checkpoint
print("\n===== 重新加载模型Checkpoint =====")

loaded_checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
)

loaded_vocab = loaded_checkpoint["vocab"]
loaded_max_length = loaded_checkpoint["max_length"]
loaded_config = loaded_checkpoint["model_config"]

loaded_model = TransformerTextClassifier(
    vocab_size=len(loaded_vocab),
    max_length=loaded_max_length,
    embed_dim=loaded_config["embed_dim"],
    num_heads=loaded_config["num_heads"],
    ffn_hidden_dim=loaded_config["ffn_hidden_dim"],
    num_layers=loaded_config["num_layers"],
).to(device)

loaded_model.load_state_dict(
    loaded_checkpoint["model_state_dict"]
)

loaded_model.eval()

print("模型参数加载完成")
print("加载的词表大小：", len(loaded_vocab))
print("加载的最大长度：", loaded_max_length)
print("加载的模型配置：", loaded_config)

#检查加载前后的结果是否一致
print("\n===== 对比加载前后的模型输出 =====")

with torch.no_grad():
    loaded_logits = loaded_model(
        test_input_ids,
        test_masks,
    )

outputs_are_equal = torch.allclose(
    test_logits,
    loaded_logits,
    atol=1e-6,
)

print("加载前后的输出是否一致：", outputs_are_equal)