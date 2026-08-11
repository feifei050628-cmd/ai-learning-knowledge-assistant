#什么是分词
print("===== Day 10：NLP文本编码 =====")


def tokenize(text):     #tokenize函数用于将一段岗位文本清洗并拆分成单词列表
    """将一段岗位文本清洗并拆分成单词列表。"""

    cleaned_text = text.lower() #text.lower()方法将字符串中的所有字符转换为小写字母
    cleaned_text = cleaned_text.replace(",", " ")   #replace()方法将字符串中的逗号替换为空格
    cleaned_text = cleaned_text.replace("/", " ")   #replace()方法将字符串中的斜杠替换为空格

    tokens = cleaned_text.split()   #split()方法将字符串拆分为单词列表，默认以空格为分隔符

    return tokens


job_texts = [
    "Python, PyTorch, RAG, Docker",
    "Python / Transformer / Agent",
    "PyTorch, NLP, Python",
]


print("\n===== 文本分词 =====")

for text in job_texts:
    tokens = tokenize(text) 

    print("原始文本：", text)
    print("分词结果：", tokens)
    print()


#构建词表 Vocabulary
print("\n===== 构建词表 =====")


vocab = {
    "<PAD>": 0,     #<PAD> token用于填充序列，使其具有相同的长度
    "<UNK>": 1,     #<UNK> token用于表示未知的单词
}


for text in job_texts:
    tokens = tokenize(text)

    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)   #len(vocab)返回词表中当前的单词数量，并将其作为新单词的ID分配给该单词


print("完整词表：")

for token, token_id in vocab.items():   
    print(f"{token:12} -> {token_id}")  


print("词表大小：", len(vocab))

#将文本转换成 token ID
print("\n===== 文本转换为Token ID =====")


def encode(text, vocab):    
    """把文本中的每个单词转换成对应的token ID。"""

    tokens = tokenize(text) 
    token_ids = []

    for token in tokens:
        token_id = vocab.get(
            token,
            vocab["<UNK>"],
        )

        token_ids.append(token_id)

    return token_ids


sample_text = "Python PyTorch LangChain"

sample_tokens = tokenize(sample_text)
sample_ids = encode(sample_text, vocab)

print("原始文本：", sample_text)
print("分词结果：", sample_tokens)
print("Token ID：", sample_ids)


#Padding 和 Attention Mask
print("\n===== Padding与Attention Mask =====")


def encode_and_pad(text, vocab, max_length):    
    """把文本转换成固定长度的Token ID和Attention Mask。"""

    token_ids = encode(text, vocab)  

    # 如果句子太长，只保留前max_length个编号
    token_ids = token_ids[:max_length]

    # 真实单词的位置用1表示
    attention_mask = [1] * len(token_ids)

    # 计算还需要补多少个<PAD>
    padding_length = max_length - len(token_ids)

    # 使用<PAD>补齐Token ID
    token_ids = token_ids + [
        vocab["<PAD>"]
    ] * padding_length

    # 补齐的位置在Attention Mask中使用0表示
    attention_mask = attention_mask + [
        0
    ] * padding_length  

    return token_ids, attention_mask


short_text = "Python RAG"

short_ids, short_mask = encode_and_pad(
    short_text,
    vocab,
    max_length=5,
)

print("原始文本：", short_text)
print("Token ID：", short_ids)
print("Attention Mask：", short_mask)


long_text = "Python PyTorch RAG Docker Agent Transformer"

long_ids, long_mask = encode_and_pad(
    long_text,
    vocab,
    max_length=5,
)

print("\n原始文本：", long_text)
print("Token ID：", long_ids)
print("Attention Mask：", long_mask)


#使用 nn.Embedding 生成词向量
print("\n===== Embedding词向量 =====")  # 输出本阶段标题，方便查看运行结果。


import torch  # 导入PyTorch，用于创建和计算Tensor。
from torch import nn  # 从PyTorch中导入神经网络模块nn。


device = torch.device(  # 创建一个表示计算设备的对象。
    "cuda" if torch.cuda.is_available() else "cpu"  # 有可用GPU就使用cuda，否则使用cpu。
)

print("当前设备：", device)  # 输出当前程序使用的计算设备。


embedding_dim = 4  # 设置每个单词转换后的词向量长度为4。


embedding = nn.Embedding(  # 创建一个Embedding层，用于把token ID转换成词向量。
    num_embeddings=len(vocab),  # 词表有多少个词，就需要准备多少个词向量。
    embedding_dim=embedding_dim,  # 每个词向量由4个浮点数组成。
    padding_idx=vocab["<PAD>"],  # 指定<PAD>的编号，使补齐位置不参与正常学习。
).to(device)  # 把Embedding层移动到GPU或CPU。


embedding_text = "Python RAG"  # 准备一段需要转换成词向量的测试文本。


embedding_ids, embedding_mask = encode_and_pad(  # 对文本进行编码和补齐。
    embedding_text,  # 传入原始文本。
    vocab,  # 传入前面构建好的词表。
    max_length=5,  # 将文本统一处理成5个token的位置。
)


input_ids = torch.tensor(  # 把Python列表转换成PyTorch Tensor。
    [embedding_ids],  # 外面增加一层列表，表示这里只有1句话。
    dtype=torch.long,  # token ID是词表行号，因此必须使用整数类型long。
    device=device,  # 直接把Tensor创建在当前计算设备上。
)


attention_mask_tensor = torch.tensor(  # 把Attention Mask列表转换成Tensor。
    [embedding_mask],  # 外面增加一层列表，表示这里只有1句话。
    dtype=torch.float32,  # 后面需要和词向量相乘，因此使用浮点类型。
    device=device,  # 把Mask Tensor创建在当前计算设备上。
)


embedded_tokens = embedding(input_ids)  # 根据每个token ID查找对应的词向量。


print("原始文本：", embedding_text)  # 输出转换前的文本。
print("输入Token ID：", input_ids)  # 输出补齐后的token ID。
print("Token ID形状：", input_ids.shape)  # 查看输入形状，应该是(1, 5)。

print("\nAttention Mask：", attention_mask_tensor)  # 输出Attention Mask。
print("Mask形状：", attention_mask_tensor.shape)  # 查看Mask形状，应该是(1, 5)。

print("\n每个Token的词向量：")  # 输出提示文字。
print(embedded_tokens)  # 输出Embedding生成的所有词向量。
print("词向量形状：", embedded_tokens.shape)  # 查看输出形状，应该是(1, 5, 4)。


#把多个词向量合成句子向量
print("\n===== 生成句子向量 =====")


# (1, 5) → (1, 5, 1)，增加一个维度才能和(1, 5, 4)的词向量相乘。
expanded_mask = attention_mask_tensor.unsqueeze(-1)

print("扩展前的Mask形状：", attention_mask_tensor.shape)
print("扩展后的Mask形状：", expanded_mask.shape)


# 真实词的位置乘1，PAD位置乘0。
masked_embeddings = embedded_tokens * expanded_mask

print("Mask处理后的词向量：")
print(masked_embeddings)


# dim=1表示沿token维度求和，把5个token向量合成一个向量。
sum_embeddings = masked_embeddings.sum(dim=1)

# 统计每句话真实token的数量；clamp防止全是PAD时除以0。
valid_token_counts = expanded_mask.sum(dim=1).clamp(min=1.0)

# 用真实词向量之和除以真实词数量，得到平均句子向量。
sentence_vector = sum_embeddings / valid_token_counts


print("词向量求和结果：", sum_embeddings)
print("真实Token数量：", valid_token_counts)
print("最终句子向量：", sentence_vector)
print("句子向量形状：", sentence_vector.shape)


#批量把多条 JD 转换成句子向量。
print("\n===== 批量生成JD句子向量 =====")


def encode_batch(texts, vocab, max_length, device):
    """批量把多段文本转换成Token ID Tensor和Mask Tensor。"""

    all_token_ids = []
    all_attention_masks = []

    for text in texts:
        token_ids, attention_mask = encode_and_pad(
            text,
            vocab,
            max_length,
        )

        all_token_ids.append(token_ids)
        all_attention_masks.append(attention_mask)

    # all_token_ids是“列表中包含多个列表”，转换后形状为(文本数量, max_length)。
    input_ids = torch.tensor(
        all_token_ids,
        dtype=torch.long,
        device=device,
    )

    attention_masks = torch.tensor(
        all_attention_masks,
        dtype=torch.float32,
        device=device,
    )

    return input_ids, attention_masks


batch_job_texts = [
    "Python PyTorch RAG",
    "Python Docker Agent",
    "NLP Transformer RAG",
]


batch_input_ids, batch_attention_masks = encode_batch(
    batch_job_texts,
    vocab,
    max_length=5,
    device=device,
)


batch_embeddings = embedding(batch_input_ids)

# Mask从(3, 5)扩展到(3, 5, 1)，才能应用到每个token的4维向量。
expanded_batch_masks = batch_attention_masks.unsqueeze(-1)

masked_batch_embeddings = (
    batch_embeddings * expanded_batch_masks
)

# 沿token维度求和：(3, 5, 4) → (3, 4)。
batch_embedding_sums = masked_batch_embeddings.sum(dim=1)

# 每条文本分别统计真实token数量，形状为(3, 1)。
batch_valid_counts = (
    expanded_batch_masks
    .sum(dim=1)
    .clamp(min=1.0)
)

batch_sentence_vectors = (
    batch_embedding_sums / batch_valid_counts
)


print("批量Token ID：")
print(batch_input_ids)
print("Token ID形状：", batch_input_ids.shape)

print("\n批量Attention Mask：")
print(batch_attention_masks)
print("Mask形状：", batch_attention_masks.shape)

print("\n批量词向量形状：", batch_embeddings.shape)

print("\n最终JD句子向量：")
print(batch_sentence_vectors)
print("句子向量形状：", batch_sentence_vectors.shape)


for index, text in enumerate(batch_job_texts):
    print(f"\nJD {index + 1}：{text}")
    print("对应句子向量：", batch_sentence_vectors[index])



#计算个人技能与JD的余弦相似度
print("\n===== 个人技能与JD匹配 =====")


import torch.nn.functional as F


user_skills_text = "Python PyTorch"


user_input_ids, user_attention_masks = encode_batch(
    [user_skills_text],  # 使用列表包装，表示这里只有一条文本。
    vocab,
    max_length=5,
    device=device,
)


user_token_embeddings = embedding(user_input_ids)

user_expanded_mask = user_attention_masks.unsqueeze(-1)

user_masked_embeddings = (
    user_token_embeddings * user_expanded_mask
)

user_embedding_sum = user_masked_embeddings.sum(dim=1)

user_valid_count = (
    user_expanded_mask
    .sum(dim=1)
    .clamp(min=1.0)
)

user_sentence_vector = (
    user_embedding_sum / user_valid_count
)


# 把个人向量从(1, 4)扩展成(3, 4)，分别与3条JD进行比较。
expanded_user_vector = user_sentence_vector.expand_as(
    batch_sentence_vectors
)


# dim=1表示分别计算每一行4维向量之间的余弦相似度。
similarities = F.cosine_similarity(
    batch_sentence_vectors,
    expanded_user_vector,
    dim=1,
)


# 返回相似度从高到低排列时对应的JD索引。
ranked_indices = torch.argsort(
    similarities,
    descending=True,
)


print("个人技能：", user_skills_text)
print("个人句子向量：", user_sentence_vector)
print("所有相似度：", similarities)

print("\n岗位匹配排序：")

for rank, index_tensor in enumerate(
    ranked_indices,
    start=1,
):
    # Tensor索引转换成普通Python整数，方便访问列表。
    index = index_tensor.item()

    # 单元素Tensor转换成普通Python浮点数。
    score = similarities[index].item()

    print(
        f"第{rank}名："
        f"{batch_job_texts[index]}，"
        f"相似度={score:.4f}"
    )