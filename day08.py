import torch

from torch.utils.data import (  
    TensorDataset,     
    DataLoader,         
)


print("===== Day 8 Dataset与DataLoader =====")


torch.manual_seed(42)


device = torch.device(      # 创建设备对象
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("训练设备：", device)


print("\n===== 准备原始数据 =====")


# 数据暂时保存在CPU
x_data = torch.tensor(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.2, 0.3],
        [0.3, 0.9],
        [0.9, 0.3],
        [0.8, 0.8],
        [0.9, 0.7],
        [0.7, 0.9],
    ],
    dtype=torch.float32,
)


y_data = torch.tensor(
    [
        [0.0],
        [0.0],
        [0.0],
        [1.0],
        [0.0],
        [0.0],
        [0.0],
        [1.0],
        [1.0],
        [1.0],
    ],
    dtype=torch.float32,
)


print("特征形状：", x_data.shape)
print("标签形状：", y_data.shape)
print("数据所在设备：", x_data.device)


print("\n===== 创建Dataset =====")


dataset = TensorDataset(        #以后相同索引的特征和标签会自动组成一组。因此 TensorDataset 负责把特征和标签配对。
    x_data,                     #注意：两个 Tensor 第一维的长度必须相同。
    y_data,
)


print("Dataset中的样本数量：", len(dataset))


first_features, first_label = dataset[0]

print("第一个样本的特征：", first_features)
print("第一个样本的标签：", first_label)


print("\n===== 创建DataLoader =====")


data_loader = DataLoader(       #主要负责：从 Dataset 中读取数据。把数据分成批次。根据设置决定是否打乱样本顺序。让训练循环可以逐批处理数据。
    dataset,
    batch_size=3,   #batch_size=3表示每个批次最多包含3个样本
    shuffle=False,  #shuffle=False表示不打乱数据顺序
)


print("每个批次最多包含：3个样本")
print("批次数量：", len(data_loader))   #len(data_loader)表示批次数量


print("\n===== 逐批读取数据 =====")


for batch_index, (  #batch_index表示批次索引，batch_features表示批次特征，batch_labels表示批次标签
    batch_features,
    batch_labels,
) in enumerate( #enumerate()函数用于将可迭代对象组合为一个索引序列，同时列出数据和数据下标，常用于在for循环中得到计数
    data_loader,
    start=1,
):
    print(f"\n第{batch_index}批：")

    print("批次特征：")
    print(batch_features)

    print("批次标签：")
    print(batch_labels)

    print(
        "特征形状：",
        batch_features.shape,
    )

    print(
        "标签形状：",
        batch_labels.shape,
    )

    print(
        "批次所在设备：",
        batch_features.device,
    )


#打乱数据并逐批移动到GPU
print("\n===== 打乱数据并移动批次到GPU =====")

device = torch.device(      
    "cuda" if torch.cuda.is_available() else "cpu"
)

shuffle_loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=True,   #shuffle=True表示打乱数据顺序
)

for batch_index, (batch_features, batch_labels) in enumerate(
    shuffle_loader,
    start=1,
):
    print(f"\n第{batch_index}批移动前：")
    print("特征：", batch_features)
    print("标签：", batch_labels)
    print("特征设备：", batch_features.device)
    print("标签设备：", batch_labels.device)

    # 将当前批次移动到GPU
    batch_features = batch_features.to(device)
    batch_labels = batch_labels.to(device)

    print(f"第{batch_index}批移动后：")
    print("特征形状：", batch_features.shape)
    print("标签形状：", batch_labels.shape)
    print("特征设备：", batch_features.device)
    print("标签设备：", batch_labels.device)


#小批量训练
import torch.nn as nn

print("\n===== 使用DataLoader进行小批量训练 =====")

# 训练时使用shuffle=True
train_loader = DataLoader(
    dataset,
    batch_size=3,   #batch_size=3表示每个批次最多包含3个样本
    shuffle=True,
)

model = nn.Sequential(  #nn.Sequential()表示按顺序堆叠多个神经网络层
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
).to(device)

loss_fn = nn.BCEWithLogitsLoss()    #nn.BCEWithLogitsLoss()表示使用二分类交叉熵损失函数，适用于二分类问题，结合了Sigmoid激活函数和二进制交叉熵损失计算。

optimizer = torch.optim.Adam(   #torch.optim.Adam()表示使用Adam优化器
    model.parameters(),         #model.parameters()表示获取模型的所有可学习参数
    lr=0.05,
)

epochs = 1000   #epochs表示训练轮数,一个 epoch：模型完整使用一次全部训练数据。一个 batch：完整数据中的一小批；

for epoch in range(epochs):
    model.train()   #model.train()表示将模型设置为训练模式，启用dropout和batch normalization等训练特性

    total_loss = 0.0    #total_loss表示当前epoch的总损失，用于计算平均损失

    for batch_index, (batch_features, batch_labels) in enumerate(   #enumerate()函数用于将可迭代对象组合为一个索引序列，同时列出数据和数据下标，常用于在for循环中得到计数
        train_loader,
        start=1,
    ):
        # 当前批次从CPU移动到GPU
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        # 第一个epoch观察每个批次的形状
        if epoch == 0:
            print(
                f"第{batch_index}批：",
                batch_features.shape,
                batch_labels.shape,
            )

        # 1. 前向传播
        logits = model(batch_features)

        # 2. 计算损失
        loss = loss_fn(logits, batch_labels)

        # 3. 清空上一批次的梯度
        optimizer.zero_grad()

        # 4. 反向传播
        loss.backward()

        # 5. 更新参数
        optimizer.step()

        # loss是当前批次的平均损失
        current_batch_size = batch_features.shape[0]
        total_loss += loss.item() * current_batch_size

    if (epoch + 1) % 200 == 0:
        average_loss = total_loss / len(dataset)    

        # 使用全部数据检查当前模型准确率
        model.eval()

        with torch.no_grad():
            all_features = x_data.to(device)
            all_labels = y_data.to(device)

            all_logits = model(all_features)
            all_probabilities = torch.sigmoid(all_logits)
            all_predictions = (
                all_probabilities >= 0.5
            ).float()

            accuracy = (
                all_predictions == all_labels
            ).float().mean().item()

        print(
            f"第{epoch + 1}轮，"
            f"平均损失={average_loss:.6f}，"
            f"准确率={accuracy:.2%}"
        )

#分批预测新数据
print("\n===== 使用DataLoader分批预测 =====")

candidate_names = [
    "候选人A",
    "候选人B",
    "候选人C",
    "候选人D",
]

test_features = torch.tensor([
    [0.95, 0.85],
    [0.85, 0.20],
    [0.25, 0.95],
    [0.75, 0.75],
], dtype=torch.float32)

# 测试数据只有特征，没有标签
test_dataset = TensorDataset(test_features)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
)

model.eval()

all_probabilities = []

with torch.no_grad():
    for batch_index, (batch_features,) in enumerate(    #batch_features表示批次特征，batch_index表示批次索引
        test_loader,    
        start=1,
    ):
        print(
            f"正在预测第{batch_index}批，"
            f"形状={batch_features.shape}"
        )

        batch_features = batch_features.to(device)  #to(device)表示将张量移动到指定设备（CPU或GPU）

        batch_logits = model(batch_features)
        batch_probabilities = torch.sigmoid(batch_logits)   #torch.sigmoid()表示对logits进行Sigmoid激活函数处理，将其转换为概率值

        # 从GPU移回CPU，再保存每一批结果
        all_probabilities.append(       #all_probabilities表示保存所有批次的预测概率
            batch_probabilities.cpu()   # 将结果移回CPU;减少GPU显存占用；方便长期保存结果；方便转换成 NumPy；方便进行后续普通 Python 数据处理。
        )

# 将多个批次的结果拼接起来
probabilities = torch.cat(  #cat()函数用于将多个张量沿指定维度拼接起来
    all_probabilities,
    dim=0,  #dim=0表示沿着第0维（行）拼接
)

predictions = (
    probabilities >= 0.5
).int() #int()函数用于将布尔类型的张量转换为整数类型的张量，True转换为1，False转换为0

print("所有概率的形状：", probabilities.shape)

for index, candidate_name in enumerate(candidate_names):
    python_score = test_features[index, 0].item()
    pytorch_score = test_features[index, 1].item()
    probability = probabilities[index].item()
    prediction = predictions[index].item()

    result_text = "匹配" if prediction == 1 else "不匹配"

    print(
        f"{candidate_name}："
        f"Python={python_score:.2f}，"
        f"PyTorch={pytorch_score:.2f}，"
        f"匹配概率={probability:.2%}，"
        f"预测结果={result_text}"
    )