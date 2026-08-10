import torch

from torch.utils.data import (  #torch.utils.data是PyTorch中用于处理数据的模块，提供了数据集和数据加载器的功能
    TensorDataset,  #eTensorDataset类用于将特征张量和标签张量组合成一个数据集，方便后续的数据加载和处理。
    DataLoader,
    random_split,   #random_split函数用于将数据集随机划分为训练集、验证集和测试集
)


print("===== Day 9 数据集划分与模型评估 =====")


torch.manual_seed(42)   #设置随机种子为42，以确保每次运行代码时生成的随机数序列相同，从而保证实验的可重复性。


device = torch.device(  #torch.device()函数用于指定计算设备，可以是CPU或GPU。这里根据是否有可用的CUDA设备来选择使用GPU还是CPU。
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备：", device)


# ============================================================
# 1. 创建模拟数据
# ============================================================

print("\n===== 创建模拟候选人数据 =====")


sample_count = 200  # 设置样本数量为200


# 生成200名候选人的两个特征
# 第1列：Python能力
# 第2列：PyTorch能力
features = torch.rand(  #torch.rand()函数用于生成指定形状的随机数张量，这里生成一个200行2列的张量，表示200名候选人的两个特征。
    sample_count,   
    2,
    dtype=torch.float32,
)


# 人工设定标签规则：
# Python能力 + PyTorch能力 >= 1.0，则标签为1
labels = (
    features[:, 0] + features[:, 1] >= 1.0      #features[:, 0]表示取所有行的第1列（Python能力），features[:, 1]表示取所有行的第2列（PyTorch能力），将两列相加后与1.0进行比较，得到一个布尔张量。
).float().unsqueeze(dim=1)  #unsqueeze(dim=1)函数用于在指定维度上增加一个维度，这里将标签张量的形状从(200,)变为(200, 1)，以便与特征张量的形状匹配。


print("特征形状：", features.shape)
print("标签形状：", labels.shape)
print("匹配样本数量：", int(labels.sum().item()))   #labels.sum()计算标签张量中所有元素的和，即匹配样本的数量。item()方法将张量转换为Python标量，int()函数将其转换为整数类型。
print(
    "不匹配样本数量：",
    int((labels == 0).sum().item()),
)


# ============================================================
# 2. 创建完整Dataset
# ============================================================

full_dataset = TensorDataset(
    features,
    labels,
)

print("完整数据集数量：", len(full_dataset))


# ============================================================
# 3. 划分训练集、验证集和测试集
# ============================================================

train_size = 140
validation_size = 30
test_size = 30


split_generator = torch.Generator().manual_seed(42)#torch.Generator()函数用于创建一个随机数生成器对象，manual_seed(42)方法用于设置随机种子为42，以确保每次运行代码时生成的随机数序列相同，从而保证实验的可重复性。


train_dataset, validation_dataset, test_dataset = random_split( #random_split()函数用于将完整数据集划分为训练集、验证集和测试集。它接受三个参数：完整数据集、每个子集的大小列表以及随机数生成器对象。
    full_dataset,
    [
        train_size,
        validation_size,
        test_size,
    ],
    generator=split_generator,  #generator参数用于指定随机数生成器对象，以确保划分的随机性可控。
)


print("\n===== 数据集划分结果 =====")
print("训练集数量：", len(train_dataset))
print("验证集数量：", len(validation_dataset))
print("测试集数量：", len(test_dataset))


# ============================================================
# 4. 分别创建DataLoader
# ============================================================

batch_size = 16 #batch_size参数用于指定每个批次的样本数量，这里设置为16。


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)


validation_loader = DataLoader( #DataLoader类用于将数据集分批次加载到模型中进行训练和评估。它接受三个主要参数：数据集、批次大小和是否打乱数据顺序。
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)


print("\n===== DataLoader信息 =====")
print("训练集批次数量：", len(train_loader))
print("验证集批次数量：", len(validation_loader))
print("测试集批次数量：", len(test_loader))


# 取出训练集的第一个批次
first_features, first_labels = next(
    iter(train_loader)  #iter()函数用于创建一个可迭代对象，next()函数用于获取该可迭代对象的下一个元素，这里获取训练集的第一个批次的特征和标签。
)


print("\n===== 训练集第一个批次 =====")
print("批次特征形状：", first_features.shape)
print("批次标签形状：", first_labels.shape)
print("批次所在设备：", first_features.device)

#封装单轮训练函数
import torch.nn as nn


# ============================================================
# 5. 创建模型
# ============================================================

print("\n===== 创建二分类模型 =====")


def create_model(): 
    """创建并返回一个新的二分类模型。"""

    return nn.Sequential(   #nn.sequential()函数用于将多个神经网络层按顺序组合成一个模型，这里创建了一个简单的前馈神经网络模型。
        nn.Linear(2, 8),    #nn.Linear()函数用于创建一个线性层，这里输入特征维度为2，输出特征维度为8。
        nn.ReLU(),
        nn.Linear(8, 1),
    )


model = create_model().to(device)   #model.to(device)方法用于将模型移动到指定的计算设备上，这里将模型移动到GPU或CPU上，以便进行训练和评估。


loss_fn = nn.BCEWithLogitsLoss()    #nn.BCEWithLogitsLoss()函数用于创建一个二分类交叉熵损失函数，它结合了Sigmoid激活函数和二分类交叉熵损失计算，适用于二分类任务。


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
)


print(model)


# ============================================================
# 6. 定义单轮训练函数
# ============================================================

def train_one_epoch(
    model,
    data_loader,    #data_loader参数用于传入训练数据的DataLoader对象，它提供了按批次加载训练数据的功能。
    loss_fn,
    optimizer,
    device,
):
    """使用全部训练数据完成一个epoch。"""

    # 切换到训练模式
    model.train()

    total_loss = 0.0    #total_loss变量用于累加整个训练集的总损失，用于计算平均损失。
    total_correct = 0   #total_correct变量用于累加整个训练集的预测正确的样本数量，用于计算准确率。
    total_samples = 0   #total_samples变量用于累加整个训练集的样本数量，用于计算平均损失和准确率。

    for batch_features, batch_labels in data_loader:
        # 当前批次移动到模型所在设备
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        # 1. 前向传播
        logits = model(batch_features)

        # 2. 计算当前批次损失
        loss = loss_fn(
            logits,
            batch_labels,
        )

        # 3. 清空上一批次的梯度
        optimizer.zero_grad()

        # 4. 反向传播
        loss.backward()

        # 5. 更新模型参数
        optimizer.step()

        # 将logits转换成分类结果
        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).float()

        # 当前批次的实际样本数量
        current_batch_size = batch_features.shape[0]    #batch_features.shape[0]表示当前批次的样本数量，即特征张量的第一维度大小。

        # 累加当前批次的样本损失总和
        total_loss += (
            loss.item() * current_batch_size     #loss.item()方法用于获取当前批次的损失值，并将其乘以当前批次的样本数量，以便累加整个训练集的总损失。
        )

        # 累加预测正确的样本数量
        total_correct += (
            predictions == batch_labels #predictions == batch_labels比较预测结果和真实标签，得到一个布尔张量。
        ).sum().item()

        # 累加已经处理的样本数量
        total_samples += current_batch_size

    # 计算整个训练集的平均损失
    average_loss = (
        total_loss / total_samples
    )

    # 计算整个训练集的准确率
    accuracy = (
        total_correct / total_samples
    )

    return average_loss, accuracy


# ============================================================
# 7. 测试一次训练
# ============================================================

print("\n===== 完成一个训练epoch =====")


train_loss, train_accuracy = train_one_epoch(
    model,
    train_loader,
    loss_fn,
    optimizer,
    device,
)


print(f"训练损失：{train_loss:.6f}")
print(f"训练准确率：{train_accuracy:.2%}")


#封装验证函数
# ============================================================
# 8. 定义模型评估函数
# ============================================================

def evaluate(
    model,
    data_loader,
    loss_fn,
    device,
):
    """评估模型，但不更新模型参数。"""

    # 切换到评估模式
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 评估阶段不需要梯度和计算图
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # 前向传播
            logits = model(batch_features)

            # 计算损失
            loss = loss_fn(
                logits,
                batch_labels,
            )

            # 将logits转换成分类结果
            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            ).float()

            current_batch_size = batch_features.shape[0]        #current_batch_size变量用于记录当前批次的样本数量，即特征张量的第一维度大小。

            total_loss += (
                loss.item() * current_batch_size
            )

            total_correct += (  #total_correct变量用于累加整个评估集的预测正确的样本数量，用于计算准确率。
                predictions == batch_labels
            ).sum().item()

            total_samples += current_batch_size #total_samples变量用于累加整个评估集的样本数量，用于计算平均损失和准确率。

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        total_correct / total_samples
    )

    return average_loss, accuracy


# ============================================================
# 9. 使用验证集检查模型
# ============================================================

print("\n===== 使用验证集评估模型 =====")


validation_loss, validation_accuracy = evaluate(
    model,
    validation_loader,
    loss_fn,
    device,
)


print(f"验证损失：{validation_loss:.6f}")
print(f"验证准确率：{validation_accuracy:.2%}")

#多轮训练并保存最佳模型
from pathlib import Path


# ============================================================
# 10. 重新创建模型，进行正式训练
# ============================================================

print("\n===== 正式进行多轮训练 =====")


# 创建一个全新的模型，避免受到前面一次测试训练的影响
model = create_model().to(device)   


optimizer = torch.optim.Adam(   #optimizer对象用于更新模型参数，这里使用Adam优化器，它是一种自适应学习率优化算法，能够在训练过程中动态调整每个参数的学习率，从而提高训练效率和收敛速度。
    model.parameters(), #parameters()方法用于获取模型的所有可训练参数，这些参数将被优化器更新。
    lr=0.01,
)


epochs = 100


# 正无穷，确保第一个验证损失一定更小
best_validation_loss = float("inf") #best_validation_loss变量用于记录当前训练过程中验证集的最低损失值，初始值设置为正无穷大，以确保第一次验证损失一定会更新该值。


# 无论从哪个目录运行代码，都把模型保存在day09.py旁边
best_model_path = ( 
    Path(__file__).resolve().parent
    / "best_model.pth"  #best_model_path变量用于指定最佳模型的保存路径，这里使用Path对象获取当前脚本所在目录，并将最佳模型保存为"best_model.pth"文件。
)


for epoch in range(epochs):
    # 使用训练集更新模型参数
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        loss_fn,
        optimizer,
        device,
    )

    # 使用验证集评估当前模型
    validation_loss, validation_accuracy = evaluate(
        model,
        validation_loader,
        loss_fn,
        device,
    )

    # 验证损失刷新最低纪录时，保存当前模型参数
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss

        torch.save(     #save()函数用于将模型的状态字典保存到指定路径，这里保存为"best_model.pth"文件。
            model.state_dict(), #state_dict()方法用于获取模型的所有参数和缓冲区的状态字典，它是一个Python字典对象，包含了模型的权重和偏置等信息。
            best_model_path,
        )

    # 每10轮输出一次训练情况
    if (epoch + 1) % 10 == 0:
        print(
            f"第{epoch + 1:3d}轮："
            f"训练损失={train_loss:.6f}，"
            f"训练准确率={train_accuracy:.2%}，"
            f"验证损失={validation_loss:.6f}，"
            f"验证准确率={validation_accuracy:.2%}"
        )


print("\n训练完成")
print(f"最低验证损失：{best_validation_loss:.6f}")
print("最佳模型保存位置：", best_model_path)