import torch
import torch.nn as nn   # 神经网络模块


print("===== Day 7 二分类神经网络 =====")


# 固定随机数，方便重复实验
torch.manual_seed(42)   # 固定随机种子


# 自动选择CPU或GPU
device = torch.device(  # 自动选择设备
    "cuda" if torch.cuda.is_available() else "cpu"  #torch.cuda.is_available() 检查是否有可用的GPU
)

print("当前设备：", device)


print("\n===== 准备训练数据 =====")


# 每行代表一名候选人
# 第1列：Python掌握程度
# 第2列：PyTorch掌握程度
x_train = torch.tensor(
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
    device=device,
)


# 0：不匹配岗位
# 1：匹配岗位
y_train = torch.tensor(
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
    device=device,
)


print("训练特征：")
print(x_train)

print("训练标签：")
print(y_train)

print("特征形状：", x_train.shape)
print("标签形状：", y_train.shape)

print("特征数据类型：", x_train.dtype)
print("标签数据类型：", y_train.dtype)

print("特征所在设备：", x_train.device)
print("标签所在设备：", y_train.device)


print("\n===== 逐个查看样本 =====")

for index, (features, label) in enumerate(      # enumerate() 函数用于获取索引和元素
    zip(x_train, y_train),      #zip() 函数用于将两个可迭代对象打包成一个元组
    start=1,            # start=1 表示索引从1开始
):
    python_level = features[0].item()
    pytorch_level = features[1].item()
    target = int(label.item())

    print(
        f"样本{index}："
        f"Python={python_level:.1f}，"
        f"PyTorch={pytorch_level:.1f}，"
        f"标签={target}"    
    )

#建立多层神经网络
print("\n===== 建立多层神经网络 =====")


model = nn.Sequential( #nn.Sequential() 是一个容器，可以将多个神经网络层按顺序组合在一起
    # 第一层：2个输入特征转换成4个隐藏特征
    nn.Linear(
        in_features=2,  #输入特征数为2
        out_features=4, #输出特征数为4
    ),

    # 激活函数
    nn.ReLU(),  #ReLU() 是一种常用的激活函数，能够引入非线性，使模型能够学习更复杂的函数映射关系

    # 第二层：4个隐藏特征转换成1个输出
    nn.Linear(
        in_features=4,
        out_features=1,
    ),
).to(device)    # 将模型移动到指定设备


print("模型结构：")
print(model)        # 打印模型结构


print("\n===== 观察数据经过每一层 =====")


with torch.no_grad():   # torch.no_grad() 用于禁用梯度计算，提高推理速度
    # 经过第一层线性变换
    hidden_before_relu = model[0](x_train)  # 计算第一层的输出

    # 经过ReLU激活函数
    hidden_after_relu = model[1](   # 计算ReLU激活后的输出  计算规则：输入小于0 → 输出0   输入大于等于0 → 保留原值
        hidden_before_relu
    )

    # 经过最后一层，得到原始输出
    initial_logits = model[2](  # 计算最后一层的输出
        hidden_after_relu
    )


print("原始输入形状：", x_train.shape)

print(
    "第一层输出形状：",
    hidden_before_relu.shape,
)

print("第一层原始输出：")
print(hidden_before_relu)


print(
    "ReLU后形状：",
    hidden_after_relu.shape,
)

print("ReLU后的输出：")
print(hidden_after_relu)


print(
    "最终输出形状：",
    initial_logits.shape,
)

print("模型的初始logits：")
print(initial_logits)



#logit、Sigmoid 与分类结果
print("\n===== 理解logit和Sigmoid =====")


# 用几个容易观察的logit进行实验
demo_logits = torch.tensor(
    [
        [-3.0],
        [-1.0],
        [0.0],
        [1.0],
        [3.0],
    ],
    dtype=torch.float32,
    device=device,
)


demo_probabilities = torch.sigmoid(  # torch.sigmoid() 是一个激活函数，将logit转换为概率值，范围在0到1之间
    demo_logits
)


print("实验logits：")
print(demo_logits)

print("经过Sigmoid后的概率：")
print(demo_probabilities)


print("\n===== 初始模型的分类结果 =====")


# 把模型原始输出转换成概率
initial_probabilities = torch.sigmoid(  #Sigmoid(z) = 1 / (1 + e^(-z))
    initial_logits
)


# 概率大于等于0.5，预测为1
# 概率小于0.5，预测为0
initial_predictions = (
    initial_probabilities >= 0.5    #torch.sigmoid() 将logit转换为概率值，范围在0到1之间   probabilities >= 0.5 返回一个布尔张量，表示每个样本的预测结果是否大于等于0.5
).float()   


# 计算初始准确率
initial_accuracy = (    # 计算预测正确的样本数
    initial_predictions == y_train  
).float().mean()       #mean() 计算平均值，即准确率


print("模型初始logits：")
print(initial_logits)

print("模型初始概率：")
print(initial_probabilities)

print("模型初始分类结果：")
print(initial_predictions)

print(
    "模型初始准确率：",
    f"{initial_accuracy.item():.2%}",
)



#损失函数与正式训练
print("\n===== 训练二分类神经网络 =====")


# 二分类损失函数
loss_fn = nn.BCEWithLogitsLoss()    # BCEWithLogitsLoss() 结合了sigmoid激活函数和二元交叉熵损失函数;Sigmoid + Binary Cross Entropy
                                    #也就是：原始logit→ 转换成概率→ 根据真实标签计算分类损失

# Adam优化器
optimizer = torch.optim.Adam(   #Adam() 是一种自适应学习率优化算法，能够根据梯度的一阶矩和二阶矩动态调整学习率;和torch.optim.SGD都负责根据梯度更新模型参数，让损失逐渐减小
    model.parameters(),     # 优化模型的可学习参数
    lr=0.01,
)
    #SGD：按照统一学习率更新参数，原理比较基础      Adam：

# 切换到训练模式
model.train()


epochs = 2000


for epoch in range(epochs):
    # 1. 前向传播，得到原始logits
    logits = model(x_train)

    # 2. 直接使用logits计算损失
    loss = loss_fn(     #loss_fn() 计算模型输出与真实标签之间的损失
        logits,
        y_train,
    )

    # 3. 清空上一轮梯度
    optimizer.zero_grad()

    # 4. 反向传播
    loss.backward()

    # 5. 更新模型参数
    optimizer.step()

    # 每200轮检查一次训练结果
    if (epoch + 1) % 200 == 0:
        with torch.no_grad():
            check_logits = model(x_train)

            check_probabilities = torch.sigmoid(
                check_logits
            )

            check_predictions = (
                check_probabilities >= 0.5
            ).float()

            check_accuracy = (
                check_predictions == y_train
            ).float().mean()

        print(
            f"第{epoch + 1}轮，"
            f"损失={loss.item():.6f}，"
            f"准确率={check_accuracy.item():.2%}"
        )


#预测新的候选人
print("\n===== 预测新的候选人 =====")


candidate_names = [
    "候选人A",
    "候选人B",
    "候选人C",
    "候选人D",
]


# 这些数据没有参与训练
test_data = torch.tensor(
    [
        [0.95, 0.85],
        [0.85, 0.20],
        [0.25, 0.95],
        [0.75, 0.75],
    ],
    dtype=torch.float32,
    device=device,
)


# 切换到预测模式
model.eval()


with torch.no_grad():
    # 模型原始输出
    test_logits = model(test_data)

    # 转换成匹配概率
    test_probabilities = torch.sigmoid(
        test_logits
    )

    # 转换成最终分类结果
    test_predictions = (
        test_probabilities >= 0.5
    ).float()


print("测试数据形状：", test_data.shape)
print("测试logits：")
print(test_logits)

print("测试概率：")
print(test_probabilities)

print("测试分类结果：")
print(test_predictions)


print("\n===== 逐个输出预测结果 =====")


for index, candidate_name in enumerate(
    candidate_names
):
    python_level = test_data[index, 0].item()   #test_data[index, 0] 获取第index个候选人的Python掌握程度，.item() 将张量转换为Python标量
    pytorch_level = test_data[index, 1].item()

    probability = (
        test_probabilities[index].item()
    )

    prediction = int(
        test_predictions[index].item()
    )

    result_text = (
        "匹配" if prediction == 1
        else "不匹配"
    )

    print(
        f"{candidate_name}："
        f"Python={python_level:.2f}，"
        f"PyTorch={pytorch_level:.2f}，"
        f"匹配概率={probability:.2%}，"
        f"预测结果={result_text}"
    )