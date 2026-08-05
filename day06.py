import torch            #import torch 导入PyTorch库，通常用于深度学习和张量计算
import torch.nn as nn   #import torch.nn as nn 导入PyTorch的神经网络模块，通常用于构建神经网络模型


print("===== PyTorch自动求导 =====")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备：", device)


# 创建需要计算梯度的Tensor
x = torch.tensor(   
    3.0,    
    dtype=torch.float32,    #dtype=torch.float32表示数据类型为32位浮点数
    requires_grad=True,     #告诉 PyTorch：后面需要计算与 x 有关的梯度。
    device=device,          #device=device表示将Tensor放在指定的设备上
)

print("x：", x)
print("x是否需要梯度：", x.requires_grad)   #x.requires_grad属性表示当前Tensor是否需要计算梯度
print("x当前的梯度：", x.grad)  #grad属性表示当前Tensor的梯度，初始为None,因为这时还没有调用 y.backward()，PyTorch 尚未计算并保存梯度


# 建立计算过程
y = x ** 2 + 2 * x

print("y：", y)
print("y的计算来源：", y.grad_fn)   #grad_fn属性表示当前Tensor的计算来源，即是由哪个操作产生的


# 反向传播，计算y关于x的梯度
y.backward()    #backward()方法会计算y关于x的梯度，并将结果存储在x.grad中

print("反向传播后的x梯度：", x.grad)
print("转换成Python数字：", x.grad.item())

#梯度累加与清空梯度
print("\n===== 梯度累加实验 =====")

weight = torch.tensor(
    2.0,
    dtype=torch.float32,
    requires_grad=True,
    device=device,
)

print("初始梯度：", weight.grad)


# 第一次反向传播
loss1 = weight * 3
loss1.backward()

print("第一次反向传播后的梯度：", weight.grad)


# 第二次反向传播
loss2 = weight * 4
loss2.backward()

print("第二次反向传播后的梯度：", weight.grad)  #PyTorch 默认累加梯度：3 + 4 = 7


# 清空梯度
weight.grad.zero_() #weight.grad.zero_()方法会将梯度清零，避免梯度累加的影响；zero_() 最后的下划线表示：直接在原 Tensor 上进行修改。

print("清空后的梯度：", weight.grad)


# 第三次反向传播
loss3 = weight * 5
loss3.backward()

print("第三次反向传播后的梯度：", weight.grad)



####模型、损失函数和优化器###
print("\n===== 模型、损失函数和优化器 =====")


# 训练数据：规律是 y = 2x + 1
x_train = torch.tensor(     #x_train是一个二维张量，表示训练数据的输入特征，每一行是一个样本，每一列是一个特征
    [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ],
    device=device,
)

y_train = torch.tensor(
    [
        [3.0],
        [5.0],
        [7.0],
        [9.0],
        [11.0],
    ],
    device=device,
)

print("输入数据形状：", x_train.shape)
print("真实结果形状：", y_train.shape)


# 创建线性模型
model = nn.Linear(      #nn.Linear是PyTorch中用于创建线性模型的类，表示一个线性变换，即y = Wx + b，其中W是权重矩阵，b是偏置向量
    in_features=1,      #in_features=1表示输入特征的维度为1，即每个样本只有一个特征
    out_features=1,     #out_features=1表示输出特征的维度为1，即每个样本只有一个输出
).to(device)            #.to(device)表示将模型的参数放在指定的设备上（CPU或GPU），以便进行计算

print("模型：", model)
print("初始权重：", model.weight)
print("初始偏置：", model.bias)


# 创建损失函数
loss_fn = nn.MSELoss()  #nn.MSELoss()是PyTorch中用于计算均方误差（Mean Squared Error, MSE）损失的类，常用于回归问题。
                        #MSE损失函数的公式为：MSE = (1/n) * Σ(y_pred - y_true)^2，其中n是样本数量，y_pred是模型预测值，y_true是真实值。
                        #MSE损失函数的目标是最小化预测值与真实值之间的平方差，从而提高模型的预测精度。
# 创建优化器
optimizer = torch.optim.SGD(    #torch.optim.SGD是PyTorch中用于实现随机梯度下降（Stochastic Gradient Descent, SGD）优化算法的类，常用于训练神经网络模型。
    model.parameters(),         #model.parameters()是一个方法，用于获取模型的所有可训练参数（即权重和偏置），以便优化器可以更新这些参数。
    lr=0.01,        #lr=0.01表示学习率（learning rate），是优化器的一个超参数，用于控制每次参数更新的步长大小。较大的学习率可能导致训练不稳定，而较小的学习率可能导致训练收敛过慢。
)


# 第一次预测
prediction = model(x_train)     #称为前向传播：把训练数据交给模型，获得预测结果。

print("第一次预测：")
print(prediction)       


# 计算预测结果与真实结果之间的误差
loss = loss_fn(     #loss_fn是损失函数对象，调用它可以计算预测结果与真实结果之间的误差。
    prediction,
    y_train,
)

print("第一次损失：", loss.item())  #loss.item()方法用于获取损失值的标量表示，即将张量转换为Python的数字类型，便于打印和记录。


# 清空模型参数以前的梯度
optimizer.zero_grad()   #optimizer.zero_grad()方法用于清空模型参数以前的梯度，以避免梯度累加的影响。
                        #在每次反向传播之前，通常需要先清空梯度，以确保计算出的梯度是当前批次的梯度，而不是累加的历史梯度。

# 反向传播，计算模型参数的梯度
loss.backward()     # 计算应该怎么改变模型参数，以减少损失函数的值。它会根据损失函数对模型参数的偏导数，计算出每个参数的梯度，并将结果存储在每个参数的.grad属性中。

print("权重的梯度：", model.weight.grad)
print("偏置的梯度：", model.bias.grad)


# 根据梯度更新模型参数
optimizer.step()    #optimizer.step()方法用于根据计算出的梯度更新模型参数，即执行一次优化步骤。它会根据优化算法（如SGD）和学习率来调整模型的权重和偏置，从而使损失函数的值减小。

print("更新后的权重：", model.weight)
print("更新后的偏置：", model.bias)


# 再预测一次
prediction_after = model(x_train)

loss_after = loss_fn(
    prediction_after,
    y_train,
)

print("更新后的预测：")
print(prediction_after)

print("更新后的损失：", loss_after.item())

#多轮训练
print("\n===== 多轮训练线性模型 =====")


# 切换到训练模式
model.train()   #model.train()方法用于将模型切换到训练模式。在训练模式下，某些层（如Dropout和BatchNorm）会表现出不同的行为，以便在训练过程中提高模型的泛化能力。
                #与之相对的是model.eval()方法，用于将模型切换到评估模式，在评估模式下，这些层会表现出固定的行为，以便在测试或验证阶段获得稳定的输出。

# 训练1000轮
epochs = 1000   #epochs表示训练的轮数，即整个训练数据集将被用于训练模型的次数。每一轮训练都会对模型参数进行一次更新，以逐步优化模型的性能。

for epoch in range(epochs):     #for epoch in range(epochs)表示一个循环，从0到epochs-1（即0到999），用于执行多轮训练。在每一轮训练中，模型会对整个训练数据集进行一次前向传播、计算损失、反向传播和参数更新。
    # 1. 前向传播：使用当前参数进行预测
    prediction = model(x_train)

    # 2. 计算本轮损失
    loss = loss_fn(
        prediction,
        y_train,
    )

    # 3. 清空上一轮梯度
    optimizer.zero_grad()

    # 4. 反向传播，计算本轮梯度
    loss.backward()

    # 5. 更新权重和偏置
    optimizer.step()

    # 每100轮输出一次训练情况
    if (epoch + 1) % 100 == 0:
        print(
            f"第 {epoch + 1} 轮，"
            f"损失：{loss.item():.6f}"
        )


print("\n训练完成")

print(
    "模型学到的权重：",
    model.weight.item(),
)

print(
    "模型学到的偏置：",
    model.bias.item(),
)

#使用模型进行预测
print("\n===== 使用训练好的模型预测 =====")


# 切换到预测模式
model.eval()    #model.eval()方法用于将模型切换到评估模式。在评估模式下，某些层（如Dropout和BatchNorm）会表现出固定的行为，以便在测试或验证阶段获得稳定的输出。


# 创建没有出现在训练数据中的新输入
test_x = torch.tensor(
    [
        [6.0],
        [7.0],
        [8.0],
    ],
    dtype=torch.float32,
    device=device,
)


# 预测阶段不需要计算梯度
with torch.no_grad():   #torch.no_grad()是一个上下文管理器，用于在其作用域内禁用梯度计算。在预测阶段，我们通常不需要计算梯度，因为我们只关心模型的输出，而不需要更新模型参数。使用torch.no_grad()可以节省内存和计算资源，提高预测效率。
    test_prediction = model(test_x)     #test_prediction是模型对新输入test_x的预测结果。由于我们在torch.no_grad()上下文中进行预测，因此不会计算梯度，也不会影响模型的参数。


print("测试数据：")
print(test_x)

print("模型预测结果：")
print(test_prediction)

print(
    "当x=6时，模型预测：",
    test_prediction[0].item(),
)

print(
    "当x=7时，模型预测：",
    test_prediction[1].item(),
)

print(
    "当x=8时，模型预测：",
    test_prediction[2].item(),
)

print(
    "预测结果是否记录梯度：",
    test_prediction.requires_grad,  #test_prediction.requires_grad属性表示当前Tensor是否需要计算梯度。在预测阶段，我们通常不需要计算梯度，因此该属性为False
)