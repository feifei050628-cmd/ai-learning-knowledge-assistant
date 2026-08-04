import numpy as np  #import numpy as np  #import numpy as np：导入 NumPy 库，提供了高性能的多维数组对象和相关操作函数。
import torch    #import torch：导入 PyTorch 库，提供了张量计算和深度学习功能。
import torch.nn.functional as F #import torch.nn.functional as F：导入 PyTorch 的函数式接口，提供了许多常用的神经网络操作函数，如激活函数、损失函数等。

print("===== PyTorch环境 =====")

device = torch.device(  #torch.device()：指定计算设备。CPU 或 GPU。
    "cuda" if torch.cuda.is_available() else "cpu"  #CUDA 可用：选择 "cuda"、CUDA 不可用：选择 "cpu"
)

print("PyTorch版本：", torch.__version__)
print("当前计算设备：", device)

print("\n===== 创建Tensor =====")

skills_tensor = torch.tensor(   #torch.tensor()：创建 Tensor。
    [1, 0, 1, 1],
    dtype=torch.float32,    #dtype=torch.float32,    #dtype：指定数据类型为 float32。
)

print("技能Tensor：", skills_tensor)
print("Python类型：", type(skills_tensor))
print("元素类型：", skills_tensor.dtype)
print("数组形状：", skills_tensor.shape)
print("维度数量：", skills_tensor.ndim)  
print("元素总数：", skills_tensor.numel())  #numel()：元素总数。    #注意 NumPy 使用：array.size
print("当前设备：", skills_tensor.device)   #device：Tensor 位于 CPU 还是 GPU。

#将 Tensor 移到 GPU
print("\n===== 移动到GPU =====")

skills_tensor = skills_tensor.to(device)    #因为 .to(device) 返回移动后的 Tensor，不会保证原变量自动改变。

print("移动后的Tensor：", skills_tensor)    #skills_tensor：Tensor 移动到 GPU 后的值。
print("移动后的设备：", skills_tensor.device)   #skills_tensor.device：Tensor 位于 CPU 还是 GPU。
#cuda：使用 NVIDIA GPU。    0：第0张显卡，也就是电脑的第一张显卡。

#Tensor 运算和切片
print("\n===== Tensor基础运算 =====")

print("乘以2：", skills_tensor * 2)
print("加上1：", skills_tensor + 1)
print("元素总和：", skills_tensor.sum())
print("平均值：", skills_tensor.mean())

print("第一个元素：", skills_tensor[0])
print("最后一个元素：", skills_tensor[-1])
print("第2到第3个元素：", skills_tensor[1:3])

#二维 Tensor
print("\n===== 二维Tensor =====")

scores = torch.tensor(
    [
        [80, 90, 70],
        [60, 75, 85],
    ],
    dtype=torch.float32,
    device=device,  #指定 Tensor 所在的设备。
)

print("分数Tensor：")
print(scores)

print("形状：", scores.shape)
print("维度数量：", scores.ndim)
print("元素总数：", scores.numel())
print("所在设备：", scores.device)

#二维索引和切片
print("\n===== 二维索引 =====")

print("第一行：", scores[0])
print("第二行：", scores[1])
print("第一行第二列：", scores[0, 1])
print("第二行第三列：", scores[1, 2])
print("第一列：", scores[:, 0])
print("第二列：", scores[:, 1])

#dim 按维度计算
#NumPy 使用 axis：  array.sum(axis=0)
#PyTorch 常用 dim： tensor.sum(dim=0)
print("\n===== 按维度计算 =====")

print("全部元素之和：", scores.sum())
print("全部元素平均值：", scores.mean())

print("每列总和：", scores.sum(dim=0))  #dim=0 → 压缩行，每列得到一个结果
print("每行总和：", scores.sum(dim=1))  #dim=1 → 压缩列，每行得到一个结果

print("每列平均值：", scores.mean(dim=0))   
print("每行平均值：", scores.mean(dim=1))

#将单个 Tensor 转成 Python 数字
total_score_tensor = scores.sum()   #scores.sum()：计算所有元素的总和，返回一个只包含一个元素的 Tensor。
total_score_number = total_score_tensor.item()  #.item() 只能用于只包含一个元素的 Tensor，常用于把损失值或准确率转换成普通 Python 数字。

print("Tensor结果：", total_score_tensor)
print("Python数字：", total_score_number)
print("Python数字类型：", type(total_score_number))

#矩阵乘法
print("\n===== 矩阵乘法 =====")

matrix_a = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
    ],
    dtype=torch.float32,
    device=device,
)

matrix_b = torch.tensor(
    [
        [1, 2],
        [3, 4],
        [5, 6],
    ],
    dtype=torch.float32,
    device=device,
)

print("矩阵A：")
print(matrix_a)
print("矩阵A形状：", matrix_a.shape)

print("矩阵B：")
print(matrix_b)
print("矩阵B形状：", matrix_b.shape)
#矩阵乘法要求：前一个矩阵的列数 = 后一个矩阵的行数
result1 = torch.matmul(matrix_a, matrix_b)
result2 = matrix_a @ matrix_b   #torch.matmul(a, b) 与 a @ b 在这里作用相同。

print("torch.matmul结果：")
print(result1)

print("@运算符结果：")
print(result2)

print("结果形状：", result1.shape)

#按元素乘法与矩阵乘法
same_shape_a = torch.tensor(
    [1, 2, 3],
    dtype=torch.float32,
    device=device,  #指定 Tensor 所在的设备。
)

same_shape_b = torch.tensor(
    [4, 5, 6],
    dtype=torch.float32,
    device=device,
)

print("按元素乘法：", same_shape_a * same_shape_b)  #对应位置分别相乘，保留各项结果。
print("向量点积：", same_shape_a @ same_shape_b)    #执行点积或矩阵乘法，进行乘积求和。

#矩阵转置
print("矩阵A转置：")
print(matrix_a.T)
print("转置后形状：", matrix_a.T.shape)
#matrix_a 的形状从 (2, 3) 变为 (3, 2)。


#NumPy 转 Tensor
print("\n===== NumPy与Tensor转换 =====")

numpy_array = np.array(
    [1, 2, 3],
    dtype=np.float32,
)

tensor_from_numpy = torch.from_numpy(numpy_array)   #torch.from_numpy()：将 NumPy 数组转换为 Tensor。注意：Tensor 与 NumPy 数组共享内存，修改一个会影响另一个。
                                                    #torch.from_numpy() 创建的 Tensor 默认位于 CPU。
print("NumPy数组：", numpy_array)
print("转换后的Tensor：", tensor_from_numpy)
print("Tensor设备：", tensor_from_numpy.device)

numpy_array[0] = 100

print("修改后的NumPy数组：", numpy_array)
print("受到影响的Tensor：", tensor_from_numpy)

#GPU Tensor 转 NumPy    NumPy 只能直接处理 CPU 内存中的数据，因此 CUDA Tensor 需要先移回 CPU：
gpu_tensor = torch.tensor(
    [10, 20, 30],
    dtype=torch.float32,
    device=device,
)
print("Tensor设备：", gpu_tensor.device)
numpy_from_tensor = gpu_tensor.cpu().numpy()    #numpy()：将 Tensor 转换为 NumPy 数组。注意：Tensor 与 NumPy 数组共享内存，修改一个会影响另一个。

print("GPU Tensor：", gpu_tensor)
print("转换后的NumPy数组：", numpy_from_tensor)

#综合练习：GPU 批量计算岗位相似度
print("\n===== PyTorch岗位相似度排名 =====")

skill_names = [
    "Python",
    "PyTorch",
    "RAG",
    "Docker",
    "Agent",
]

my_skills = torch.tensor(
    [1, 1, 0, 1, 0],
    dtype=torch.float32,
    device=device,
)

job_names = [
    "百度 - 大模型算法工程师",
    "阿里巴巴 - AI应用开发工程师",
    "字节跳动 - PyTorch算法工程师",
]

job_vectors = torch.tensor( 
    [
        [1, 1, 1, 0, 0],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 0],
    ],
    dtype=torch.float32,
    device=device,
)

print("个人技能形状：", my_skills.shape)
print("岗位矩阵形状：", job_vectors.shape)

#增加批次维度
my_skills_batch = my_skills.unsqueeze(0)    #unsqueeze(0)：在第 0 维增加一个维度，变成 (1, 5)，表示一个批次的技能向量。它没有增加数据，只改变形状。

print("增加维度前：", my_skills.shape)
print("增加维度后：", my_skills_batch.shape)

#批量计算余弦相似度
similarities = F.cosine_similarity(     #F.cosine_similarity()：计算两个张量沿指定维度的余弦相似度。余弦相似度衡量两个向量之间的相似性，值范围为 [-1, 1]，1 表示完全相同，-1 表示完全相反。
    job_vectors,
    my_skills_batch,
    dim=1,  #dim=1：沿第 2 维（列）计算相似度，即对每个岗位向量与个人技能向量进行比较。
)

print("相似度Tensor：", similarities)
print("相似度形状：", similarities.shape)

#排序
ranking_indices = torch.argsort(    #torch.argsort()：返回输入张量沿指定维度排序后的索引。可以用来获取排序后的元素位置。
    similarities,
    descending=True,        #descending=True：指定为 True 表示按降序排序，False 表示按升序排序。
)

print("排序后的索引：", ranking_indices)

#输出排名
for rank, index_tensor in enumerate(    #enumerate()：用于遍历可迭代对象，同时获取索引和值。这里用于遍历排序后的索引列表 ranking_indices。
    ranking_indices,
    start=1,
):
    index = index_tensor.item()     #item()：将只包含一个元素的 Tensor 转换为 Python 数字。这里用于获取岗位索引。
    similarity = similarities[index].item()

    print(
        f"{rank}. "
        f"{job_names[index]} - "
        f"{similarity:.2%}"
    )