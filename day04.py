import numpy as np      


skills_vector = np.array([1, 0, 1, 1])  #np.array()：把列表转换为 NumPy 数组。

print("技能向量：", skills_vector)
print("数据类型：", type(skills_vector))
print("元素类型：", skills_vector.dtype)    #dtype：数组中元素的数据类型。
print("维度数量：", skills_vector.ndim)     #ndim：数组有几个维度。
print("数组形状：", skills_vector.shape)    #shape：每个维度有多少个元素。
print("元素数量：", skills_vector.size)     #size：数组总共有多少个元素。
#比较列表与 NumPy 数组
python_list = [1, 2, 3]
numpy_array = np.array([1, 2, 3])

print("列表乘以2：", python_list * 2)   #Python 列表乘以 2：把列表内容重复两次。
print("数组乘以2：", numpy_array * 2)   #NumPy 数组乘以 2：对数组中的每个元素进行乘法运算。NumPy 这种一次操作整个数组的方式叫“向量化计算”。
#索引与切片
print("第一个技能值：", skills_vector[0])
print("最后一个技能值：", skills_vector[-1])
print("第2到第3个值：", skills_vector[1:3])

skills_vector[1] = 1
print("修改后的技能向量：", skills_vector)

#二维数组
print("\n===== 二维数组 =====")

numbers = np.array([1, 2, 3, 4, 5, 6])
matrix = numbers.reshape(2, 3)  #reshape() 只改变排列形状，不改变元素本身和元素总数。

print("原始数组：")
print(numbers)

print("二维数组：")
print(matrix)

print("维度数量：", matrix.ndim)
print("数组形状：", matrix.shape)   #两行，每行3个元素
print("元素总数：", matrix.size)

#二维数组索引
print("\n===== 二维数组索引 =====")

print("第一行：", matrix[0])
print("第二行：", matrix[1])

print("第一行第二列：", matrix[0, 1])
print("第二行第三列：", matrix[1, 2])

print("第一列：", matrix[:, 0])     #:表示所有行，0表示第一列
print("第二列：", matrix[:, 1])     #:表示所有行，1表示第二列

#矩阵转置
print("\n===== 矩阵转置 =====")

transposed_matrix = matrix.T    #T 属性用于获取矩阵的转置

print("转置前：")
print(matrix)
print("转置前形状：", matrix.shape)

print("转置后：")
print(transposed_matrix)
print("转置后形状：", transposed_matrix.shape)

#按维度计算
print("\n===== 聚合计算 =====")

scores = np.array([
    [80, 90, 70],
    [60, 75, 85],
])

print("分数矩阵：")
print(scores)

print("全部元素之和：", scores.sum())
print("全部元素平均值：", scores.mean())
print("最大值：", scores.max())
print("最小值：", scores.min())
#这些操作把多个数汇总成一个结果，所以称为“聚合计算”。

#理解 axis
print("\n===== 按维度计算 =====")

print("每一列的总和：", scores.sum(axis=0))#第0维被压缩掉，保留列。
print("每一行的总和：", scores.sum(axis=1))#第1维被压缩掉，保留行。

print("每一列的平均值：", scores.mean(axis=0))
print("每一行的平均值：", scores.mean(axis=1))

print("每一列的最大值：", scores.max(axis=0))
print("每一行的最大值：", scores.max(axis=1))

#向量点积
print("\n===== 向量点积 =====")

my_vector = np.array([1, 1, 0, 1], dtype=float)     #dtype=float：指定数组元素的数据类型为浮点数。
job_vector = np.array([1, 0, 1, 1], dtype=float)

dot_product = np.dot(my_vector, job_vector)     #dot()：计算两个向量的点积。

print("个人技能向量：", my_vector)
print("岗位技能向量：", job_vector)
print("向量点积：", dot_product)

#向量长度
my_length = np.linalg.norm(my_vector)   #linalg.norm()：计算向量的长度（即欧几里得范数）。
job_length = np.linalg.norm(job_vector) 

print("个人向量长度：", my_length)
print("岗位向量长度：", job_length)

#余弦相似度
cosine_similarity = dot_product / (my_length * job_length)  #dot_product：向量点积，my_length：个人向量长度，job_length：岗位向量长度。

print("余弦相似度：", cosine_similarity)    
print(f"余弦相似度：{cosine_similarity:.2%}")

#封装成函数
def calculate_cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)  #计算两个向量的点积。

    length1 = np.linalg.norm(vector1)   #计算向量的长度（即欧几里得范数）。
    length2 = np.linalg.norm(vector2)

    denominator = length1 * length2

    if denominator == 0:
        return 0.0

    return dot_product / denominator

similarity = calculate_cosine_similarity(
    my_vector,
    job_vector,
)

print(f"函数计算结果：{similarity:.2%}")

#综合练习：岗位相似度排名
print("\n===== 岗位相似度排名 =====")

skill_names = [
    "Python",
    "PyTorch",
    "RAG",
    "Docker",
    "Agent",
]

my_skills = np.array([1, 1, 0, 1, 0], dtype=float)

jobs = [
    {
        "company": "百度",
        "position": "大模型算法工程师",
        "vector": np.array([1, 1, 1, 0, 0], dtype=float),
    },
    {
        "company": "阿里巴巴",
        "position": "AI应用开发工程师",
        "vector": np.array([1, 0, 1, 1, 1], dtype=float),
    },
    {
        "company": "字节跳动",
        "position": "PyTorch算法工程师",
        "vector": np.array([1, 1, 0, 0, 0], dtype=float),
    },
]

#计算每个岗位的相似度
results = []

for job in jobs:
    similarity = calculate_cosine_similarity(   #calculate_cosine_similarity()：调用之前定义的函数，计算个人技能向量与岗位技能向量之间的余弦相似度。
        my_skills,
        job["vector"],
    )

    results.append({    #results.append()：将岗位信息和相似度结果添加到 results 列表中。
        "company": job["company"],      
        "position": job["position"],
        "similarity": similarity,
    })

#按相似度从高到低排序
results.sort(
    key=lambda item: item["similarity"],    #key：指定排序的依据，这里使用一个匿名函数 lambda 来提取每个岗位字典中的相似度值。
    reverse=True,       #reverse=True：指定排序顺序为降序，即从高到低排序。
)

#输出排名
for rank, result in enumerate(results, start=1):
    print(
        f"{rank}. " #rank：岗位排名，enumerate()：用于遍历可迭代对象并返回索引和值，start=1：指定索引从1开始。
        f"{result['company']} - "   #   result['company']：岗位所在公司，result['position']：岗位名称，result['similarity']：岗位相似度。
        f"{result['position']} - "
        f"{result['similarity']:.2%}"
    )