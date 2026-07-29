print("Hello,Python!")

#变量与基本类型
company = "字节跳动"
salary_min = 25
salary_max = 50
is_campus = True
publish_date = None

print(type(company))
print(type(salary_min))
print(type(salary_max))
print(type(is_campus))
print(type(publish_date))

name = "AI工程师"       # str
salary = 25000         # int
score = 0.85           # float
is_open = True         # bool
value = None           # NoneType

job_name: str = "大模型应用工程师"
salary: int = 25000
is_open: bool = True


#字符串
company = "百度"
position = "大模型算法工程师"

description = company + ":" + position
print(description)

description = f"{company}:{position}"
print(description)


jd = "Python, PyTorch, Transformer, RAG"

print(jd.lower())    #小写
print(jd.upper())    #大写
print(jd.split(","))   #按逗号分隔
print(jd.replace("RAG", "检索增强生成"))   #替换字符串 
print("Python" in jd)       #检查字符串是否在其中
print(jd.strip())  #去除两侧空格

skills = jd.split(",")

print(skills)
# ['Python', ' PyTorch', ' Transformer', ' RAG']


cleaned_skills = []

for skill in skills:
    cleaned_skills.append(skill.strip())

print(cleaned_skills)                              #清除每项两侧空格
cleaned_skills = [skill.strip() for skill in skills] #更紧凑的写法


#四种常用容器
#list:有顺序，可重复
skills = ["Python", "PyTorch", "Python"]    #列表，中括号

skills.append("RAG")    #添加元素
skills.remove("PyTorch")    #删除元素

print(skills)   #打印列表
print(skills[0])    #获取第一个元素
print(skills[-1])   #获取最后一个元素
print(len(skills))  #获取列表长度

#tuple:有顺序，不可变
salary_range = (20000, 35000)   #元组，小括号

minimum, maximum = salary_range #解包
print(minimum, maximum) #打印最小值和最大值

#dict：键值对应
job = {
    "company": "阿里巴巴",
    "position": "大模型应用工程师",
    "city": "杭州",
    "skills": ["Python", "PyTorch", "RAG"],
}

print(job["company"])   #job["salary"]   键不存在时抛出KeyError
print(job.get("salary"))  #使用get方法，键不存在时返回None
salary = job.get("salary", "面议")  #使用get方法，键不存在时返回默认值

#set：不重复、适合去重和集合运算
skills = {"Python", "PyTorch", "Python"}    # 重复的技能会被自动去除,花括号
print(skills)
my_skills = {"Python", "Git", "SQL"}    # 我掌握的技能
required_skills = {"Python", "PyTorch", "RAG"}  # 必需技能

matched = my_skills & required_skills   # 交集，已掌握的技能
missing = required_skills - my_skills   # 差集，待学习的技能

print("已掌握：", matched)
print("待学习：", missing)


#5. 条件判断
match_rate = 0.65

if match_rate >= 0.8:
    print("优先投递")
elif match_rate >= 0.5:
    print("可以尝试")
else:
    print("暂时不匹配")     #注意：条件后面需要冒号；代码块通过缩进表示；通常使用4个空格缩进。

#6. 循环
#遍历列表：
skills = ["Python", "PyTorch", "RAG"]

for skill in skills:        #遍历列表中的每个元素
    print(skill)

#同时获取序号：
for index, skill in enumerate(skills, start=1):     #enumerate()函数可以同时获取元素的索引和值，start=1表示索引从1开始计数
    print(index, skill)     #打印序号和技能名称

#遍历字典：
job = {
    "company": "百度",
    "position": "AI工程师",     #键值对
}

for key, value in job.items():  #items()方法返回一个包含字典中所有键值对的视图对象，每个键值对以元组的形式表示
    print(key, value)

#while循环：
count = 1   #初始化计数器

while count <= 3:   #当计数器小于等于3时，执行循环体
    print(count)    #打印计数器的值
    count += 1      #计数器加1，等价于count = count + 1

#练习1：清洗技能列表
raw_skills = " Python,PyTorch, RAG,python, Docker ,RAG "#原始技能字符串，包含重复项和多余空格
#输出不重复、无多余空格、全部小写的结果：['python', 'pytorch', 'rag', 'docker']
#要求：使用split()；使用strip()；使用循环或列表推导式；尽量保留原始出现顺序。提示：直接使用set可能破坏原始顺序。
print(raw_skills.split(","))
cleaned_skills = [raw_skills.strip().lower()for raw_skills in raw_skills.split(",")]
print(cleaned_skills)
skills = {raw_skills.strip().lower()for raw_skills in raw_skills.split(",")}    #set不能保证原始顺序
print(skills)

#答案：
raw_skills = " Python,PyTorch, RAG,python, Docker ,RAG "

# 第一步：拆分、去除空格、转换为小写
cleaned_skills = [
    skill.strip().lower()
    for skill in raw_skills.split(",")  #拆分字符串为列表，并去除每项的空格和转换为小写
]

print("清洗后：", cleaned_skills)

# 第二步：按照原始顺序去重
unique_skills = []

for skill in cleaned_skills:
    if skill not in unique_skills:
        unique_skills.append(skill)     #将不重复的技能添加到unique_skills列表中

print("去重后：", unique_skills)


#练习2：计算岗位匹配度
#给定：
my_skills = {"python", "git", "sql", "docker"}

required_skills = {
    "python",
    "pytorch",
    "transformer",
    "rag",
    "docker",
}
#计算已掌握技能
#缺失技能
#匹配率     匹配率公式：已掌握的岗位技能数量 / 岗位要求技能总数

#期望得到类似输出：
#已掌握：{'python', 'docker'}
#缺失：{'pytorch', 'transformer', 'rag'}
#匹配率：40.0%
matched = my_skills & required_skills   # 交集，已掌握的技能
missing = required_skills - my_skills   # 差集，待学习的技能
match_rate = len(matched) / len(required_skills) if required_skills else 0
print("已掌握：",matched)
print("缺失：",missing)
print(f"匹配率：{match_rate:.1%}")

#正确答案
my_skills = {"python", "git", "sql", "docker"}

required_skills = {
    "python",
    "pytorch",
    "transformer",
    "rag",
    "docker",
}

# 交集：自己已经掌握的岗位技能
matched = my_skills & required_skills

# 差集：岗位要求但自己缺少的技能
missing = required_skills - my_skills

# 匹配率
match_rate = len(matched) / len(required_skills)

print("已掌握：", matched)
print("缺失：", missing)
print(f"匹配率：{match_rate:.1%}")  # 保留一位小数的百分比格式

#练习3：统计多个JD的技能频率
#给定：
jobs = [
    {
        "company": "公司A",
        "skills": ["Python", "PyTorch", "RAG"],
    },
    {
        "company": "公司B",
        "skills": ["Python", "Docker", "Agent"],
    },
    {
        "company": "公司C",
        "skills": ["Python", "PyTorch", "Transformer"],
    },
]
#统计结果应该类似：
Python: 3
PyTorch: 2
RAG: 1
Docker: 1
Agent: 1
Transformer: 1
#进阶要求：按照出现次数从高到低输出
#排序提示：
sorted(
    skill_counts.items(),
    key=lambda item: item[1],
    reverse=True,
)
#答案：
jobs = [
    {
        "company": "公司A",
        "skills": ["Python", "PyTorch", "RAG"],
    },
    {
        "company": "公司B",
        "skills": ["Python", "Docker", "Agent"],
    },
    {
        "company": "公司C",
        "skills": ["Python", "PyTorch", "Transformer"],
    },
]

# 用字典保存：技能名称 -> 出现次数
skill_counts = {}

# 遍历每个岗位
for job in jobs:
    # 取出这个岗位的技能列表
    for skill in job["skills"]:
        # 如果技能不存在，get()返回0；然后加1
        skill_counts[skill] = skill_counts.get(skill, 0) + 1

print("统计结果：", skill_counts)

# 按出现次数从高到低排序
sorted_skills = sorted(
    skill_counts.items(),
    key=lambda item: item[1],
    reverse=True,
)

print("技能频率：")

for skill, count in sorted_skills:# 遍历排序后的技能列表
    print(f"{skill}: {count}")  # 打印技能名称和出现次数


# Python Day 1 completed