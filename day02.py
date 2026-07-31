# Python Day 2：函数、参数和返回值


def greet(name):    #def：告诉Python现在要定义一个函数；greet：函数名称；
    message = f"你好，{name}"   #name：函数参数；
    return message  #return会把函数计算结果交给调用者


result = greet("小明")  #return会把函数计算结果交给调用者;name是形参；"小明"是实参；greet("小明")是调用函数；调用函数就是写出函数名称，并在括号中传入实际数据;result接收返回值。

#第一个实验：观察执行顺序
print(result)   

print("1. 准备定义函数")


def greet(name):
    print("3. 函数开始执行")
    message = f"你好，{name}"
    return message


print("2. 函数定义完成")

result = greet("小明")

print("4. 函数返回结果")
print(result)
    #这说明：定义函数时不会执行函数体，调用函数时才会执行。

#第一个小练习
def introduce_job(company, position):
    message = f"{company}正在招聘{position}"# 使用f-string构造介绍文字
    return message# 返回介绍文字

job_description = introduce_job(
    "百度",
    "大模型算法工程师",
)

print(job_description)

#第二阶段：三种参数传递方式
def introduce_job(company, position, city="地点未填写"):
    message = f"{company}正在招聘{position}，工作地点：{city}"
    return message
#city="地点未填写"表示 city是默认参数。如果调用函数时没有传入城市，就使用默认值
#1. 位置参数
result1 = introduce_job(
    "百度",
    "大模型算法工程师",
    "北京",
)

print(result1)
#参数按照位置对应："百度"   → company
#参数按照位置对应："大模型算法工程师"   → position
#参数按照位置对应："北京"   → city

#2. 关键字参数
result2 = introduce_job(
    position="AI应用工程师",
    city="杭州",
    company="阿里巴巴",
)

print(result2)
#写出参数名称后，参数顺序可以改变。

#3. 使用默认参数
result3 = introduce_job(
    "公司A",
    "Python开发工程师",
)

print(result3)
#没有传入 city，因此使用默认值。默认参数是调用函数时没有提供对应实参，就使用函数定义中的默认值。


#print()与return实验
def add_with_print(a, b):
    print(a + b)


def add_with_return(a, b):
    return a + b


result_print = add_with_print(10, 20)   #add_with_print()只显示了30，没有返回结果，所以变量得到 None；
result_return = add_with_return(10, 20) #add_with_return()将30返回，变量可以继续使用它。

print("print函数的结果：", result_print)
print("return函数的结果：", result_return)
#还可以验证：
final_result = result_return * 2
print(final_result)

#第三阶段：返回多个结果
#函数可以一次返回多个数据：
def analyze_skills(my_skills, required_skills):
    matched = my_skills & required_skills
    missing = required_skills - my_skills

    return matched, missing
#调用时可以使用两个变量接收：
matched, missing = analyze_skills(
    {"python", "docker"},
    {"python", "pytorch", "rag", "docker"},
)

print("已掌握：", matched)
print("缺失：", missing)
#集合显示顺序不同是正常的。实际上：return matched, missing返回的是一个元组(类型是tuple)，相当于:return (matched, missing)

#变量作用域
company = "百度"    #company定义在函数外，是全局变量；


def build_job_title(position):      #position是函数参数，也是局部变量；
    prefix = "招聘岗位："       #prefix在函数内部定义，是局部变量；
    return f"{company}｜{prefix}{position}"     #result在函数外定义。


result = build_job_title("AI工程师")
print(result)
#如果在函数外执行：print(prefix)    会出现：NameError: name 'prefix' is not defined 因为局部变量只在函数执行期间有效。
#更推荐把需要的数据通过参数传入：
def build_job_title(company, position):
    prefix = "招聘岗位："
    return f"{company}｜{prefix}{position}"
result = build_job_title("百度", "AI工程师")
print(result)
#这样函数不依赖外部的全局变量，更容易复用和测试。

#类型注解与文档字符串
def analyze_skills(
    my_skills: set[str],
    required_skills: set[str],
) -> tuple[set[str], set[str]]:
    """计算已掌握技能和缺失技能。"""

    matched = my_skills & required_skills
    missing = required_skills - my_skills

    return matched, missing     

#第三阶段练习：岗位匹配函数
#请独立补全：
def analyze_match(
    my_skills: set[str],
    required_skills: set[str],
) -> tuple[set[str], set[str], float]:
    """计算已掌握技能、缺失技能和匹配率。"""

    matched = my_skills & required_skills
    missing = required_skills - my_skills

    if required_skills != set():        #也能写成if len(required_skills) != 0:
        match_rate = len(matched) / len(required_skills)
    else:
        match_rate = 0.0

    return matched, missing, match_rate
#要求：matched为两个集合的交集；missing为岗位要求中自己缺少的技能；匹配率为“已掌握岗位技能数 ÷ 岗位要求技能总数”当 required_skills为空时，匹配率设为 0.0，避免除以零；返回三个结果。
matched, missing, match_rate = analyze_match(
    {"python", "git", "docker"},
    {"python", "pytorch", "rag", "docker"},
)

print("已掌握：", matched)
print("缺失：", missing)
print(f"匹配率：{match_rate:.1%}")

matched, missing, match_rate = analyze_match(
    {"python"},
    set(),
)

print("空岗位要求的匹配率：", match_rate)

#练习一：技能清洗函数
#" Python,PyTorch, RAG,python, Docker ,RAG "转换为：["python", "pytorch", "rag", "docker"]
#函数框架：
def clean_skills(raw_skills: str) -> list[str]:     #
    """清洗技能字符串，并按照原始顺序去重。"""

    unique_skills = []  #
    seen_skills = set()

    for skill in raw_skills.split(","):
        cleaned_skill = skill.strip().lower()

        if cleaned_skill and cleaned_skill not in seen_skills:
            unique_skills.append(cleaned_skill)
            seen_skills.add(cleaned_skill)

    return unique_skills
#清洗单项技能使用：skill.strip().lower()
#同时满足两个条件：清洗后不是空字符串；该技能还没有出现在 seen_skills中。为什么需要判断空字符串？因为输入可能是："Python,,RAG,   ,Docker" 使用 split(",")后，其中可能出现：""或者只有空格的字符串。
#测试：
raw_skills = " Python,PyTorch, RAG,python, Docker ,RAG "

result = clean_skills(raw_skills)   #

print(result)
#再测试异常格式：
result = clean_skills(
    "Python,, RAG,   ,python,Docker"
)

print(result)


#练习二：技能频率统计函数
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

def count_skills(jobs: list[dict]) -> list[tuple[str, int]]:    #表示返回一个列表，列表中的每一项都是：("python", 3)
    """统计全部岗位的技能频率，并按次数从高到低排序。"""

    skill_counts = {}

    for job in jobs:    #外层循环从各个岗位每次去一个岗位
        for skill in job["skills"]:     #内层循环从岗位的技能列表每次去一个技能
            normalized_skill = skill.strip().lower()    
            skill_counts[normalized_skill] = skill_counts.get(normalized_skill, 0) + 1  

    sorted_skills = sorted(
        skill_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return sorted_skills
#统一技能格式:skill.strip().lower()这样 "Python"和 "python"会被统计为同一个技能。
#更新出现次数skill_counts.get(normalized_skill, 0) + 1
skill_frequency = count_skills(jobs)

for skill, count in skill_frequency:
    print(f"{skill}: {count}")

#练习小题4：多个返回值
def get_result():
    return "python", 3
#返回的是：("python", 3)  整体类型是：tuple也就是元组
#验证：
result = get_result()

print(result)
print(type(result))

#使用两个变量接收叫作“解包”：
skill, count = get_result()

print(skill)
print(count)
#要求变量数量与返回元素数量一致，否则会报错。