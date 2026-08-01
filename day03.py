import json #JSON 是大模型开发中非常重要的数据格式。调用模型 API、保存配置、处理数据集时都会经常遇到。
skills = ["Python", "PyTorch", "RAG", "Docker"] 

with open("skills.txt", "w", encoding="utf-8") as file: #"skills.txt"：文件名称/ "w" 是写入模式，而且会清空并覆盖原内容 /encoding="utf-8"：使用 UTF-8 编码，避免中文乱码
    for skill in skills:
        file.write(skill + "\n")    #file.write()：向文件中写入内容     "\n"：换行 

print("技能已经写入 skills.txt")
#注意："w" 会覆盖文件原来的内容。
with open("skills.txt", "r", encoding="utf-8") as file: #with：代码执行完后自动关闭文件 #r:读取模式read

    content = file.read()#返回 str，也就是一个完整的字符串。
# 到这里，文件已经自动关闭
print("文件中的全部内容：")
print(content)

with open("skills.txt", "a", encoding="utf-8") as file:     #这里的 "a" 是追加模式，不会删除文件原来的内容。
    file.write("Transformer\n")
    file.write("Agent\n")

print("新技能追加完成")

print("逐行读取技能：")

with open("skills.txt", "r", encoding="utf-8") as file:
    for number, line in enumerate(file, start=1):   #for line in file：每次读取文件的一行。 enumerate(..., start=1)：同时得到编号和这一行内容。
        skill = line.strip()    #line.strip()：删除每行末尾的 \n。  strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。如果不用 line.strip()因为 line 本身带有 \n，而 print() 还会再换一次行，所以每项之间通常会多出一个空行
        print(f"{number}. {skill}") #f"{number}. {skill}"：把编号与技能组合起来。

job_data = {
    "company": "百度",
    "position": "大模型算法工程师",
    "skills": ["Python", "PyTorch", "RAG"],
    "salary": 25000,    #salary=25000：表示薪资，单位为人民币元。
    "is_campus": True,  #is_campus=True：表示是否为校招岗位，True 表示是校招岗位，False 表示不是校招岗位。
}

#将字典保存为 JSON 文件
with open("job.json", "w", encoding="utf-8") as file:
    json.dump(      #把 Python 数据写入 JSON 文件。
        job_data,    #job_data：要写入 JSON 文件的 Python 数据。
        file,   #file：要写入的文件对象。
        ensure_ascii=False,     #ensure_ascii=False：让中文正常显示。
        indent=4,   #使用四个空格缩进，使文件更易读。
    )

print("岗位数据已经保存到 job.json")

#从 JSON 文件读取数据
with open("job.json", "r", encoding="utf-8") as file:
    loaded_job = json.load(file)    #json.load(file) 会解析 JSON，并转换成对应的 Python 数据类型。

print("读取到的岗位数据：")
print(loaded_job)

print("公司：", loaded_job["company"])
print("岗位：", loaded_job["position"])
print("技能：", loaded_job["skills"])
print("数据类型：", type(loaded_job))#数据类型： <class 'dict'>字典

#异常处理
#程序读取不存在的文件时会报错并停止。try/except 可以捕获并处理错误，让程序继续运行。
print("开始测试异常处理")

try:
    with open("not_exists.json", "r", encoding="utf-8") as file:
        test_job = json.load(file)

    print("读取成功：", test_job)

except FileNotFoundError:
    print("读取失败：指定的文件不存在")

print("程序仍然可以继续执行")

#综合写成函数
def load_job(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            job = json.load(file)   #json.load(file) 会解析 JSON，并转换成对应的 Python 数据类型。

        return job  #返回读取到的岗位数据

    except FileNotFoundError:   #except FileNotFoundError: 捕获文件不存在的异常
        print(f"错误：找不到文件 {filename}")
        return None #失败时没有有效岗位数据，因此返回 None，表示“没有结果”;None 本身不是错误，它是一个专门表示“没有值”的对象。

    except json.JSONDecodeError:    #except json.JSONDecodeError: 捕获 JSON 解码错误的异常
        print(f"错误：{filename} 不是正确的 JSON 格式")
        return None
    
result1 = load_job("job.json")  
print("第一次读取结果：", result1)

result2 = load_job("abc.json")
print("第二次读取结果：", result2)

#Day 3 综合练习：岗位数据管理器
jobs = [
    {
        "company": "百度",
        "position": "大模型算法工程师",
        "skills": ["Python", "PyTorch", "RAG"],
    },
    {
        "company": "阿里巴巴",
        "position": "AI应用开发工程师",
        "skills": ["Python", "Docker", "Agent"],
    },
    {
        "company": "字节跳动",
        "position": "算法工程师",
        "skills": ["Python", "Transformer", "PyTorch"],
    },
]
#保存函数
def save_jobs(filename, jobs):
    try:
        with open(filename, "w", encoding="utf-8") as file: 
            json.dump(  #json.dump()：把 Python 数据写入 JSON 文件。
                jobs,
                file,
                ensure_ascii=False,
                indent=4,
            )

        return True

    except OSError as error:
        print("保存失败：", error)
        return False

#读取函数
def load_jobs(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError:
        print(f"读取失败：找不到 {filename}")
        return []

    except json.JSONDecodeError:
        print(f"读取失败：{filename} 的 JSON 格式错误")
        return []

#调用函数
save_success = save_jobs("jobs.json", jobs) #save_jobs("jobs.json", jobs)：调用 save_jobs 函数，将岗位数据保存到 jobs.json 文件中。返回值 save_success 表示保存是否成功。

if save_success:
    print("岗位数据保存成功")

loaded_jobs = load_jobs("jobs.json")    #load_jobs("jobs.json")：调用 load_jobs 函数，从 jobs.json 文件中读取岗位数据。返回值 loaded_jobs 是一个列表，包含所有读取到的岗位信息。

print(f"共读取到 {len(loaded_jobs)} 个岗位")

for number, job in enumerate(loaded_jobs, start=1):
    print(f"\n岗位 {number}")
    print("公司：", job["company"])
    print("岗位：", job["position"])
    print("技能：", job["skills"])

missing_jobs = load_jobs("missing_jobs.json")
print("不存在文件的读取结果：", missing_jobs)