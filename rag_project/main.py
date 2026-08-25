from rag_project.config import PROJECT_NAME
from rag_project.rag_pipeline import RAGPipeline


def print_result(result: dict):
    """在终端展示一次RAG回答结果。"""

    print("\n" + "=" * 60)
    print("用户问题：", result["query"])
    print("模型回答：")
    print(result["answer"])

    print("\n检索信息：")
    print("是否通过门槛：", result["passed"])
    print(
        "最高相似度：",
        f"{result['max_score']:.4f}",
    )

    print("\n引用来源：")

    if not result["sources"]:
        print("无")
        return

    for source in result["sources"]:
        print(
            f"【资料{source['rank']}】"
            f"{source['title']}，"
            f"文本块{source['chunk_id']}，"
            f"相似度={source['score']:.4f}"
        )


def main():
    """启动命令行RAG问答程序。"""

    print("=" * 60)
    print(PROJECT_NAME)
    print("输入问题后按Enter开始提问。")
    print("输入 exit、quit 或 退出 可以结束程序。")
    print("=" * 60)

    # 在进入循环前初始化，保证所有问题复用同一套模型。
    pipeline = RAGPipeline()

    while True:
        try:
            query = input("\n请输入问题：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n程序已结束。")
            break

        if query.lower() in {
            "exit",
            "quit",
            "退出",
        }:
            print("程序已结束。")
            break

        if not query:
            print("问题不能为空，请重新输入。")
            continue

        try:
            result = pipeline.answer(query)
            print_result(result)
        except Exception as error:
            # 单次问答失败时保留程序，用户仍然可以继续提问。
            print("\n本次问答失败：")
            print(type(error).__name__, error)


if __name__ == "__main__":
    main()