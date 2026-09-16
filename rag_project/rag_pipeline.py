from rag_project.config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TOP_K,
    DIFY_API_BASE_URL,
    DIFY_API_KEY,
    DIFY_TIMEOUT_SECONDS,
    DIFY_USER,
    DIFY_VERIFY_SSL,
    GENERATION_PROVIDER,
)
from rag_project.dify_client import DifyClient
from rag_project.generation import generate_answer
from rag_project.model_manager import (
    load_generation_components,
    load_retrieval_components,
)
from rag_project.retrieval import (
    load_knowledge_base,
    retrieve_chunks,
)


class RAGPipeline:
    """组合知识库、检索模型和生成模型。"""

    def __init__(self):
        print("正在初始化RAG系统……")

        (
            self.metadata,
            self.chunks,
            self.embeddings,
        ) = load_knowledge_base()

        (
            self.retrieval_tokenizer,
            self.retrieval_model,
        ) = load_retrieval_components(
            self.metadata["model_id"]
        )

        self.generation_provider = GENERATION_PROVIDER
        self.generation_tokenizer = None
        self.generation_model = None
        self.dify_client = None

        if self.generation_provider == "dify":
            self.dify_client = DifyClient(
                base_url=DIFY_API_BASE_URL,
                api_key=DIFY_API_KEY,
                user=DIFY_USER,
                timeout_seconds=DIFY_TIMEOUT_SECONDS,
                verify_ssl=DIFY_VERIFY_SSL,
            )
            print(f"生成后端：Dify（{DIFY_API_BASE_URL}）")
        else:
            (
                self.generation_tokenizer,
                self.generation_model,
            ) = load_generation_components()
            print("生成后端：本地 Qwen")

        print("RAG系统初始化完成")


    @staticmethod
    def build_rag_prompt(
        query: str,
        retrieved_chunks: list,
    ) -> str:
        """把已经通过门槛的资料与问题组合成提示词。"""

        context_parts = []

        for rank, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            context_parts.append(
                (
                    f"【资料{rank}】\n"
                    f"标题：{chunk['title']}\n"
                    f"正文：{chunk['text']}"
                )
            )

        context = "\n\n".join(context_parts)

        return (
            f"【参考资料】\n{context}\n\n"
            f"【用户问题】\n{query}\n\n"
            "【任务】\n"
            "根据参考资料直接回答问题。\n"
            "请覆盖问题中的主要概念和步骤。\n"
            "使用两到四个简洁要点。\n"
            "不要复述题目，不要输出思考过程。"
        )

    @staticmethod   #表示这个方法不依赖于类的实例属性或方法，可以直接通过类名调用，而不需要创建类的实例。
    def append_source_citations(
        answer: str,
        source_count: int,
    ) -> str:
        """由程序添加可靠的资料编号。"""

        cleaned_answer = answer.strip()

        if source_count <= 0:
            return cleaned_answer

        if (
            cleaned_answer
            .rstrip("。")
            .strip()
            == "现有资料不足"
        ):
            return cleaned_answer

        citations = " ".join(
            f"【资料{rank}】"
            for rank in range(
                1,
                source_count + 1,
            )
        )

        return (
            f"{cleaned_answer}\n\n"
            f"参考资料：{citations}"
        )

    def answer(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> dict:
        """执行检索、增强和生成。"""

        retrieval_result = retrieve_chunks(
            query=query,
            tokenizer=self.retrieval_tokenizer,
            model=self.retrieval_model,
            chunks=self.chunks,
            embeddings=self.embeddings,
            top_k=top_k,
            min_similarity=min_similarity,
        )

        # 没有通过门槛时提前结束，不调用生成模型。
        if not retrieval_result["passed"]:
            return {
                "query": query,
                "answer": "现有资料不足",
                "passed": False,
                "max_score": retrieval_result["max_score"],
                "sources": [],
            }

        retrieved_chunks = retrieval_result[
            "retrieved_chunks"
        ]

        context_chunks = retrieved_chunks

        rag_prompt = self.build_rag_prompt(
            query,
            context_chunks,
        )

        if self.generation_provider == "dify":
            answer = self.dify_client.generate(rag_prompt).answer
        else:
            answer = generate_answer(
                prompt=rag_prompt,
                tokenizer=self.generation_tokenizer,
                model=self.generation_model,
                max_new_tokens=max_new_tokens,
            )

        answer = self.append_source_citations(
            answer=answer,
            source_count=len(context_chunks),
        )

        sources = []

        for rank, chunk in enumerate(
            context_chunks,
            start=1,
        ):
            sources.append(
                {
                    "rank": rank,
                    "title": chunk["title"],
                    "chunk_id": chunk["chunk_id"],
                    "score": chunk["score"],
                }
            )

        return {
            "query": query,
            "answer": answer,
            "passed": True,
            "max_score": retrieval_result["max_score"],
            "sources": sources,
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()

    test_questions = [
        "RAG由哪三个阶段组成？",
        "世界上最高的山峰是什么？",
    ]

    for question in test_questions:
        result = pipeline.answer(question)

        print("\n" + "=" * 50)
        print("用户问题：", result["query"])
        print("是否通过门槛：", result["passed"])
        print("最高相似度：", result["max_score"])
        print("模型回答：", result["answer"])

        print("引用来源：")

        if result["sources"]:
            for source in result["sources"]:
                print(
                    f"【资料{source['rank']}】"
                    f"{source['title']}，"
                    f"文本块{source['chunk_id']}，"
                    f"相似度={source['score']:.4f}"
                )
        else:
            print("无")
