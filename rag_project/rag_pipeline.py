from rag_project.config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TOP_K,
)
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

        (
            self.generation_tokenizer,
            self.generation_model,
        ) = load_generation_components()

        print("RAG系统初始化完成")

    @staticmethod
    def build_rag_prompt(
        query: str,
        retrieved_chunks: list,
    ) -> str:
        """把检索资料与用户问题组合成增强提示词。"""

        context_parts = []

        for rank, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            context_part = (
                f"【资料{rank}】\n"
                f"标题：{chunk['title']}\n"
                f"文本块编号：{chunk['chunk_id']}\n"
                f"正文：{chunk['text']}"
            )

            context_parts.append(context_part)

        context = "\n\n".join(context_parts)

        return (
            "请严格根据以下资料回答用户问题。\n"
            "不要使用资料之外的信息。\n"
            "回答中请使用【资料1】这样的格式引用来源。\n"
            "如果资料不足，请回答“现有资料不足”。\n\n"
            f"{context}\n\n"
            f"【用户问题】\n{query}"
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

        rag_prompt = self.build_rag_prompt(
            query,
            retrieved_chunks,
        )

        answer = generate_answer(
            prompt=rag_prompt,
            tokenizer=self.generation_tokenizer,
            model=self.generation_model,
            max_new_tokens=max_new_tokens,
        )

        sources = []

        for rank, chunk in enumerate(
            retrieved_chunks,
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