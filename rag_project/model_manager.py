import json

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from rag_project.config import (
    DEVICE,
    GENERATION_DTYPE,
    GENERATION_MODEL_ID,
    METADATA_PATH,
)


def load_retrieval_components(model_id: str):
    """加载用于生成查询向量的检索模型及其分词器。"""

    print(f"正在加载检索模型：{model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
    )

    model = AutoModel.from_pretrained(
        model_id,
    ).to(DEVICE)

    model.eval()

    return tokenizer, model


def load_generation_components():
    """加载用于生成自然语言回答的模型及其分词器。"""

    print(f"正在加载生成模型：{GENERATION_MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(
        GENERATION_MODEL_ID,
    )

    # 直接按指定数据类型加载，减少不必要的显存占用。
    model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL_ID,
        torch_dtype=GENERATION_DTYPE,
    ).to(DEVICE)

    model.eval()

    return tokenizer, model


if __name__ == "__main__":
    # 从知识库记录中读取创建向量时使用的模型名称。
    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    retrieval_model_id = metadata["model_id"]

    retrieval_tokenizer, retrieval_model = (
        load_retrieval_components(retrieval_model_id)
    )

    generation_tokenizer, generation_model = (
        load_generation_components()
    )

    print("\n===== 模型加载检查 =====")
    print("检索模型ID：", retrieval_model_id)
    print(
        "检索模型设备：",
        next(retrieval_model.parameters()).device,
    )
    print(
        "生成模型设备：",
        next(generation_model.parameters()).device,
    )
    print(
        "生成模型数据类型：",
        next(generation_model.parameters()).dtype,
    )