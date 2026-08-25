import torch

from rag_project.config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEVICE,
)


SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。"
    "你只能根据用户提供的资料回答问题。"
    "不要编造资料之外的信息。"
    "如果资料不足，必须回答：现有资料不足。"
)


def generate_answer(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """根据增强提示词生成自然语言回答。"""

    if not prompt.strip():
        raise ValueError("提示词不能为空")

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens必须大于0")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # 按照当前模型规定的对话格式组合system和user消息。
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    model_inputs = {
        name: tensor.to(DEVICE)
        for name, tensor in model_inputs.items()
    }

    # generate()返回“原始输入Token + 新生成Token”，所以先记录输入长度。
    input_length = model_inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_token_ids = output_ids[
        0,
        input_length:,
    ]

    answer = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True,
    ).strip()

    return answer


if __name__ == "__main__":
    from rag_project.model_manager import (
        load_generation_components,
    )

    tokenizer, model = load_generation_components()

    test_prompt = """
【资料1】
RAG由Retrieval、Augmentation和Generation三个阶段组成。
Retrieval负责检索资料，Augmentation负责把资料加入提示词，
Generation负责根据资料生成回答。

【用户问题】
RAG由哪三个阶段组成？

请只根据以上资料回答，并引用资料编号。
"""

    answer = generate_answer(
        prompt=test_prompt,
        tokenizer=tokenizer,
        model=model,
    )

    print("\n===== 生成模块测试 =====")
    print("模型回答：")
    print(answer)