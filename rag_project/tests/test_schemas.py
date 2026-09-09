import pytest
from pydantic import ValidationError

from rag_project.schemas import AskRequest


def test_ask_request_uses_defaults():   #test_ask_request_uses_defaults 函数用于测试 AskRequest 模型的默认值
    request = AskRequest(query="什么是 RAG？")

    assert request.query == "什么是 RAG？"
    assert request.top_k == 5
    assert request.min_similarity == 0.45
    assert request.max_new_tokens == 200


def test_query_strips_surrounding_whitespace(): #test_query_strips_surrounding_whitespace 函数用于测试 AskRequest 模型是否正确去除 query 字段的前后空白字符
    request = AskRequest(query="  RAG由哪三个阶段组成？ \n")

    assert request.query == "RAG由哪三个阶段组成？"


def test_query_rejects_whitespace_only():   #test_query_rejects_whitespace_only 函数用于测试 AskRequest 模型是否正确拒绝仅包含空白字符的 query 字段
    with pytest.raises(ValidationError):
        AskRequest(query="  \n\t")


@pytest.mark.parametrize(   #pytest.mark.parametrize 装饰器用于参数化测试函数 test_ask_request_accepts_boundary_values，传入不同的字段名和有效值组合，以测试 AskRequest 模型是否接受边界值。
    "field_name, valid_value",
    [
        ("top_k", 1),
        ("top_k", 10),
        ("min_similarity", -1.0),
        ("min_similarity", 1.0),
        ("max_new_tokens", 1),
        ("max_new_tokens", 512),
    ],
)
def test_ask_request_accepts_boundary_values(
    field_name,
    valid_value,
):
    request = AskRequest(
        query="测试问题",
        **{field_name: valid_value},    #**{field_name: valid_value} 用于动态地将字段名和有效值传递给 AskRequest 模型的构造函数，以测试模型是否接受边界值。
    )

    assert getattr(request, field_name) == valid_value  #getattr(request, field_name) 用于获取 request 对象中指定字段的值，并与 valid_value 进行比较，确保模型接受边界值。


@pytest.mark.parametrize(   #pytest.mark.parametrize 装饰器用于参数化测试函数 test_ask_request_rejects_out_of_range_values，传入不同的字段名和无效值组合，以测试 AskRequest 模型是否拒绝超出范围的值。
    "field_name, invalid_value",
    [
        ("top_k", 0),
        ("top_k", 11),
        ("min_similarity", -1.1),
        ("min_similarity", 1.1),
        ("max_new_tokens", 0),
        ("max_new_tokens", 513),
    ],
)
def test_ask_request_rejects_out_of_range_values(
    field_name,
    invalid_value,
):
    with pytest.raises(ValidationError):
        AskRequest(
            query="测试问题",
            **{field_name: invalid_value},
        )


def test_query_accepts_maximum_length():
    query = "问" * 500

    request = AskRequest(query=query)

    assert request.query == query


def test_query_rejects_more_than_maximum_length():
    query = "问" * 501

    with pytest.raises(ValidationError):
        AskRequest(query=query)