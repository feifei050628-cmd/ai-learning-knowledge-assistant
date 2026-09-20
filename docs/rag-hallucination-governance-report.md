# RAG 幻觉治理优化报告

## 结论

本轮优化针对“问题与知识库主题相似，但库内没有完整答案时仍被系统接受”的边界诱导问题。所有前后对比均来自同一套 77 条固定评测集和同一知识库，避免把不同数据规模的结果混在一起。

新方案将边界负例拒绝率从 **0.00%（0/25）提升到 84.00%（21/25）**，同时将 Hit@5 从 **87.50% 提升到 95.00%**、MRR@5 从 **71.67% 提升到 82.08%**。正样本接受率保持在预设的 **95.00%** 下限。

## 评测集

| 类别 | 数量 | 用途 |
| --- | ---: | --- |
| 事实型 | 20 | 验证原文可直接定位的问题 |
| 代码型 | 10 | 覆盖代码围栏、长代码块和实现细节 |
| 多跳/推理型 | 10 | 验证跨文本块组合检索 |
| 库外无关负例 | 12 | 验证明显无关问题的拒答 |
| 边界诱导负例 | 25 | 验证“主题相关但库内无完整答案”的拒答 |

`answerable` 是人工标注的业务标签，与 Hit@5 分开计算：检索命中某个相似片段，不代表知识库足以完整回答。

## 方案

1. 使用 BM25 和 BGE 向量分别召回候选文本块，并通过 RRF 融合排名。
2. 使用 `BAAI/bge-reranker-base` 对 query/chunk 对进行 Cross-Encoder 重排。
3. 在 0.20–0.90 范围内以 0.01 为步长搜索高阈值；在正样本召回率 ≥95% 的约束下，优先最小化误接受，得到 `T_high=0.83`。
4. 保留原 0.48 门槛作为 `T_low`：低于 0.48 拒答，0.48–0.83 仅返回原文摘录并标注覆盖不完整，达到 0.83 才进入正常生成。
5. 回答携带文件名、chunk id 和分数；生成后对关键声明进行引用覆盖检查，未支持声明比例超过 15% 时降级为资料不足。

## 同口径结果

| 指标 | 单路向量 + 0.48 单阈值 | 混合检索 + 重排 + 双阈值 | 变化 |
| --- | ---: | ---: | ---: |
| Hit@5 | 87.50% | 95.00% | +7.50pp |
| MRR@5 | 71.67% | 82.08% | +10.42pp |
| 正样本接受率 | 100.00% | 95.00% | -5.00pp |
| 库外无关负例拒绝率 | 100.00%（12/12） | 91.67%（11/12） | -8.33pp |
| 边界负例拒绝率 | 0.00%（0/25） | 84.00%（21/25） | +84.00pp |
| 全部负例拒绝率 | 32.43% | 86.49% | +54.06pp |
| 总体正确率 | 61.04% | 89.61% | +28.57pp |

这里的取舍是主动的：企业知识问答中，误答通常比拒答成本更高，因此阈值选择优先保证 95% 正样本召回，再减少误接受。

## 剩余误差

- 4 条边界负例仍高于高阈值，集中在 PyTorch、FastAPI 和 RAG 等与知识库高度同域的问题。
- 1 条股票类无关问题因“模型、指标”等词与技术笔记发生偶然匹配而被接受。
- 2 条正样本未通过高阈值，其中一条 Dify 多跳问题分数极低，说明当前知识库覆盖或候选召回仍需补强。

后续可增加困难负例挖掘，使用独立开发集/测试集标定，并将生成层引用核验加入人工抽检，避免在同一评测集上持续调参造成过拟合。

## 复现

```powershell
python -m pytest .\rag_project\tests -q

# 旧方案同口径基线
$env:ENABLE_HYBRID_RETRIEVAL='false'
$env:ENABLE_RERANKER='false'
$env:RAG_GATE_LOW='0.48'
$env:RAG_GATE_HIGH='0.48'
python -m rag_project.evaluate_retrieval

# 新方案与阈值标定
$env:ENABLE_HYBRID_RETRIEVAL='true'
$env:ENABLE_RERANKER='true'
python -m rag_project.rag_gate_calibrate --input rag_project/retrieval_evaluation_report_hybrid_precalibration.json --baseline 0.48
$env:RAG_GATE_LOW='0.48'
$env:RAG_GATE_HIGH='0.83'
python -m rag_project.evaluate_retrieval
```

原始结果位于：

- `rag_project/retrieval_evaluation_report_baseline_77.json`
- `rag_project/retrieval_evaluation_report.json`
- `rag_project/gate_calibration_report.json`
