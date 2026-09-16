import type { DemoDocument } from "./types";

// MOCK: 当前 FastAPI 未提供文档列表、上传、删除、重新解析或会话历史接口。
// 下列数据只用于前端交互演示，不会写入真实知识库。
export const demoDocuments: DemoDocument[] = [
  { id: "doc-rag", name: "RAG 实践指南.md", type: "MD", size: "18.4 KB", updatedAt: "今天 09:42", status: "ready", chunks: 36 },
  { id: "doc-transformer", name: "Transformer 原理.md", type: "MD", size: "24.1 KB", updatedAt: "昨天 18:20", status: "ready", chunks: 48 },
  { id: "doc-pytorch", name: "PyTorch 学习笔记.txt", type: "TXT", size: "91.7 KB", updatedAt: "9 月 8 日", status: "ready", chunks: 127 },
  { id: "doc-eval", name: "大模型评估方法.pdf", type: "PDF", size: "2.8 MB", updatedAt: "刚刚", status: "processing", chunks: 0 },
  { id: "doc-agent", name: "Agent 工程化清单.pdf", type: "PDF", size: "4.2 MB", updatedAt: "9 月 6 日", status: "failed", chunks: 0 },
];

export const demoHistory = [
  "RAG 的完整处理流程",
  "如何校准相关性门槛",
  "Transformer 的注意力机制",
  "生成质量评估指标",
];
