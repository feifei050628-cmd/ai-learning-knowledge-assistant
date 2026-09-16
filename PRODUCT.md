# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

现有 FastAPI 应用内置静态 HTML、CSS 和原生 JavaScript；不引入新的前端框架。

## Users

主要用户是需要在本地浏览器中查询个人中文学习知识库的使用者；项目维护者也会把仓库链接放入简历，供面试官查看实现与使用方式。

## Product Purpose

让用户通过聊天界面向本地知识库提问，快速获得中文回答，并核对回答所依据的资料来源。成功意味着用户无需打开 Swagger 或编写命令即可完成一次可追溯的问答。

## Positioning

系统在本地完成中文语义检索、相关性门控和答案生成，并把检索分数与来源一起返回，而不是只展示无法核验的聊天文本。

## Operating Context

用户先在本机启动 FastAPI 服务，再通过浏览器使用聊天页面。知识库由项目资料和每日学习笔记构建；回答可能成功生成，也可能因资料不足而明确拒答。

## Capabilities and Constraints

- 现有 `/ask` 接口接收问题、Top-K、相似度门槛和最大生成长度。
- 返回内容包含回答、是否通过门槛、最高相似度与来源列表。
- 模型和知识库在本机加载，首次启动可能需要等待。
- 当前阶段优先快速完成单页聊天体验，不增加登录、历史云同步或新的前端依赖。
- 保持已有 API 行为和自动化测试兼容。

## Brand Commitments

产品名称为“本地中文 RAG 问答系统”。界面文案使用清楚、克制的中文，避免营销式承诺。

## Evidence on Hand

- `rag_project/api.py`：FastAPI 接口与模型生命周期。
- `rag_project/schemas.py`：请求和响应数据结构。
- `rag_project/knowledge_base/`：本地知识库元数据与向量文件。
- `rag_project/tests/`：现有 API 自动化测试。

项目没有现成品牌资产、用户评价或线上性能数据，后续页面不得虚构这些内容。

## Product Principles

- 问答任务始终位于首屏中心。
- 回答必须和引用来源一起呈现，便于核验。
- 资料不足时明确拒答，不伪造答案。
- 保持本地部署简单，避免非必要依赖。
- 状态、错误和等待过程都应对用户可见且可理解。

## Accessibility & Inclusion

页面需要支持键盘操作、清晰焦点、语义化状态提示、移动端布局以及减少动画偏好。
