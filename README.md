# AI应用工程实战手册 — 学习计划
# AI Engineering Hands-on Handbook — Learning Plan

Prompt 工程 · 知识库 RAG · Agent 编排 — 从零到生产的学习路径
Prompt Engineering · RAG Knowledge Base · Agent Orchestration — A learning path from zero to production

## 目录结构 / Directory Structure

| 阶段 Phase | 内容 Content | 文档 Docs |
|------------|-------------|-----------|
| **一** | Prompt 工程基础 + 进阶 Prompt Engineering Basics + Advanced | `01-Prompt工程/` (14篇) |
| **二** | 知识库 RAG 完整实战 RAG Knowledge Base Full Practice | `02-知识库RAG/` (未实现 TBD) |
| **三** | Agent 编排从入门到生产 Agent Orchestration from Zero to Production | `03-Agent编排/` (未实现 TBD) |
| **四** | RAG + Agent 融合项目 RAG + Agent Integration Project | `04-融合项目/` (未实现 TBD) |

每篇文档包含：知识点讲解 + 可运行代码 + 动手实验 + 验收清单。
Each document includes: concept explanation + runnable code + hands-on experiments + checklist.

## 使用方式 / How to Use

### 安装依赖 / Install Dependencies

```bash
pip install openai python-dotenv chromadb langchain-text-splitters pyyaml jinja2 numpy dspy langgraph langchain-openai langchain-core langchain-community langchain-chroma fastapi pydantic streamlit requests sentence-transformers jieba diskcache transformers torch
```

### 配置 API Key / Configure API Key

创建 `.env` 文件（参考项目根目录的 `.env.example`）：
Create a `.env` file (refer to `.env.example` in the project root):

```
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
SILICONFLOW_API_KEY=sk-your-siliconflow-api-key
```

> `DEEPSEEK_API_KEY` 用于所有 LLM 调用（DeepSeek V4 Flash）。
> `DEEPSEEK_API_KEY` is used for all LLM calls (DeepSeek V4 Flash).
>
> `SILICONFLOW_API_KEY` 用于 Embedding 模型（RAG 章节依赖 SiliconFlow 的 `BAAI/bge-m3` 等嵌入模型）。
> `SILICONFLOW_API_KEY` is used for embedding models (RAG chapters depend on SiliconFlow's `BAAI/bge-m3` and other embedding models).

从第一天开始：打开 `01-Prompt工程/Day1-基础与核心框架/P1-调用LLM-API.md`
Start from Day 1: open `01-Prompt工程/Day1-基础与核心框架/P1-调用LLM-API.md`

## 许可证 / License

MIT
