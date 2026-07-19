# P1：主流 RAG 框架对比与RAG 前沿技术展望 — LangChain / LlamaIndex / Haystack

## 目标
了解三大 RAG 框架的特点，选择最适合你场景的工具，了解主流 RAG 框架、前沿研究方向及差距分析

|          | 学习路径                                                     |
| -------- | ------------------------------------------------------------ |
| 已有基础 | Day3-Day5 全部（手写实现的所有组件） |
| 本章内容 | 对比 LangChain/LlamaIndex/Haystack 三大框架与手写实现的取舍，展望 RAG 前沿方向（端到端化/长上下文/评估体系化/多模态+工具/幻觉检测），建立框架选型和趋势认知。 |

## 主流 RAG 框架对比

当下主流的五个 RAG 框架各有侧重，下表从定位、特点、RAG 复杂度、Agent 支持等维度进行横向对比，帮助你快速了解各框架的适用边界。

| 框架 | 定位 | 特点 | RAG复杂度 | Agent支持 | 学习曲线 | 社区 | 适用场景 |
|------|------|------|-----------|-----------|---------|------|---------|
| LangChain | 通用 LLM 编排 | 生态最丰富，组件化 | 中 | 强 | 中等 | 最大 | 快速原型 |
| LlamaIndex | 数据框架 | 索引能力最强，工具链完善 | 低（开箱即用） | 支持 | 简单 | 中 | 复杂索引 |
| Haystack | 生产级 NLP | 管道化设计，部署友好 | 中 | 有限 | 中等 | 中 | 企业级 |
| Ragas | RAG 评估 | 自动化指标评估 | — | — | — | 小 | 效果验证 |
| Canopy | Pinecone 官方 | 与 Pinecone 深度集成 | — | — | — | 小 | 生产级检索 |

## LangChain 实现 RAG

LangChain 通过组件化设计将 RAG 流程拆解为 Embedding、文档切分、向量库、LLM 和检索链五个步骤，以下代码展示了完整的实现流程。

```bash
pip install langchain langchain-community langchain-chroma
```

```python
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os

# 1. Embedding
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 2. 文档切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# 3. 向量库
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./langchain_chroma"
)

# 4. LLM
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com"
)

# 5. RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",  # stuff / map_reduce / refine / map_rerank
    return_source_documents=True
)

# 使用
result = qa_chain.invoke({"query": "RAG 是什么？"})
print(result['result'])
```

## LlamaIndex 实现 RAG

LlamaIndex 以数据索引为核心，通过简洁的 API 封装了文档加载、向量化、检索和生成的完整链路，以下代码仅需几行即可搭建 RAG 系统。

```bash
pip install llama-index llama-index-embeddings-huggingface
```

```python
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import os

# 配置
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.llm = OpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com"
)

# 创建索引（一行代码）
documents = [Document(text="RAG 是检索增强生成技术...")]
index = VectorStoreIndex.from_documents(documents)

# 查询
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("RAG 是什么？")
print(response)
```

## Haystack 实现 RAG

Haystack 采用管道化设计，将检索器、提示构建器和生成器串联为可编排的 Pipeline，以下代码演示了基于 BM25 检索的 RAG 实现。

```bash
pip install haystack-ai
```

```python
from haystack import Pipeline, Document
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
import os

# 文档存储
doc_store = InMemoryDocumentStore()
doc_store.write_documents([
    Document(content="RAG 是检索增强生成技术...")
])

# Pipeline
prompt_template = """基于以下资料回答问题：
{% for doc in documents %}
[{{ loop.index }}] {{ doc.content }}
{% endfor %}
问题：{{ query }}
回答："""

pipeline = Pipeline()
pipeline.add_component("retriever", InMemoryBM25Retriever(doc_store, top_k=3))
pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
pipeline.add_component("llm", OpenAIGenerator(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base_url="https://api.deepseek.com",
    model="deepseek-v4-flash"
))

pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder", "llm")

result = pipeline.run({"retriever": {"query": "RAG 是什么？"}})
print(result['llm']['replies'][0])
```

## 框架选择指南

不同场景适合不同的框架，下表根据你的经验水平和使用需求给出推荐选择及理由。

| 你的情况 | 推荐框架 | 原因 |
|---------|---------|------|
| 初学者 | LlamaIndex | 最简单，开箱即用 |
| 需要 Agent | LangChain + LangGraph | Agent 生态最完善 |
| 搜索场景 | Haystack | 搜索和检索最强 |
| 生产环境 | LangChain | 社区大，文档全 |
| 自定义需求 | 手写 + Chroma | 完全控制 | 

## RAG 研究前沿方向

RAG 领域仍在快速发展，以下几个方向值得持续关注，它们代表了从"检索辅助生成"到"检索与生成深度融合"的趋势。

### 1. 检索即生成的端到端化

端到端化 RAG 将检索过程融入生成模型内部，通过 Attention 或 KV Cache 实现检索与生成的深度耦合，代表了 RAG 的下一个演进方向。

```
传统 RAG：检索 → 拼接 → 生成
端到端 RAG：检索过程融合到生成过程的 Attention / KV Cache 中
```

代表工作：

- **REALM**：将检索器纳入 MLM 预训练
- **RETRO**（DeepMind）：基于 Chunked Cross-Attention 的检索增强
- **Atlas**（Meta）：少样本检索增强

### 2. 长上下文 RAG

随着 Gemini 1M / GPT-4-128K / DeepSeek 长上下文普及：

```
问题：已经有 128K 上下文窗口了，还需要 RAG 吗？
回答：需要。128K ≠ 128K 的有效利用率。
```

- **Lost in the Middle** 效应：长上下文中间部分被"遗忘"
- **RAG + Long Context**：先 RAG 精选 Top-K，再注入长上下文模型
- **Ring Attention / YaRN**：位置编码扩展

### 3. RAG 评估体系化

| 评估维度 | 指标                    | 工具                |
| -------- | ----------------------- | ------------------- |
| 检索质量 | Recall@K, MRR, NDCG     | Ragas               |
| 生成质量 | Faithfulness, Relevancy | Ragas, TruLens      |
| 端到端   | Answer Correctness      | 人工 + LLM-as-Judge |
| 延迟     | P50/P99 响应时间        | 自建                |
| 成本     | 每查询 Token 消耗       | 自建                |

```python
class RAGEvaluator:
    """RAG 系统自动评估"""
    
    def evaluate_retrieval(self, queries: list, ground_truths: list, retrieved: list):
        hits = 0
        for q, gt, ret in zip(queries, ground_truths, retrieved):
            if any(gt_item in ret for gt_item in gt):
                hits += 1
        recall = hits / len(queries)
        
        mrr = 0
        for q, gt, ret in zip(queries, ground_truths, retrieved):
            for rank, doc in enumerate(ret, 1):
                if any(gt_item in doc for gt_item in gt):
                    mrr += 1 / rank
                    break
        mrr /= len(queries)
        
        return {"Recall": recall, "MRR": mrr}
    
    def evaluate_generation(self, questions: list, answers: list, contexts: list, llm_judge):
        """使用 LLM 评估回答忠实度"""
        scores = []
        for q, a, ctx in zip(questions, answers, contexts):
            resp = llm_judge.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": f"""评估以下回答是否基于给定资料，没有编造。

资料：{ctx[:500]}

回答：{a}

输出 0-10 分（仅数字）："""}],
                temperature=0.0,
            )
            score = float(resp.choices[0].message.content.strip())
            scores.append(score)
        return {"avg_faithfulness": sum(scores) / len(scores), "scores": scores}

# 使用
# evaluator = RAGEvaluator()
# ret_metrics = evaluator.evaluate_retrieval(test_queries, test_ground_truths, retrieved_docs)
# gen_metrics = evaluator.evaluate_generation(test_queries, generated_answers, contexts, client)
```

### 4. 多模态 + 工具增强

RAG 的边界正在从纯文本检索扩展到图片、音频、视频等多模态数据，并融合代码执行和 API 调用等工具能力。

```
RAG 的边界正在扩展：
- 文本 → 文本+图片+表格+代码+音频+视频
- 纯检索 → 检索+代码执行+API调用+数据库查询
- 单轮 → 多轮对话+主动追问
```

### 5. 幻觉检测与缓解

幻觉是 RAG 系统面临的核心风险之一，下表对比了当前主流的幻觉检测与缓解方法及其效果。

| 方法                 | 原理                      | 效果 |
| -------------------- | ------------------------- | ---- |
| SelfCheckGPT         | 多次采样对比一致性        | 中   |
| RAGAS Faithfulness   | 原子事实分解验证          | 高   |
| CRAG                 | 检索质量评分 + 选择性增强 | 高   |
| Contrastive Decoding | 对比原始 vs 扰动模型输出  | 中   |

## 差距分析：RAG 工程化挑战

从研究到生产落地仍存在显著差距，下表从检索准召率、响应延迟、幻觉率、成本和可维护性五个维度分析了当前水平与理想态的差距。

| 挑战       | 当前水平                  | 理想态            | 关键路径                    |
| ---------- | ------------------------- | ----------------- | --------------------------- |
| 检索准召率 | 70-85%                    | 95%+              | 更好的 Embedding + 混合检索 |
| 响应延迟   | 1-5s                      | <500ms            | 缓存 + 检索重叠             |
| 幻觉率     | 5-15%                     | <1%               | 自反思 + 多轮验证           |
| 成本       | $0.01-0.1/query | <$0.001 | 小模型 + 投机推理 |                             |
| 可维护性   | 手动调参                  | 自适应            | AutoRAG + 在线学习          |

## 完成标准

- [ ] 至少用两个框架实现 RAG
- [ ] 理解各框架的特点和差异
- [ ] 能根据场景选择合适的框架
- [ ] 在自己项目中选用一个框架

## 下一步 → P2-缓存策略.md
