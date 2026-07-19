# 知识库RAG | P5：Naive RAG 完整链路

## 前言

本节实现最基础的 **Naive RAG**——"检索→增强→生成"三步曲，并讨论如何把检索到的文档注入 Prompt 才能让模型用好。这是所有高级 RAG 架构的起点。

## RAG 三步曲

RAG 将知识问答分解为三个标准阶段：先从知识库中检索相关文档，再将文档与问题组合成提示词，最后由 LLM 生成回答。

```
用户问题
    ↓
① 检索（Retrieve）：从知识库中找到相关文档片段
    ↓
② 增强（Augment）：将检索结果 + 问题组合成 Prompt
    ↓
③ 生成（Generate）：LLM 根据增强后的 Prompt 生成回答
```

```python
class NaiveRAG(BaseRAG):
    """Naive RAG 三步流程"""
    def retrieve(self, query: str, k: int = 3) -> dict:
        return self.store.search(query, n_results=k)

    def augment(self, query: str, retrieved_docs: dict) -> list:
        context = "\n\n".join([
            f"[文档{i+1}] {doc}"
            for i, doc in enumerate(retrieved_docs['documents'][0])
        ])
        messages = [
            {"role": "system", "content": "你是一个知识库问答助手。请基于以下参考资料回答问题。如果资料不充分，请说'根据现有资料无法完整回答'。"},
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query}"}
        ]
        return messages

    def generate(self, messages: list) -> str:
        response = self.llm.chat.completions.create(model=self.llm_model, messages=messages)
        return response.choices[0].message.content

    def ask(self, query: str, k: int = 3) -> dict:
        retrieved = self.retrieve(query, k)
        messages = self.augment(query, retrieved)
        answer = self.generate(messages)
        return {"query": query, "answer": answer, "sources": retrieved['documents'][0]}
```

## Prompt 在 RAG 中的角色

检索到的文档不会自动影响模型输出，必须通过 Prompt 注入到 LLM 的输入中。典型 RAG Prompt 分三个层次：

```
System Prompt → 角色设定 + 行为规则（如"基于资料回答，不编造"）
    ↓
检索上下文  → 按 [文档1][文档2]... 排列的参考资料
    ↓
用户问题    → 原始查询
```

Prompt 承担三个职责：① 告诉 LLM 它是知识库问答助手（角色设定）；② 把检索到的多段文档拼接为结构化上下文；③ 要求 LLM 基于资料回答、标注来源、处理检索失败。

## RAG 对比纯 LLM

用同一个问题分别询问纯 LLM（无外部知识）和 RAG（带检索资料），直观展示差异：

```python
rag = NaiveRAG(persist_dir=str(db_path), collection_name="my_knowledge_base")

query = "RAG 相比微调有什么优势？"

# 纯 LLM 回答（无外部知识）
response = rag.llm.chat.completions.create(
    model=rag.llm_model,
    messages=[{"role": "user", "content": query}]
)
print("=== 纯 LLM 回答 ===")
print(response.choices[0].message.content)

# RAG 回答（带检索资料）
print("\n=== RAG 回答 ===")
result = rag.ask(query, k=3)
print(result["answer"])

print("\n=== 检索到的资料 ===")
for i, (doc, meta) in enumerate(zip(result["sources"], result["source_metadatas"])):
    print(f"来源 {i+1} [{meta.get('topic', 'N/A')}]: {doc[:100]}...")
```

| 维度 | 纯 LLM | RAG |
|------|--------|-----|
| 知识时效性 | 训练数据截止日期前 | 可包含最新信息 |
| 幻觉控制 | 易产生幻觉 | 有资料约束，幻觉降低 |
| 可追溯性 | 黑盒，不知道依据 | 可追溯来源文档 |
| 领域知识 | 依赖训练数据 | 可注入私有知识 |
| 实施复杂度 | 低 | 中（需要向量库和检索） |

## 上下文注入策略

当检索结果包含多篇文档时，全部塞入 Prompt 会迅速填满上下文窗口，且 AI 只喜欢关注首尾内容（Lost in the Middle）。因此要解决"选择哪些文档"和"如何排列它们"。

### 策略 1：按相关性排列（最常用）

将文档按相关性分数从高到低排序，把最相关的放在开头和结尾（U 型曲线的两端）：

```python
def sort_by_relevance(docs, distances):
    paired = sorted(zip(docs, distances), key=lambda x: x[1])
    return [p[0] for p in paired]
```

### 策略 2：分而治之（Map-Reduce 模式）

当文档太多时，先分别回答再汇总——每个文档独立回答，最后综合所有答案：

```python
def map_reduce_rag(docs: list, query: str, llm, model: str = "deepseek-v4-flash") -> str:
    # Map 阶段：每个文档独立回答
    partial_answers = []
    for doc in docs:
        resp = llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"基于以下资料回答问题。如果不包含答案，返回'无相关信息'。\n\n资料：{doc}\n\n问题：{query}"}],
        )
        partial_answers.append(resp.choices[0].message.content)

    # Reduce 阶段：汇总
    all_answers = "\n".join([f"[资料{i+1}] {a}" for i, a in enumerate(partial_answers)])
    final_resp = llm.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"以下是多个独立回答，请综合生成完整一致的最终回答：\n\n{all_answers}\n\n问题：{query}"}],
    )
    return final_resp.choices[0].message.content
```

### 策略选择指南

| 情景 | 推荐策略 | 原因 |
|------|---------|------|
| 文档少（≤3）且短 | 直接拼接 | 简单有效 |
| 文档多（≥5） | 按相关性排列 | 控制质量，利用 U 型曲线 |
| 文档长 | 先压缩再拼接 | 节省 Token |
| 文档矛盾 | Map-Reduce | 发现矛盾并综合 |
| 实时要求高 | 直接拼接 | 速度优先 |

## Context Tracker：避免重复读取

在多轮检索中追踪已读文档，避免重复读取和 Token 浪费：

```python
import hashlib

class ContextTracker:
    def __init__(self):
        self.read_chunks = set()
        self.read_content = set()

    def has_read(self, chunk_id: str) -> bool:
        return chunk_id in self.read_chunks

    def mark_read(self, chunk_id: str, content: str):
        self.read_chunks.add(chunk_id)
        self.read_content.add(hashlib.md5(content.encode()).hexdigest())

    def filter_new(self, chunks: list) -> list:
        new = []
        for c in chunks:
            cid = c.get("id", "")
            if cid not in self.read_chunks:
                new.append(c)
                self.mark_read(cid, c.get("content", ""))
        return new
```

多轮对话中，已读过的内容不再注入 Prompt，能显著节省 Token。

## RAG Prompt 设计要点

1. **上下文放置位置**：推荐"上下文在前、问题在后"，让模型先消化资料再聚焦问题
2. **处理检索失败**：检索为空时明确告诉模型"没有找到相关资料"，让它如实说明而非编造
3. **引用来源**：在 Prompt 中要求模型按编号引用文档（如 `[1]`），增强可追溯性
