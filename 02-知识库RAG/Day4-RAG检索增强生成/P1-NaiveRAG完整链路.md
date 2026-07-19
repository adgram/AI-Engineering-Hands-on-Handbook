# P1：Naive RAG 完整链路 — 检索 → 增强 → 生成

## 目标
> **Day4 的整体定位**：Day3-P7 已经把检索和生成拼在了一起，但那只是一个"能跑"的原型。Day4 要做的是把这条链路**形式化、可评估、可优化**——定义标准三步曲、建立评估体系、引入高级检索和精排，最终形成一个"实验驱动优化"的闭环。

实现从检索到生成的完整 RAG 链路，理解每个环节的作用。

|          | 学习路径                                                     |
| -------- | ------------------------------------------------------------ |
| 已有基础 | Day3-P7（CLI 工具已有检索+生成雏形）+ Day1-P2（消息结构、System Prompt 设计）+ Day2-P1（上下文注入的位置策略） |
| 本章内容 | 把 Day3-P7 的工程原型升级为标准方法框架，并通过"纯 LLM vs RAG"对比实验让效果差异可量化，为 Day4 后续所有篇章提供统一基类。 |

## RAG 三步曲

RAG 将知识问答分解为三个标准阶段：先从知识库中检索相关文档，再将文档与问题组合成提示词，最后由 LLM 生成回答。这种"检索→增强→生成"的范式让模型能够基于外部知识作答，而非仅依赖其内部参数记忆。

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
# 前置：from common.rag_base import BaseRAG

class NaiveRAG(BaseRAG):
    """Naive RAG 三步流程"""
    def retrieve(self, query: str, k: int = 3) -> dict:
        return self.store.search(query, n_results=k)

    def augment(self, query: str, retrieved_docs: dict) -> list:
        context = "\n\n".join([
            f"[文档{i+1}] {doc}"
            for i, doc in enumerate(retrieved_docs['documents'][0])
        ])

        # Prompt 模板的多种注入策略详见 P2
        messages = [
            {"role": "system", "content": "你是一个知识库问答助手。请基于以下参考资料回答问题。如果资料不充分，请说'根据现有资料无法完整回答'，并说明缺少哪些信息。"},
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query}"}
        ]
        return messages

    def generate(self, messages: list) -> str:
        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=messages,
        )
        return response.choices[0].message.content

    def ask(self, query: str, k: int = 3) -> dict:
        retrieved = self.retrieve(query, k)
        messages = self.augment(query, retrieved)
        answer = self.generate(messages)

        return {
            "query": query,
            "answer": answer,
            "sources": retrieved['documents'][0],
            "source_metadatas": retrieved['metadatas'][0],
            "distances": retrieved['distances'][0]
        }
```

## Prompt 在 RAG 中的角色

RAG 的核心价值在于"用检索到的外部知识指导 LLM 生成回答"，而 Prompt 正是连接检索与生成的桥梁。具体来说：

- **它在哪里？** 在三步曲的第二步"增强（Augment）"中。检索到的文档不会自动影响模型输出，必须通过 Prompt 将文档注入到 LLM 的输入中，模型才能"看到"并"遵循"这些资料。
- **它起什么作用？** Prompt 承担三个职责：① 告诉 LLM 它是一个知识库问答助手（角色设定）；② 把检索到的多段文档拼接为结构化的上下文；③ 要求 LLM 基于资料回答、标注来源、处理检索失败等情况。
- **它的结构是什么？** 典型 RAG Prompt 分为三个层次：

```
System Prompt → 角色设定 + 行为规则（如"基于资料回答，不编造"）
    ↓
检索上下文  → 按 [文档1][文档2]... 排列的参考资料
    ↓
用户问题    → 原始查询
```

下面这个 `augment` 方法中的 Prompt 就是一个基础实例。后续 P2 会深入讨论多种注入策略和优化技巧。

## 实验：RAG 对比纯 LLM

以下实验使用同一个问题分别询问纯 LLM（无外部知识）和 RAG（带检索到的资料），直观展示 RAG 在知识时效性、幻觉控制和可追溯性上的优势。通过输出对比，可以清晰量化 RAG 带来的改善。

```python
# 知识库从 common/text_data/rag_knowledge 目录加载真实技术文档
rag = NaiveRAG(
    persist_dir=str(db_path),
    collection_name="my_knowledge_base",
)

if rag.store.count() == 0:
    loaded = load_directory(data_dir)
    rag.add_documents(
        documents=[d["content"] for d in loaded],
        metadatas=[{"source": "rag_knowledge", "topic": d.get("topic", "general")} for d in loaded],
        ids=[f"doc_{i+1}" for i in range(len(loaded))]
    )

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

# 输出检索来源
print("\n=== 检索到的资料 ===")
for i, (doc, meta) in enumerate(zip(result["sources"], result["source_metadatas"])):
    print(f"来源 {i+1} [{meta.get('topic', 'N/A')}]: {doc[:100]}...")
```

实验结果会写入 `P1_naive_rag_result.txt`，包含纯 LLM 回答、RAG 回答和检索到的完整资料。

## RAG vs 纯 LLM 对比

| 维度 | 纯 LLM | RAG |
|------|--------|-----|
| 知识时效性 | 训练数据截止日期前 | 可包含最新信息 |
| 幻觉控制 | 易产生幻觉 | 有资料约束，幻觉降低 |
| 可追溯性 | 黑盒，不知道依据 | 可追溯来源文档 |
| 领域知识 | 依赖训练数据 | 可注入私有知识 |
| 实施复杂度 | 低 | 中（需要向量库和检索） |

## RAG Prompt 设计要点

> 通用模板化技术（f-string/Jinja2/YAML）详见 Prompt D1-P6

### 1. 上下文放置位置

检索到的上下文放在问题之前还是之后，会影响 LLM 对信息的注意力分配。通常推荐"上下文在前、问题在后"，让模型先消化资料再聚焦问题，从而减少幻觉。

```
# 方案 A：上下文在前（推荐）
System: 基于以下参考资料回答问题。
User: 参考资料：...
      问题：...

# 方案 B：问题在前
User: 问题：...
      参考资料：...
```

### 2. 处理检索失败

当检索结果为空或相关性很低时：

```python
请回答以下问题。注意：当前没有找到相关的参考资料。
如果你知道答案，可以直接回答。如果不确定，请如实说不知道。

问题：{query}
```

### 3. 引用来源

在回答中标注来源编号可以增强可追溯性，让用户能验证每个结论对应的原始资料。通过在 Prompt 中要求模型按编号引用文档并标注多个来源，即可实现基本的引用机制。

```
基于以下参考资料回答问题。回答时请在相关句子后标注来源编号，如[1]。

参考资料：
[1] {doc1}
[2] {doc2}

问题：{query}

要求：
- 如果引用了资料，必须标注来源编号
- 没有参考资料支持的观点不要编造
- 综合多个来源的信息时标注多个编号，如[1][2]
```

## 动手实验

1. 准备 5 个问题和对应的期望答案
2. 用纯 LLM 和 RAG 分别回答，对比差异
3. 测试不同的 Prompt 模板对回答质量的影响
4. 测试检索结果数量 k 不同时（k=1,3,5,10）的效果

## 完成标准
- [ ] 实现并理解了 RAG 三步骤的完整流程
- [ ] 对比了 RAG 和纯 LLM 的回答差异
- [ ] 理解了 RAG Prompt 的设计要点
- [ ] 测试了不同 k 值的影响

## 下一步 → [P2-Prompt中注入上下文的最佳实践](P2-Prompt中注入上下文的最佳实践.md)