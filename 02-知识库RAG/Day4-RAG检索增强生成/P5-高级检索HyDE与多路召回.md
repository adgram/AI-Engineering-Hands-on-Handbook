# P5：高级检索 — HyDE 与多路召回

## 目标
掌握 HyDE（假设文档嵌入）和多路召回（Multi-Route Retrieval）两种高级检索技术。

|          | 学习路径                                                     |
| -------- | ------------------------------------------------------------ |
| 已有基础 | Day4-P1（NaiveRAG 单路向量检索）+ Day3-P5（RRF 融合算法）+ Day1-P4（CoT 思想） |
| 本章内容 | 用 HyDE（假设文档嵌入）弥补短查询/术语查询的语义不足，用多路召回+RRF 融合提升召回率，为 Rerank 提供更大候选池。 |

## 一、HyDE（Hypothetical Document Embedding）

HyDE（Hypothetical Document Embedding）的核心思路是：让 LLM 先根据用户问题生成一段"假设的理想答案"，再用这段假设答案去向量库中检索。这能有效弥补短查询和术语查询的语义信息不足问题。

### 核心思想

HyDE 的核心理念是"先假设，再检索"：让 LLM 先生成一段假设的理想文档，再将其向量化用于检索，从而弥合短查询与真实文档之间的语义鸿沟。

```
传统方法：问题 → 向量化 → 在知识库中搜索
HyDE：    问题 → 让 LLM 先生成一个"假设的理想答案"
          → 把假设答案向量化 → 用假设答案去搜索
          → 实际知识库中相似的文档
```

为什么有效？因为假设答案和真实文档在语义空间上更接近。

### 实现

以下代码展示了 HyDE 的完整实现流程：先用 LLM 生成假设文档，再用该文档替代原始查询进行向量检索。

```python
from common.rag_base import BaseRAG

def hyde_search(store, query: str, llm, model: str = "deepseek-v4-flash", k: int = 5) -> dict:
    """HyDE 检索"""
    # Step 1: 生成假设文档
    hyde_prompt = f"""基于你的知识，针对以下问题生成一段假设的、理想的 Wikipedia 风格的回答。

问题：{query}

要求：
- 使用客观、事实性的语言
- 假设回答内容在标准百科全书中
- 长度在 100-200 字之间"""

    response = llm.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": hyde_prompt}],
        temperature=0.3
    )
    hypothetical_doc = response.choices[0].message.content
    print(f"[HyDE] 假设文档:\n{hypothetical_doc[:150]}...\n")

    # Step 2: 用假设文档搜索
    results = store.search(hypothetical_doc, n_results=k)

    return results

# 对比普通搜索和 HyDE
# rag = BaseRAG(persist_dir="./chroma_db")
# query = "RAG 相比微调有什么优势？"

# print("=== 普通搜索 ===")
# normal_results = rag.store.search(query, n_results=3)
# for doc in normal_results['documents'][0]:
#     print(f"  - {doc[:80]}...")

# print("\n=== HyDE 搜索 ===")
# hyde_results = hyde_search(rag.store, query, rag.llm, k=3)
# for doc in hyde_results['documents'][0]:
#     print(f"  - {doc[:80]}...")
```

### HyDE 适用场景

不同查询类型下 HyDE 的效果差异明显，下表总结了 HyDE 的适用场景与预期提升效果。

| 场景 | 效果 | 原因 |
|------|------|------|
| 问题表述不清晰 | ✅ 提升明显 | 假设答案澄清了意图 |
| 专业术语查询 | ✅ 好 | 假设答案补充了术语解释 |
| 短查询（1-2 词） | ✅ 很好 | 短查询缺乏语义信息 |
| 长查询（完整句子） | ⚠️ 提升有限 | 原查询已包含足够信息 |

## 二、多路召回（Multi-Route Retrieval）

多路召回是指同时使用多种检索策略（如原始查询向量检索 + HyDE 检索 + 关键词检索等），将各路结果合并后通过 RRF 算法统一排序。不同策略的侧重点不同，组合使用可以互补优劣，显著提升召回率。

### 核心思想
同时使用多种检索策略，合并结果后统一排序。

```
同一问题
    ↓
┌─ 向量搜索 ─┐
├─ 关键词搜索 ─┤  → 合并结果 → Rerank → Top-k
├─ HyDE 搜索 ──┤
└─ 其他策略 ───┘
```

### 实现

MultiRouteRetriever 类整合了向量搜索、关键词搜索和 HyDE 搜索三种路由，并通过 RRF 算法对各路结果进行融合排序。

```python
from collections import defaultdict

class MultiRouteRetriever:
    def __init__(self, store, llm, model: str = "deepseek-v4-flash"):
        self.store = store
        self.llm = llm
        self.model = model

    def vector_search(self, query: str, k: int = 5) -> list:
        results = self.store.search(query, n_results=k)
        return list(zip(results['ids'][0], results['documents'][0], results['distances'][0]))

    # 注：以下实现为纯向量搜索（通过扩大候选集模拟不同"路由"）
    # 真正的关键词搜索需接入 BM25 等全文检索引擎
    def keyword_search(self, query: str, k: int = 5) -> list:
        results = self.store.search(query, n_results=k * 2)
        return list(zip(results['ids'][0], results['documents'][0], results['distances'][0]))

    def hyde_search(self, query: str, k: int = 5) -> list:
        resp = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"针对问题生成一段百科式的回答：{query}"}],
        )
        hyde_doc = resp.choices[0].message.content
        results = self.store.search(hyde_doc, n_results=k)
        return list(zip(results['ids'][0], results['documents'][0], results['distances'][0]))

    def multi_route_search(self, query: str, k: int = 5) -> dict:
        routes = {
            "vector": self.vector_search(query, k),
            "hyde": self.hyde_search(query, k),
        }

        K = 60
        scores = defaultdict(float)
        route_details = {}

        for route_name, results in routes.items():
            route_details[route_name] = []
            for rank, (doc_id, doc, dist) in enumerate(results, 1):
                scores[doc_id] += 1 / (K + rank)
                route_details[route_name].append({"id": doc_id, "doc": doc[:80], "rank": rank})

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        final_results = []
        for doc_id, score in ranked:
            doc = self.store.collection.get(ids=[doc_id])
            final_results.append({
                "id": doc_id,
                "content": doc['documents'][0] if doc['documents'] else "",
                "score": score
            })

        return {
            "query": query,
            "routes": route_details,
            "results": final_results
        }

# 使用
# rag = BaseRAG(persist_dir="./chroma_db")
# retriever = MultiRouteRetriever(rag.store, rag.llm)
# result = retriever.multi_route_search("什么是 RAG？", k=3)

# print("各路由结果:")
# for route, docs in result["routes"].items():
#     print(f"\n  [{route}]")
#     for d in docs:
#         print(f"    排名{d['rank']}: {d['doc']}")

# print("\n融合后最终结果:")
# for r in result["results"]:
#     print(f"  [{r['id']}] 得分={r['score']:.4f}: {r['content'][:80]}...")
```

## 工业实践：混合检索的动态权重调整

> 注：以下展示的是**模拟混合检索**的权重融合思路（两侧均为向量搜索），真正的混合检索需接入 BM25 等全文检索引擎。

动态权重核心思路——根据查询类型调整语义搜索与关键词搜索的权重比：

```python
# 查询分类 → 动态权重
weights = {
    "semantic": (0.8, 0.2),  # 语义查询侧重向量搜索
    "keyword": (0.3, 0.7),   # 关键词查询侧重精确匹配
    "hybrid": (0.6, 0.4),    # 混合查询均衡
}

# 两侧各自搜索后按权重融合评分
# sem_score = (1 - distance) * sem_weight
# kw_score = (1 - distance) * kw_weight
# final_score = sem_score + kw_score
```

此外，字节跳动还优化了 BM25 算法（ByteBM25）：加入**词频饱和机制**（词出现超10次不再线性增长）和**领域词权重调整**（领域词 IDF 提升 1.5-2 倍）。

```python
class ByteBM25:
    def __init__(self, domain_terms: set = None):
        self.domain_terms = domain_terms or set()

    def score(self, term: str, term_freq: int, doc_len: int, avg_doc_len: float) -> float:
        k1 = 1.2
        tf_saturated = term_freq / (term_freq + k1 * (1 - 0.75 + 0.75 * doc_len / avg_doc_len))
        boost = 2.0 if term in self.domain_terms else 1.0
        return tf_saturated * boost
```

## 动手实验

1. 在同一个查询上对比普通搜索和 HyDE 搜索的结果差异
2. 实现多路召回（至少 3 路），用 RRF 融合
3. 对比单路搜索和多路召回在 Hit Rate 上的差异
4. 分析哪些类型的查询 HyDE 提升最明显
5. 实现动态权重混合检索，对比固定权重与动态权重的效果

## 完成标准
- [ ] 理解 HyDE 的原理并成功实现
- [ ] 实现了一个多路召回 + RRF 融合系统
- [ ] 对比了不同检索策略的效果差异
- [ ] 能判断何种场景适合使用 HyDE
- [ ] 了解动态权重混合检索和 ByteBM25 的工业实践

## 下一步 → [P6-Rerank重排序](P6-Rerank重排序.md)