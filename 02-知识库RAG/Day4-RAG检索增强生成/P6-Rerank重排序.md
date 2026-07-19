# P6：Rerank 重排序

## 目标
理解 Rerank 的作用，实现两种 Rerank 方法。

|          | 学习路径                                                     |
| -------- | ------------------------------------------------------------ |
| 已有基础 | Day4-P5（多路召回返回 Top-N 候选）+ Day3-P1（Embedding 双塔架构的精度局限） |
| 本章内容 | 引入"粗筛（向量搜索）→ 精排（Cross-Encoder）"两阶段架构，用 Cross-Encoder 捕捉 Query-Doc 细粒度交互，提升 Top-K 排序质量。 |

## 为什么需要 Rerank？

向量搜索（粗筛）返回的结果中，高排名的文档不一定最相关，因为向量嵌入只能捕捉大致语义，缺乏细致的 pairwise 比较。

```
向量搜索返回 Top-30 → 已经比较接近了
但第 3 名可能比第 1 名更相关

原因：向量搜索是"快速初筛"，Rerank 是"精确打分"
```

## Rerank 的工作方式

Rerank 采用"先粗筛后精排"的两阶段策略：第一阶段用高效的向量搜索快速过滤出候选集，第二阶段用更精确的模型对候选集重新打分排序。

```
搜索阶段（快速）: 10000 条 → 粗筛到 30 条
Rerank 阶段（精确）: 30 条 → 精确排序 → Top-5
```

## 方法一：LLM Rerank（用 LLM 打分）

直接利用 LLM 对每个候选文档逐一打分，判断其与查询的相关性，虽然精度高但速度较慢，适合候选文档数量较少的场景。

```python
def llm_rerank(query: str, documents: list, llm, model: str = "deepseek-v4-flash") -> list:
    scored = []

    for doc in documents:
        prompt = f"""判断以下文档和问题的相关性。

问题：{query}
文档：{doc}

请输出 0-10 的相关性分数：
- 0-3: 不相关
- 4-6: 部分相关
- 7-8: 相关
- 9-10: 高度相关

只输出一个数字。"""

        resp = llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            score = float(resp.choices[0].message.content.strip())
        except:
            score = 5

        scored.append({"doc": doc, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

# 使用
# from common.rag_base import BaseRAG
# rag = BaseRAG(persist_dir="./chroma_db")

# initial_results = rag.store.search("RAG 相比微调有什么优势？", n_results=10)
# reranked = llm_rerank("RAG 相比微调有什么优势？", initial_results['documents'][0], rag.llm)

# print("Rerank 前:")
# for i, doc in enumerate(initial_results['documents'][0]):
#     print(f"  #{i+1}: {doc[:60]}...")

# print("\nRerank 后:")
# for i, item in enumerate(reranked):
#     print(f"  #{i+1} (分数={item['score']}): {item['doc'][:60]}...")
```

## 方法二：API Rerank（通过 API 调用专用 Rerank 模型）

Cross-Encoder 模型本身精度很高，但需要下载到本地。这里调用Rerank API，进行演示。

```python
import httpx, os

def api_rerank(query: str, documents: list, top_k: int = 5, model: str = "BAAI/bge-reranker-v2-m3") -> list:
    """通过 API 调用 Rerank 模型，无需本地下载"""
    api_key = os.getenv("SILICONFLOW_API_KEY")
    resp = httpx.post(
        "https://api.siliconflow.cn/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "query": query, "documents": documents, "top_n": top_k},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return [(documents[item["index"]], item["relevance_score"]) for item in data.get("results", [])]

# 使用
documents = [
    "RAG 是检索增强生成，结合检索和 LLM",
    "RAG 可以减少 LLM 的幻觉问题",
    "Prompt Engineering 是设计提示词的技术",
    "ChromaDB 是一个向量数据库",
    "Rerank 模型通过 Cross-Encoder 精确计算 query-doc 相关性",
]

reranked = api_rerank("RAG 的优势", documents, top_k=3)
for doc, score in reranked:
    print(f"  ({score:.4f}) {doc}")
```

## Rerank 效果评估

以下函数通过对比 Rerank 前后的 Hit Rate 和 MRR 指标，量化评估 Rerank 带来的排序质量提升。

```python
from common.rag_base import BaseRAG
from P3_retrieval_evaluation import SearchEvaluator  # 复用 P3 评估工具

def evaluate_rerank(store, test_cases: list, k_before=10, k_after=5):
    evaluator = SearchEvaluator(store)

    hr_before = evaluator.hit_rate(test_cases, k=k_before)
    mrr_before = evaluator.mrr(test_cases, k=k_before)

    hits_after = 0
    for query, expected_ids in test_cases:
        initial = store.search(query, n_results=k_before)
        if any(eid in initial['ids'][0][:k_after] for eid in expected_ids):
            hits_after += 1

    hr_after = hits_after / len(test_cases)

    return {
        "before": {"k": k_before, "hit_rate": hr_before, "mrr": mrr_before},
        "after": {"k": k_after, "hit_rate": hr_after},
        "improvement": hr_after - hr_before
    }
```

## Rerank 策略对比

不同 Rerank 方案在速度、效果和成本上各有优劣，下表总结了各方案的特性以帮助选择。

| 方案 | 速度 | 效果 | 成本 | 适用场景 |
|------|------|------|------|----------|
| LLM Rerank | 慢（每个文档一次调用） | 好 | 高（消耗 API） | 精度要求极高 |
| API Rerank（SiliconFlow） | 中等 | 很好 | 按量付费（有免费额度） | 不想下载模型 / 大多数场景 |
| Cross-Encoder（本地） | 快（秒级 100 条） | 很好 | 免费 | 可下载模型时推荐 |
| 不用 Rerank | 极快 | 一般 | 免费 | 对精度要求不高 |

## 动手实验

1. 用 Cross-Encoder Rerank 对 20 条文档做重排序
2. 对比 Rerank 前后的 Top-3 结果是否一致
3. 用 SearchEvaluator 量化评估 Rerank 的改进
4. 测试 Cross-Encoder 不同模型（bge-reranker-large vs bge-reranker-m3）的效果

## 完成标准
- [ ] 理解 Rerank 在 RAG 中的位置和作用
- [ ] 实现了一种 Rerank 方法（LLM 或 Cross-Encoder）
- [ ] 量化对比了 Rerank 前后的效果指标
- [ ] 能根据场景选择合适的 Rerank 方案

## 下一步 → [P7-带评估的RAG-Pipeline小项目](P7-带评估的RAG-Pipeline小项目.md)