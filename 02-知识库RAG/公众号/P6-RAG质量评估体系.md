# 知识库RAG | P6：RAG 质量评估体系

## 前言

本节建立 RAG 的量化评估体系，分两部分：**检索质量评估**（检得准不准）和**生成质量评估**（答得好不好）。检索和生成是 RAG 的两条腿，必须分别评估才能定位问题出在哪。

## 检索质量评估

检索是 RAG 的第一步，检索质量直接决定回答质量上限。三大核心指标：

### Hit Rate（命中率）

期望文档是否出现在 Top-K 结果中。公式：`命中查询数 / 总查询数`。回答的是"能不能找得到"。

### MRR（平均倒数排名）

第一个正确答案的排名位置倒数取平均。公式：`平均(1/r)`——排第 1 得 1.0，排第 3 得 0.33，排第 10 得 0.1。回答的是"找到的排得够不够靠前"。

### NDCG（归一化折损累计增益）

考虑多级相关性的排序质量，是最严格的指标：

```
DCG = Σ (2^rel - 1) / log₂(i+1)    # rel 是相关度等级，i 是排名位置
NDCG = DCG / IDCG                   # 归一化到 [0, 1]
```

排名越靠前且相关度越高，NDCG 越大。它比 MRR 更精细——MRR 只看第一个正确结果，NDCG 看全部结果的排序质量。

```python
class SearchEvaluator:
    """检索质量评估器——三指标 + 多 k 值批量评估"""
    def evaluate(self, collection, test_cases, k_values=[1, 3, 5, 10]):
        results = {}
        for k in k_values:
            hit_rate, mrr = 0, 0
            for query, expected_ids in test_cases:
                res = collection.query(query_texts=[query], n_results=k)
                retrieved_ids = res['ids'][0]
                if any(eid in retrieved_ids for eid in expected_ids):
                    hit_rate += 1
                for rank, rid in enumerate(retrieved_ids, 1):
                    if rid in expected_ids:
                        mrr += 1.0 / rank
                        break
            n = len(test_cases)
            results[k] = {"hit_rate": hit_rate / n, "mrr": mrr / n}
        return results

    def find_best_params(self, collection, test_cases, param_grid):
        """网格搜索找最优 chunk_size / overlap / k 组合"""
        best, best_score = None, 0
        for chunk_size in param_grid["chunk_size"]:
            for overlap in param_grid["overlap"]:
                for k in param_grid["k"]:
                    score = self.evaluate(collection, test_cases, [k])[k]["mrr"]
                    if score > best_score:
                        best, best_score = (chunk_size, overlap, k), score
        return best, best_score
```

**评估要点**：构建 50-100 条黄金测试集（`query → 期望结果`）；多 k 值评估（1/3/5/10 看不同召回深度的表现）；每次改动做基线对比。

## 生成质量评估

检索到的文档好，不代表回答好——LLM 可能曲解资料、编造内容或答非所问。需要用 LLM 自动评估生成质量。

### 四个评估维度

| 维度 | 含义 | 评估什么 |
|------|------|---------|
| **Faithfulness（忠实度）** | 回答是否基于检索资料 | 有没有编造（幻觉） |
| **Relevance（相关性）** | 回答是否切中问题 | 有没有答非所问 |
| **Completeness（完整性）** | 是否覆盖了资料中的关键信息 | 有没有遗漏 |
| **Conciseness（简洁性）** | 回答是否简洁不啰嗦 | 有没有废话 |

### LLM-as-Judge 实现

通常用 LLM 对 Faithfulness/Relevance 打 0-10 分，通过 `response_format` 强制 JSON 输出便于程序解析：

```python
class RAGEvaluator:
    """用 LLM 自动评估生成质量"""
    def evaluate_answer(self, query, answer, contexts, llm, model="deepseek-v4-flash"):
        prompt = f"""请评估以下 RAG 回答的质量，返回 JSON：
{{
  "faithfulness": 0-10的分数（回答是否忠于资料）,
  "relevance": 0-10的分数（回答是否切中问题）,
  "is_hallucination": true/false（是否有编造内容）,
  "hallucinated_parts": ["编造的具体内容"]
}}

问题：{query}
检索资料：{contexts}
回答：{answer}"""
        resp = llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}  # 强制 JSON 输出
        )
        return json.loads(resp.choices[0].message.content)
```

### RAGAS 框架

RAGAS 框架是专用 RAG 评估库，内置四个标准化指标：

- `faithfulness`：忠实度，回答是否基于上下文
- `answer_relevancy`：回答相关性
- `context_recall`：上下文召回率（检索质量）
- `context_precision`：上下文精确率（检索质量）

### 评估陷阱

- LLM 偏向给长回答高分（"长度偏见"）
- 答非所问的回答可能很流畅但得分不应高
- 同义表达识别困难，需人工校准

## 端到端评估闭环

检索评估和生成评估合在一起，形成"策略选择→评估→优化"的实验闭环：

```
构建测试集 → 选择检索策略 → 生成回答 → 检索评估(HR/MRR) + 生成评估(Faithfulness)
     ↑                                                        ↓
     └────────────── 调参优化（网格搜索找最优组合）←──────────┘
```

```python
def evaluate_rag_pipeline(rag, test_cases, evaluator):
    """端到端评估：检索 + 生成 + 评分"""
    results = []
    for query, expected in test_cases:
        result = rag.ask(query, k=3)
        eval_score = evaluator.evaluate_answer(
            query, result["answer"], result["sources"], rag.llm
        )
        results.append({
            "query": query,
            "hit": any(eid in result.get("source_ids", []) for eid in expected),
            "faithfulness": eval_score["faithfulness"],
            "relevance": eval_score["relevance"],
            "is_hallucination": eval_score["is_hallucination"]
        })

    n = len(results)
    return {
        "avg_faithfulness": sum(r["faithfulness"] for r in results) / n,
        "avg_relevance": sum(r["relevance"] for r in results) / n,
        "hallucination_rate": sum(1 for r in results if r["is_hallucination"]) / n
    }
```
