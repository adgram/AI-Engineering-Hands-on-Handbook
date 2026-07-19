# P4：高级 RAG 模式 — Self-RAG / Corrective RAG / Adaptive RAG

## 目标
了解三种高级 RAG 模式，掌握它们解决的问题和适用场景

|          | 学习路径                                                     |
| -------- | ------------------------------------------------------------ |
| 已有基础 | Day4-P1（NaiveRAG）+ Day5-P9（A-RAG 自主决策）+ Day4-P3/P4（评估体系） |
| 本章内容 | 介绍 Self-RAG（自我反思）、Corrective RAG（检索质量保障）、Adaptive RAG（策略自适应）三种模式，以及 Speculative RAG 和 Stream2LLM 加速方案，丰富模式选择。 |

## 模式对比

下表从核心思想与解决的核心问题两个维度，横向对比 Naive RAG、Self-RAG、Corrective RAG 和 Adaptive RAG 四种模式。

| 模式 | 核心思想 | 解决什么问题 |
|------|---------|-------------|
| **Naive RAG** | 检索 → 增强 → 生成 | 基础知识问答 |
| **Self-RAG** | 生成时自我反思，判断是否需要检索 | 检索时机决策 |
| **Corrective RAG** | 检索后评估质量，不合格则重试 | 检索质量保障 |
| **Adaptive RAG** | 动态选择不同检索策略 | 策略自适应 |

## 一、Self-RAG — 自我反思式 RAG

让 LLM 决定"是否需要检索"以及"检索结果是否相关"：

```python
class SelfRAG:
    def __init__(self, collection, client):
        self.collection = collection
        self.client = client
    
    def should_retrieve(self, query: str) -> bool:
        """判断是否需要检索外部知识"""
        response = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": f"""判断以下问题是否需要查找外部知识才能回答。

问题：{query}

如果该问题是常识性或模型训练数据中已有，输出：不需要
如果需要最新信息或特定领域知识，输出：需要
只输出"需要"或"不需要"。"""}],
        )
        decision = response.choices[0].message.content.strip()
        return "需要" in decision
    
    def is_relevant(self, query: str, doc: str) -> bool:
        """判断检索到的文档是否与问题相关"""
        response = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": f"""判断以下文档是否与问题相关。

问题：{query}
文档：{doc[:200]}

只输出"相关"或"不相关"。"""}],
        )
        return "相关" in response.choices[0].message.content
    
    def ask(self, query: str) -> str:
        # 第一步：决定是否需要检索
        if not self.should_retrieve(query):
            response = self.client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": query}]
            )
            return response.choices[0].message.content
        
        # 第二步：检索
        results = self.collection.query(query_texts=[query], n_results=5)
        
        # 第三步：过滤不相关结果
        relevant_docs = []
        for doc in results['documents'][0]:
            if self.is_relevant(query, doc):
                relevant_docs.append(doc)
        
        if not relevant_docs:
            return "检索到的资料与问题不相关，无法回答。"
        
        # 第四步：用相关文档回答
        context = "\n".join(relevant_docs)
        response = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": f"资料：{context}\n\n问题：{query}"}]
        )
        return response.choices[0].message.content
```

## 二、Corrective RAG — 纠正式 RAG

检索后评估质量，如果不够好则重写查询重试：

```python
class CorrectiveRAG:
    def __init__(self, collection, client, max_retries=2):
        self.collection = collection
        self.client = client
        self.max_retries = max_retries
    
    def evaluate_retrieval(self, query: str, docs: list) -> dict:
        """评估检索质量"""
        prompt = f"""评估以下检索结果是否足以回答问题。

问题：{query}

检索结果：
{"".join([f"[{i+1}] {d[:150]}\n" for i, d in enumerate(docs)])}

评估：
1. 检索结果是否与问题相关？(0-10)
2. 信息量是否足够回答问题？(0-10)
3. 是否需要重新检索？

输出 JSON：
{{"relevance": 0-10, "sufficiency": 0-10, "should_retry": true/false, "rewrite_suggestion": "改写建议"}}"""
        
        resp = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    
    def ask(self, query: str) -> str:
        current_query = query
        
        for attempt in range(self.max_retries + 1):
            results = self.collection.query(query_texts=[current_query], n_results=3)
            docs = results['documents'][0]
            
            if not docs:
                return "未检索到相关信息。"
            
            # 评估
            eval_result = self.evaluate_retrieval(current_query, docs)
            
            if not eval_result.get("should_retry", False):
                # 质量达标，生成答案
                context = "\n".join(docs)
                resp = self.client.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=[{"role": "user", "content": f"资料：{context}\n\n问题：{query}"}]
                )
                return resp.choices[0].message.content
            
            # 质量不达标，重写查询
            rewrite = eval_result.get("rewrite_suggestion", "")
            if rewrite and attempt < self.max_retries:
                print(f"第{attempt+1}次检索质量不足，重写查询: {current_query} → {rewrite}")
                current_query = rewrite
        
        return "多次检索后仍无法获得足够信息。"
```

## 三、Adaptive RAG — 自适应 RAG

根据问题类型动态选择策略：

```python
class AdaptiveRAG:
    def __init__(self, collection, client):
        self.collection = collection
        self.client = client
    
    def classify_question(self, query: str) -> str:
        """分类问题类型"""
        resp = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": f"""将问题分类：
- "factual": 事实性问题（需要精确知识）
- "reasoning": 推理问题（需要逻辑分析）
- "creative": 创意问题（需要生成）
- "simple": 简单常识（不需要检索）

问题：{query}
分类："""}],
        )
        return resp.choices[0].message.content.strip()
    
    def ask(self, query: str) -> str:
        qtype = self.classify_question(query)
        print(f"[AdaptiveRAG] 问题类型: {qtype}")
        
        if qtype == "simple":
            # 简单问题，直接用 LLM
            resp = self.client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": query}]
            )
            return resp.choices[0].message.content
        
        elif qtype == "factual":
            # 事实问题：检索 + 严格基于资料回答
            results = self.collection.query(query_texts=[query], n_results=5)
            context = "\n".join([f"[{i+1}] {d}" for i, d in enumerate(results['documents'][0])])
            resp = self.client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": f"严格基于以下资料回答问题。如果资料中没有，请说不知道。\n\n{context}\n\n问题：{query}"}],
            )
            return resp.choices[0].message.content
        
        elif qtype == "reasoning":
            # 推理问题：检索 + CoT
            results = self.collection.query(query_texts=[query], n_results=3)
            context = "\n".join(results['documents'][0])
            resp = self.client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": f"资料：{context}\n\n问题：{query}\n\n请一步步推理。"}],
            )
            return resp.choices[0].message.content
        
        else:
            # 创意问题：宽松检索作为参考
            results = self.collection.query(query_texts=[query], n_results=2)
            context = "\n".join(results['documents'][0]) if results['documents'][0] else "无"
            resp = self.client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": f"参考信息：{context}\n\n问题：{query}\n\n发挥创意。"}],
            )
            return resp.choices[0].message.content
```

## 三种模式选择指南

根据知识库质量、问题类型和准确率要求，下表给出了不同场景下的推荐模式选择。

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| 知识库质量高 | Naive RAG | 简单高效 |
| 知识库质量参差 | Corrective RAG | 需要质量保障 |
| 混合类型问题 | Adaptive RAG | 动态选择策略 |
| 需要高准确率 | Self-RAG | 自我反思减少幻觉 |

## 四、Speculative RAG — 投机式加速（加速方案，与缓存策略 P2 同维度）

借鉴"投机采样"思想：用小模型并行起草候选答案，大模型只做验证：

```python
class SpeculativeRAG:
    """
    Speculative RAG：小模型起草 + 大模型验证。
    用便宜的模型先生成多个候选，大模型只需验证（而非从头阅读长上下文）。
    """
    
    def __init__(self, small_model="deepseek-chat", large_model="deepseek-v4-flash"):
        self.small_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        self.large_client = self.small_client  # 这里用同一client区分模型名
        self.small_model = small_model
        self.large_model = large_model
    
    def draft_answers(self, context: str, query: str, n_drafts: int = 3) -> list:
        """用小模型生成多个候选答案"""
        drafts = []
        for i in range(n_drafts):
            resp = self.small_client.chat.completions.create(
                model=self.small_model,
                messages=[{"role": "user", "content": f"基于以下资料回答问题：\n{context}\n\n问题：{query}"}],
                temperature=0.7 + i * 0.1,  # 不同温度生成多样性
            )
            drafts.append(resp.choices[0].message.content)
        return drafts
    
    def verify(self, query: str, context: str, draft: str) -> dict:
        """大模型验证候选答案的正确性"""
        resp = self.large_client.chat.completions.create(
            model=self.large_model,
            messages=[{"role": "user", "content": f"""验证以下候选答案是否正确。

资料：{context}

问题：{query}

候选答案：{draft}

输出 JSON：{{"correct": true/false, "score": 0-10, "corrections": "..."}}"""}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    
    def answer(self, query: str, context: str) -> str:
        """投机式 RAG 主流程"""
        # 小模型并行起草
        drafts = self.draft_answers(context, query, n_drafts=3)
        
        # 大模型验证（只读简短候选，而非长上下文）
        best_draft = None
        best_score = -1
        for draft in drafts:
            result = self.verify(query, context, draft)
            if result.get("score", 0) > best_score:
                best_score = result["score"]
                best_draft = draft if result.get("correct") else result.get("corrections", draft)
        
        return best_draft

# 使用
spec = SpeculativeRAG()
answer = spec.answer("RAG 是什么？", "RAG 是检索增强生成技术，结合检索和LLM生成回答。")
print(answer)
```

## 五、Stream2LLM — 流式检索与预填充重叠（加速方案）

将检索操作与 LLM 预填充阶段重叠，实现"零等待"流式交互：

```
核心思路：
检索（向量查询）   ──→  LLM 预填充（KV Cache 计算）
        ↑ 并行执行 ↑
两者重叠 → 用户感知"零等待"

实际部署需配合 vLLM / TensorRT-LLM 等推理框架的 LCP（最长公共前缀）缓存机制，
避免不必要的 KV Cache 重计算。
```

简化的异步伪代码示意：

```python
# 注：实际需配合推理框架，以下仅为概念示意
async def stream_rag(collection, client, query: str) -> str:
    # 检索与系统提示预填充并行执行
    results = await async_retrieve(collection, query)  # 异步检索
    contexts = results['documents'][0]
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "system", "content": "你是知识库问答助手。"},
                  {"role": "user", "content": f"资料：{'\n'.join(contexts)}\n\n问题：{query}"}],
    )
    return response.choices[0].message.content
```

## 完成标准
- [ ] 理解三种高级 RAG 模式的原理
- [ ] 实现至少一种高级 RAG 模式
- [ ] 对比高级 RAG 与 Naive RAG 的效果差异
- [ ] 能根据场景选择合适的 RAG 模式
- [ ] 了解 Speculative RAG 和 Stream2LLM 等加速方案

## 下一步 → P5-GraphRAG.md
