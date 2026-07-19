# P2：Prompt 中注入上下文的最佳实践

## 目标
掌握将检索结果注入 LLM Prompt 的各种策略及其效果差异。

|          | 学习路径                                                     |
| -------- | ------------------------------------------------------------ |
| 已有基础 | Day2-P1（GSSC 框架：Select/Structure/Compress）+ Day4-P1（NaiveRAG 的朴素拼接）+ Day1-P6（模板化技术） |
| 本章内容 | 把 GSSC 方法论落地到 RAG 场景，实现按相关性排列、Map-Reduce 分而治之、Context Tracker 去重、Query Routing 和五段式工业模板。 |

## 上下文注入的核心问题

当检索结果包含多篇文档时，全部塞入 Prompt 会迅速填满上下文窗口，且不相关信息会稀释关键信号，导致模型"迷失在中间"（Lost in the Middle）。核心是要解决"选择哪些文档"和"如何排列它们"两个问题。

```
检索到 N 篇文档 → 全部塞进 Prompt？
                   ↓
                  问题：窗口有限、关键信息被淹没
                   ↓
                  如何选择和排列？
```

> 前置知识：上下文注入的 GSSC 框架详见 Prompt D2-P1。**Structure→U 型曲线（Lost in the Middle）**和 **Compress→摘要压缩**的原理与实现在该文已完整讲解，本文不再重复，仅聚焦 RAG 场景特有的注入策略。

## 策略 1：按相关性排列（最常用）

将文档按与查询的相关性分数从高到低排序，把最相关的信息放在开头和结尾（U 型曲线的两端），这是最基础也最常用的策略，能在不修改内容的情况下显著提升回答质量。

```python
def sort_by_relevance(docs, distances):
    """按相关性（距离）从高到低排列"""
    paired = sorted(zip(docs, distances), key=lambda x: x[1])
    return [p[0] for p in paired]
```

## 策略 2：分而治之（Map-Reduce 模式）

当文档太多时，先分别回答再汇总：

```python
def map_reduce_rag(docs: list, query: str, llm, model: str = "deepseek-v4-flash") -> str:
    """Map: 每个文档独立回答 → Reduce: 汇总所有回答"""
    # Map 阶段：每个文档独立回答
    partial_answers = []
    for doc in docs:
        resp = llm.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": f"基于以下资料回答问题。如果该资料不包含答案，返回'无相关信息'。\n\n资料：{doc}\n\n问题：{query}"
            }],
        )
        partial_answers.append(resp.choices[0].message.content)
    
    # Reduce 阶段：汇总
    all_answers = "\n".join([f"[资料{i+1}] {a}" for i, a in enumerate(partial_answers)])
    
    final_resp = llm.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"以下是对同一个问题的多个独立回答，请综合它们生成一个完整、一致的最终回答：\n\n{all_answers}\n\n问题：{query}"
        }],
    )
    
    return final_resp.choices[0].message.content
```

## 策略对比实验

在实际项目中，不同注入策略的效果差异很大。以下代码在同一组文档和问题上对比"简单拼接""按相关性排列""Map-Reduce"三种策略，帮助你直观理解各自的优缺点。

```python
def _ask(llm, prompt: str, model: str = "deepseek-v4-flash") -> str:
    """调用 LLM 获取回答"""
    resp = llm.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def naive_prompt(docs: list, query: str, llm=None) -> str:
    """简单拼接：将所有文档直接拼接到 prompt 中"""
    context = "\n".join([f"[{i+1}] {d}" for i, d in enumerate(docs)])
    prompt = f"参考资料：\n{context}\n\n问题：{query}"
    return _ask(llm, prompt)


def sorted_prompt(docs: list, query: str, llm=None) -> str:
    """排序拼接：按相关性排序后再拼入 prompt"""
    mock_distances = [i for i in range(len(docs))]  # 正序
    sorted_docs = sort_by_relevance(docs, mock_distances)
    context = "\n".join([f"[{i+1}] {d}" for i, d in enumerate(sorted_docs)])
    prompt = f"参考资料（按相关性排列）：\n{context}\n\n问题：{query}"
    return _ask(llm, prompt)


def compare_injection_strategies(query: str, docs: list, llm):
    """对比不同注入策略的效果"""
    strategies = {
        "简单拼接": lambda: naive_prompt(docs, query, llm),
        "按相关性排列": lambda: sorted_prompt(docs, query, llm),
        "Map-Reduce": lambda: map_reduce_rag(docs, query, llm),
    }
    results = {}
    for name, fn in strategies.items():
        answer = fn()
        results[name] = answer
        print(f"\n=== {name} ===")
        print(answer[:200] + "...")
    return results

# 准备文档（从 rag_knowledge 目录加载）
# loaded = load_directory(data_dir)
# docs = [d["content"] for d in loaded]

# 前置准备：通过 BaseRAG 获取 llm
# rag = BaseRAG(persist_dir="./chroma_db")
# compare_injection_strategies("RAG 相比微调有什么优势？", docs, rag.llm)

# 延伸：压缩策略的实现见 Prompt D2-P1（GSSC-Compress）
```

## 注入策略选择指南

不同场景下上下文注入策略的选择直接影响效果与成本，下表提供了快速决策参考：

| 情景 | 推荐策略 | 原因 |
|------|---------|------|
| 文档少（≤3）且短 | 直接拼接 | 简单有效 |
| 文档多（≥5） | 按相关性排列（排列策略见 Prompt D2-P1 Structure→U型曲线） | 控制质量 |
| 文档长 | 先压缩再拼接（压缩原理见 Prompt D2-P1） | 节省 Token |
| 文档矛盾 | Map-Reduce | 发现矛盾并综合 |
| 实时要求高 | 直接拼接（最简单，最快） | 速度优先 |

## 策略 3：Context Tracker（智能体上下文追踪）

> 记忆管理的完整体系（短期/长期/三层架构）见 Agent，本节仅展示 RAG 独有的检索去重追踪功能。

在多次检索中追踪已读文档，避免重复读取和 Token 浪费（来自 A-RAG 的设计）：

```python
import hashlib

class ContextTracker:
    """追踪已读过的文档块，避免重复读取"""
    def __init__(self):
        self.read_chunks = set()  # 已读 chunk ID
        self.read_content = set()  # 已读内容指纹
    
    def has_read(self, chunk_id: str) -> bool:
        return chunk_id in self.read_chunks
    
    def mark_read(self, chunk_id: str, content: str):
        self.read_chunks.add(chunk_id)
        fp = hashlib.md5(content.encode()).hexdigest()
        self.read_content.add(fp)
    
    def filter_new(self, chunks: list) -> list:
        """过滤出未读过的chunk"""
        new = []
        for c in chunks:
            cid = c.get("id", "")
            if cid not in self.read_chunks:
                new.append(c)
                self.mark_read(cid, c.get("content", ""))
        return new

# 在多轮检索中使用
# 前置准备：from common.rag_base import BaseRAG
#           rag = BaseRAG(persist_dir="./chroma_db")

tracker = ContextTracker()

# 第一轮
results1 = rag.store.search("RAG 是什么？", n_results=3)
new_results = tracker.filter_new([
    {"id": rid, "content": doc}
    for rid, doc in zip(results1['ids'][0], results1['documents'][0])
])
print(f"新增结果: {len(new_results)}")

# 第二轮 — 已读的内容不再注入
results2 = rag.store.search("RAG 的优势", n_results=3)
new_results2 = tracker.filter_new([
    {"id": rid, "content": doc}
    for rid, doc in zip(results2['ids'][0], results2['documents'][0])
])
print(f"新增结果: {len(new_results2)}（已过滤掉重复）")
```

## 查询路由（Query Routing）完整讲解见 Day5

将查询路由到不同的处理策略，而不是对所有查询使用相同的 RAG 流程：

```python
class QueryRouter:
    """根据问题类型路由到不同的策略。依赖注入：store(VectorStore) + llm(OpenAI)"""
    def __init__(self, store, llm, model: str = "deepseek-v4-flash"):
        self.store = store
        self.llm = llm
        self.model = model
    
    def route(self, query: str) -> str:
        """返回策略名: simple / factual / reasoning / creative"""
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"""分类问题类型（只输出单词）：
- simple: 简单常识（如"今天星期几？"）
- factual: 事实性知识库问题（如"RAG 是什么？"）
- reasoning: 需要多步推理（如"比较 A 和 B 的区别"）
- creative: 创意生成（如"写一首诗"）

问题：{query}
类型："""}],
        )
        return response.choices[0].message.content.strip().lower()
    
    def execute(self, query: str) -> str:
        qtype = self.route(query)
        
        if qtype == "simple":
            resp = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": query}]
            )
            return resp.choices[0].message.content
        
        elif qtype == "factual":
            results = self.store.search(query, n_results=5)
            context = "\n".join(results['documents'][0])
            resp = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": f"严格基于以下资料回答：\n{context}\n\n问题：{query}"}]
            )
            return resp.choices[0].message.content
        
        elif qtype == "reasoning":
            results = self.store.search(query, n_results=3)
            context = "\n".join(results['documents'][0])
            resp = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": f"资料：{context}\n\n问题：{query}\n请一步步推理。"}]
            )
            return resp.choices[0].message.content
        
        else:
            results = self.store.search(query, n_results=2)
            context = "\n".join(results['documents'][0]) if results['documents'][0] else "无"
            resp = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": f"参考信息：{context}\n\n问题：{query}\n发挥创意回答。"}]
            )
            return resp.choices[0].message.content

# 使用
# from common.rag_base import BaseRAG
# rag = BaseRAG(persist_dir="./chroma_db")
# router = QueryRouter(store=rag.store, llm=rag.llm)
# for q in ["RAG 是什么？", "写一首关于AI的诗", "比较RAG和微调"]:
#     print(f"[{router.route(q)}] {q}")
```

## 工业实践：五段式提示模板

> 角色设定方法论详见 Prompt D1-P2；通用模板化技术（f-string/Jinja2/YAML）详见 Prompt D1-P6。

字节跳动在生成层使用标准化的五段式提示模板，结构如下：

```
【角色定义】你是专业的知识库问答助手，需基于提供的检索信息回答问题。
【任务指令】用户的问题是：{query}，请结合以下检索信息，生成准确、简洁的回答。
【检索信息】
{context_str}
【格式要求】回答需分点说明，每点不超过30字。若信息不足，请说明"当前信息不足以完全回答"。
【示例引导】
用户查询：抖音小店入驻需多少钱？
检索信息：1. 入驻需缴纳2000元保证金，不同类目可能调整
参考回答：1. 入驻需缴2000元保证金，不同类目可能调整
```

配合动态筛选——相似度 < 0.7 的片段不传入提示，避免噪声干扰模型注意力：

```python
def build_prompt(query: str, contexts: list) -> str:
    top = [c for c in contexts if c.get("score", 0) >= 0.7][:5]
    context_str = "\n".join([f"{i+1}. {c['content'][:200]}（相似度：{c.get('score', 0):.2f}）" for i, c in enumerate(top)])
    return f"""【角色定义】你是专业的知识库问答助手，需基于提供的检索信息回答问题。
【任务指令】用户的问题是：{query}，请结合以下检索信息，生成准确、简洁的回答。
【检索信息】
{context_str}
【格式要求】回答需分点说明，每点不超过30字。若信息不足，请说明"当前信息不足以完全回答"。
【示例引导】
用户查询：抖音小店入驻需多少钱？
检索信息：1. 入驻需缴纳2000元保证金，不同类目可能调整
参考回答：1. 入驻需缴2000元保证金，不同类目可能调整"""

prompt = build_prompt("RAG 是什么？", [
    {"content": "RAG 是检索增强生成...", "score": 0.92},
    {"content": "不相关的内容...", "score": 0.35},
])
print(prompt)
```

关键策略：**动态筛选**——相似度<0.7的片段不传入提示，避免噪声干扰模型注意力。

## 动手实验

1. 准备 8-10 篇混合相关/不相关的文档
2. 用至少 3 种注入策略回答同一个问题
3. 对比各策略的回答质量、Token 消耗、响应速度
4. 总结出你自己在不同场景下的策略选择
5. 实现 Context Tracker 并验证其在多轮检索中的 Token 节省效果
6. 实现五段式模板并对比与简单拼接的生成质量差异
7. 阅读 Prompt D2-P1 的 GSSC 框架，思考 Structure→U 型曲线和 Compress→摘要压缩在 RAG 场景的应用

## 完成标准
- [ ] 理解 GSSC 框架对 RAG 上下文注入的指导意义
- [ ] 实现至少 3 种上下文注入策略
- [ ] 对同一问题对比了不同策略的效果
- [ ] 能根据场景选择合适的策略
- [ ] 了解五段式提示模板的工业实践

## 下一步 → [P3-检索质量评估](P3-检索质量评估.md)