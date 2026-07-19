# 知识库RAG | P8：进阶 RAG 架构

## 前言

本节介绍这些实用的进阶技术，让 RAG 从"能跑"走向"好用"。下一篇 P9 将把这些技术整合为一个完整的企业级案例。

## 一、上下文窗口管理

P5 的 Naive RAG 固定选取 Top-K 个文档，但并不一定这几个都是合适的，或不一定包含了所有合适的，还需要继续进行动态过滤。

**策略 1：Score 阈值过滤**——设定固定距离上限，丢弃低质量片段。简单但参数难以确定。

**策略 2：动态 Top-K**——检测距离突变点（相邻结果距离差超过 gap 则截断），自适应选择返回数量。通用性强。

**策略 3：相对距离比**——以最佳匹配距离为基准乘以倍数设定截止线，适合长尾分布。

```python
class DynamicFilter:
    def filter_by_relative_ratio(self, results, ratio=1.5):
        """相对距离比过滤：最佳距离 × ratio 为截止线"""
        distances = results['distances'][0]
        if not distances:
            return []
        threshold = distances[0] * ratio  # 以最近距离为基准
        return [d for d, dist in zip(results['documents'][0], distances) if dist <= threshold]

    def filter_by_gap(self, results, gap_threshold=0.15):
        """动态 Top-K：检测距离突变点"""
        distances = results['distances'][0]
        cutoff = len(distances)
        for i in range(1, len(distances)):
            if distances[i] - distances[i-1] > gap_threshold:
                cutoff = i
                break
        return results['documents'][0][:cutoff]
```

## 二、多轮对话

在应用中，通常知识库是多轮对话，且用户可能不会一次完整表述所有条件，并且存在一些代词（指代前文内容），需要进行上下文管理。

1. **指代消解**：用 LLM 结合最近 3 轮历史将当前问题改写为独立查询
2. **历史上下文注入**：将格式化历史拼入 Prompt，用滑动窗口控制轮数
3. **对话状态管理**：追踪当前主题、提及实体、待澄清项

```python
class ConversationalRAG:
    def __init__(self, rag, max_history=3):
        self.rag = rag
        self.max_history = max_history

    def rewrite_query(self, query: str, history: list) -> str:
        """指代消解：将带代词的追问改写为独立查询"""
        history_str = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in history[-self.max_history:]])
        resp = self.rag.llm.chat.completions.create(
            model=self.rag.llm_model,
            messages=[{"role": "user", "content": f"""根据对话历史，将当前问题改写为独立完整的问题（只输出改写后的问题）：

对话历史：
{history_str}

当前问题：{query}

改写后："""}]
        )
        return resp.choices[0].message.content.strip()

    def ask(self, query: str, history: list) -> dict:
        # 先改写查询，再检索
        rewritten = self.rewrite_query(query, history) if history else query
        result = self.rag.ask(rewritten)
        return {**result, "rewritten_query": rewritten}
```

查询改写成本低、效果好，是多轮对话 RAG 的首选方案。

## 三、引用溯源与可信度

为了让回答更可信，应该让 RAG 标注信息来源，满足可信度、可验证、可追溯、合规四方面需求。

**方案 1：Prompt 引导引用**——System Prompt 要求 LLM 在句末标注 `[编号]`，生成与引用同步。最简单。

**方案 2：结构化引用输出**——用 JSON Mode 输出 `answer + citations` 数组，便于程序解析。最适合工程化。

**方案 3：Post-hoc 引用匹配**——先生成回答，再用 LLM 逐文档匹配句子来源。灵活但成本高。

```python
def generate_with_citations(rag, query, contexts):
    """结构化引用输出"""
    context_str = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    resp = rag.llm.chat.completions.create(
        model=rag.llm_model,
        response_format={"type": "json_object"},  # 强制 JSON
        messages=[{"role": "user", "content": f"""基于以下资料回答问题，返回 JSON：
{{
  "answer": "回答正文，在引用处标注[编号]",
  "citations": [{{"id": 1, "text": "引用的原文片段"}}]
}}

参考资料：
{context_str}

问题：{query}"""}]
    )
    return json.loads(resp.choices[0].message.content)
```

## 四、查询重写与分解

用户输入往往带有口语化表达、错别字或指代不清等问题，直接检索效果不佳。非标准输入和复合问题是检索的两大难点，需要查询优化技术。

### 查询重写

用 LLM 补全术语、消除口语化、纠正错别字；还可生成多个同义变体做多查询检索合并去重，提升查全率：

```python
class QueryRewriter:
    def rewrite(self, query: str) -> str:
        """标准化查询：补全术语、纠正口语"""
        resp = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"将以下查询改写为更清晰、术语更准确的形式（只输出改写结果）：\n{query}"}]
        )
        return resp.choices[0].message.content.strip()

    def expand_with_synonyms(self, query: str, n=3) -> list:
        """生成同义变体，多查询检索提升查全率"""
        resp = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"将以下查询改写为{n}个同义但表述不同的变体（每行一个）：\n{query}"}]
        )
        return [q.strip() for q in resp.choices[0].message.content.split("\n") if q.strip()]
```

### 查询分解

将复合问题拆为 2-5 个子问题分别检索，合并上下文后生成完整回答。适合"比较 A 和 B 的区别"这类多跳问题：

```python
class DecompositionRAG:
    def decompose(self, query: str) -> list:
        """将复合问题拆为子问题"""
        resp = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"将以下问题拆解为2-5个独立子问题（每行一个）：\n{query}"}]
        )
        return [q.strip() for q in resp.choices[0].message.content.split("\n") if q.strip()]

    def ask(self, query: str) -> str:
        sub_queries = self.decompose(query)
        all_contexts = []
        for sq in sub_queries:
            results = self.store.search(sq, n_results=3)
            all_contexts.extend(results['documents'][0])
        # 去重后合并生成
        unique_contexts = list(dict.fromkeys(all_contexts))
        return self.generate(query, unique_contexts)
```

| 技术 | 适用场景 | 效果 | 成本 |
|------|---------|------|------|
| 查询重写 | 标准化输入 | 准确率↑10-30% | 低（1次LLM） |
| 同义扩展 | 提升查全率 | 召回↑，查准可能↓ | 中（多次检索） |
| 查询分解 | 多跳复合问题 | 复杂问题大幅提升 | 高（多次检索+生成） |

## 五、增量更新 — 知识库动态维护

知识库不是静态的——文档会新增、修改、删除。每次都全量重建索引成本太高，需要增量同步策略，让知识库从静态构建升级为动态维护。

**全量重建**：删旧重建，简单但成本高。适合每日批量更新，保证一致性。

**增量添加**：检查 ID 去重后只加新文档。适合实时新增，低延迟。

**文档更新**：`update` 失败则 `upsert` 原地替换。适合频繁修改。

**定时同步**：用 MD5 哈希监控文件系统变化，自动增删改。适合文件型知识库的自动化维护。

```python
class FileSyncIndexer:
    """文件系统监控 + 自动同步"""
    def __init__(self, store, watch_dir):
        self.store = store
        self.watch_dir = watch_dir
        self.file_hashes = {}  # 记录已知文件指纹

    def sync(self):
        current_files = {}
        for root, _, files in os.walk(self.watch_dir):
            for f in files:
                path = os.path.join(root, f)
                with open(path, "rb") as fh:
                    current_files[path] = hashlib.md5(fh.read()).hexdigest()

        # 新增/修改的文件
        for path, fhash in current_files.items():
            if path not in self.file_hashes or self.file_hashes[path] != fhash:
                self.store.upsert_document(path)  # 重新索引
                self.file_hashes[path] = fhash

        # 删除的文件
        for path in list(self.file_hashes):
            if path not in current_files:
                self.store.delete_document(path)
                del self.file_hashes[path]
```

**版本管理**：记录版本号支持回滚。适合需要审计追溯的场景。

- 选型建议

| 场景 | 推荐策略 |
|------|---------|
| 每日批量更新 | 全量重建 |
| 实时新增 | 增量添加 |
| 频繁修改 | Upsert |
| 文件型知识库 | 定时同步（FileSyncIndexer） |
| 需审计回滚 | 版本管理 |
