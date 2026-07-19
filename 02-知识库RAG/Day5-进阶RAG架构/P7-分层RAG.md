# P7：分层 RAG（Hierarchical RAG）— 多级知识库路由

## 目标
理解分层 RAG 的多级路由策略，实现按主题/粒度分发检索

|          | 学习路径                                                     |
| -------- | ------------------------------------------------------------ |
| 已有基础 | Day4-P2（Query Routing 按类型路由）+ Day3-P4（多知识库组织：独立 DB vs 独立集合） |
| 本章内容 | 按知识层级路由（全局层/领域层/文档层/ask_llm），解决单一大库噪声多、延迟高的问题，实现精准路由。 |

## 什么是分层 RAG？

分层 RAG 将知识库分为多个层级（如全局层、领域层、文档层），查询时先路由到正确的层级再检索。

```
用户查询 → 路由判断 → [全局知识库 | 领域知识库 | 文档层 | 原始文档]
```

## 分层 RAG 架构

分层 RAG 架构将知识库划分为全局、领域和文档三个层级，并提供一个回退到 LLM 直接回答的选项，通过路由判断决定查询应投向哪个层级。

```python
from typing import Literal

class HierarchicalRAG:
    """
    分层 RAG：多级知识库路由检索
    
    层级设计：
    1. 全局层（Global）：高频/热门知识，小索引，低延迟
    2. 领域层（Domain）：按业务领域划分的中粒度知识
    3. 文档层（Document）：完整文档库，大索引，高延迟
    """
    
    def __init__(self, global_collection, domain_collections: dict, doc_collection, llm_client):
        self.global_col = global_collection
        self.domain_cols = domain_collections  # {"finance": col1, "tech": col2, ...}
        self.doc_col = doc_collection
        self.client = llm_client
    
    def route(self, query: str) -> Literal["global", "domain", "document", "ask_llm"]:
        """路由判断：查询属于哪个层级"""
        resp = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": f"""判断以下查询最适合哪个知识库层级：
- global：基础概念、通用知识
- domain：需要特定领域专业知识
- document：需要查阅具体文档细节
- ask_llm：不需要检索，LLM 可直接回答

查询：{query}

只输出一个词。"""}],
            temperature=0.0,
        )
        level = resp.choices[0].message.content.strip().lower()
        return level if level in ("global", "domain", "document", "ask_llm") else "document"
    
    def route_domain(self, query: str) -> str:
        """进一步判断具体领域"""
        topics = list(self.domain_cols.keys())
        resp = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": f"""可用领域：{', '.join(topics)}
查询：{query}
最适合的领域是？只输出领域名。"""}],
            temperature=0.0,
        )
        domain = resp.choices[0].message.content.strip()
        return domain if domain in self.domain_cols else topics[0]
    
    def retrieve(self, query: str, top_k: int = 5) -> list:
        level = self.route(query)
        
        if level == "ask_llm":
            return []  # LLM 直接回答
        
        if level == "global":
            results = self.global_col.query(query_texts=[query], n_results=top_k)
        elif level == "domain":
            domain = self.route_domain(query)
            results = self.domain_cols[domain].query(query_texts=[query], n_results=top_k)
        else:  # document
            results = self.doc_col.query(query_texts=[query], n_results=top_k)
        
        return results['documents'][0] if results['documents'] else []

# 使用
# hrag = HierarchicalRAG(global_col, domain_cols, doc_col, client)
# docs = hrag.retrieve("什么是 RAG？")  # → 全局层
# docs = hrag.retrieve("2024年财报净利润")  # → 领域层(finance)
# docs = hrag.retrieve("合同第3条第2款")  # → 文档层
```

## 无路由 vs 分层 RAG 效果对比

下表对比无路由的全量搜索与分层 RAG 在不同查询场景下的表现差异，突出分层路由在精准度和效率上的优势。

| 场景 | 无路由（全量搜索） | 分层 RAG |
|------|-------------------|----------|
| 通用知识问答 | 噪声多 | 直达全局层，精准 |
| 跨领域查询 | 可能漏召 | 路由到正确领域 |
| 文档级细节 | 碎片化 | 直达文档层 |
| 简单问候 | 浪费检索 | 直接 LLM 回答 |

## 动手实验

1. 设计至少 3 个层级的知识库
2. 实现路由判断逻辑
3. 测试不同查询的路由准确率
4. 对比分层 vs 单一检索的延迟和准确率

## 完成标准
- [ ] 理解分层 RAG 的核心思想
- [ ] 实现了路由判断逻辑
- [ ] 验证了分层检索的效果提升

## 下一步 → [P8-知识库问答Web应用](P8-知识库问答Web应用.md)
