# P5：GraphRAG — 知识图谱 + RAG

## 目标
了解 GraphRAG 的原理，用知识图谱增强 RAG 的关系理解能力

|          | 学习路径                                                     |
| -------- | ------------------------------------------------------------ |
| 已有基础 | Day3-P5（元数据关联、实体级检索）+ Day5-P4（查询扩展） |
| 本章内容 | 用 LLM 提取实体和关系构建知识图谱，查询时通过图扩展找到相关实体，补足向量检索在多跳推理上的短板。 |

## 什么是 GraphRAG？

传统 RAG：文档之间的跳转是独立的
GraphRAG：用知识图谱连接实体，支持多跳推理

## GraphRAG 的两种实现方式

根据实际需求和技术栈的不同，GraphRAG 有多种实现路径。目前主流的有两种：一是微软官方的 GraphRAG 框架（基于 LLM 自动构建知识图谱），二是结合 Neo4j 图数据库的自定义实现方式。

### 方式 1：微软 GraphRAG（官方实现）

微软 GraphRAG 提供开箱即用的知识图谱构建能力，通过 pip 安装后配置 LLM 与 Embedding 即可使用。

```bash
pip install graphrag
```

详见微软 GraphRAG 文档，需配置 LLM + Embedding。

### 方式 2：自建轻量知识图谱 + RAG

手动建图：

```python
import json
from collections import defaultdict

class SimpleGraphRAG:
    """简单版 GraphRAG：实体提取 + 关系构建 + 图检索"""
    
    def __init__(self, collection, client):
        self.collection = collection
        self.client = client
        self.graph = defaultdict(list)  # entity → [related_entities]
        self.entity_docs = defaultdict(list)  # entity → [doc_ids]
    
    def extract_entities(self, text: str) -> list:
        """从文本中提取实体"""
        resp = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": f"""从以下文本中提取所有实体（人名、技术名词、概念等）。

文本：{text[:500]}

输出 JSON，格式：{{"entities": ["实体1", "实体2", ...]}}"""}],
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(resp.choices[0].message.content)
            return data.get("entities", []) if isinstance(data, dict) else []
        except:
            return []
    
    def build_graph(self, texts: list):
        """从文档构建知识图谱"""
        for doc_id, text in enumerate(texts):
            entities = self.extract_entities(text)
            for e in entities:
                self.entity_docs[e].append(doc_id)
            
            # 同一文档中的实体建立连接
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    self.graph[entities[i]].append(entities[j])
                    self.graph[entities[j]].append(entities[i])
        
        print(f"图谱构建完成: {len(self.graph)} 个实体")
    
    def expand_query(self, query: str) -> str:
        """通过知识图谱扩展查询"""
        # 提取查询中的实体
        entities = self.extract_entities(query)
        
        # 在图中查找相关实体
        related = set()
        for e in entities:
            for rel in self.graph.get(e, []):
                related.add(rel)
        
        if related:
            context = f"相关问题涉及：{', '.join(entities)}\n"
            context += f"相关概念：{', '.join(list(related)[:5])}"
            return f"{query}\n\n{context}"
        
        return query
    
    def ask(self, query: str) -> str:
        # 用图谱扩展查询
        expanded_query = self.expand_query(query)
        
        # 检索
        results = self.collection.query(query_texts=[expanded_query], n_results=5)
        
        if not results['documents'][0]:
            return "未找到相关信息。"
        
        context = "\n".join(results['documents'][0])
        
        # 生成
        resp = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": f"资料：{context}\n\n问题：{query}"}]
        )
        return resp.choices[0].message.content
```

## 标准 GraphRAG 论文方案简介

微软 GraphRAG 论文中的完整流程：

```
1. 文档切分
2. 实体提取（LLM）
3. 关系提取（LLM）
4. 实体聚类（Leiden 算法）
5. 社区摘要生成
6. 查询 → 找到相关社区 → 汇总回答
```

## 传统 RAG vs GraphRAG vs Simple GraphRAG

下表从单文档问答、多文档关联、多跳推理、全局汇总和实现复杂度五个维度对比三种方案的差异。

| 能力 | 传统 RAG | GraphRAG（微软） | Simple GraphRAG |
|------|---------|-----------------|-----------------|
| 单文档问答 | ✅ | ✅ | ✅ |
| 多文档关联 | ❌ | ✅ | ⚠️ 有限 |
| 多跳推理 | ❌ | ✅ | ⚠️ 依赖实体 |
| 全局汇总 | ❌ | ✅ | ❌ |
| 实现复杂度 | ⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

## 完成标准
- [ ] 理解 GraphRAG 的核心思想
- [ ] 实现了一个简单的图增强 RAG
- [ ] 对比了 RAG 和 GraphRAG 在关系型问题上的效果
- [ ] 了解微软 GraphRAG 的完整流程

## 下一步 → P6-企业级RAG知识库MVP.md
