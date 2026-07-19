# 知识库RAG | P2：向量数据库 ChromaDB

## 前言

P1 介绍了 Embedding 如何把文本变成向量。但实际应用中，知识库可能有成千上万条文档，每次查询都要和所有向量逐一计算相似度显然不现实——这就需要**向量数据库**。

向量数据库专门存储和检索向量数据，能在海量向量中快速找到最相似的那些。本节使用 ChromaDB 进行演示。

## 什么是向量数据库

一条向量数据库记录存储三类内容：

| 内容 | 示例 | 说明 |
|------|------|------|
| **向量** | `[0.12, -0.34, 0.56, ...]` | 文本经过 Embedding 后的语义向量，用于相似度计算 |
| **原始文本** | `"RAG 全称 Retrieval-Augmented Generation"` | 向量的原始内容，检索后直接返回给用户 |
| **元数据** | `{"source": "wiki", "topic": "RAG"}` | 附带的标签/属性，用于过滤筛选 |

核心能力：存储向量 + 原始文本 + 元数据，并支持快速近似最近邻搜索（ANN）和元数据过滤。

## 基础操作

```bash
pip install chromadb
```

```python
import chromadb
from chromadb.config import Settings

# 1. 创建客户端（持久化存储到本地）
client = chromadb.PersistentClient(
    path="./chroma_db",  # 数据存储目录
    settings=Settings(anonymized_telemetry=False)
)

# 2. 创建/获取集合
collection = client.get_or_create_collection(
    name="my_knowledge_base",
    metadata={"description": "我的第一个知识库"}
)

# 3. 添加文档（ChromaDB 会自动调用 Embedding 函数向量化）
collection.add(
    documents=[
        "RAG 全称 Retrieval-Augmented Generation，即检索增强生成",
        "Prompt Engineering 是设计和优化 AI 提示词的技术",
        "Agent 编排是指管理和协调多个 AI Agent 协同工作",
        "向量数据库专门用于存储和检索向量数据",
        "ChromaDB 是一个轻量级向量数据库，适合学习和原型开发"
    ],
    metadatas=[
        {"source": "wiki", "topic": "RAG"},
        {"source": "wiki", "topic": "Prompt"},
        {"source": "wiki", "topic": "Agent"},
        {"source": "wiki", "topic": "Database"},
        {"source": "wiki", "topic": "Database"}
    ],
    ids=["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
)

print(f"集合中已有 {collection.count()} 条文档")
```

### 相似度搜索

查询时只需传入文本，ChromaDB 会自动（调用大模型）将其向量化并与库中所有向量比较，返回最相似的 K 条原始文本：

```python
results = collection.query(
    query_texts=["什么是 RAG？"],
    n_results=3  # 返回最相似的 3 条
)

for i, (doc, metadata, distance) in enumerate(zip(
    results['documents'][0], results['metadatas'][0], results['distances'][0]
)):
    print(f"{i+1}. 文档: {doc}")
    print(f"   来源: {metadata['source']}  距离: {distance:.4f}")  # 数值越小越相似
```

### 更新与删除

```python
collection.update(ids=["doc_1"], documents=["更新后的内容..."])
collection.delete(ids=["doc_5"])
```

## 接入自定义 Embedding 函数

ChromaDB 默认使用 Sentence Transformers 的 `all-MiniLM-L6-v2`，这是一个英文模型，对中文的语义理解效果很差。这些需要替换为中文优化模型。

```python
from chromadb import Documents, EmbeddingFunction
from openai import OpenAI
import os

class SiliconFlowEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: str = "BAAI/bge-m3"):
        self.client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1"
        )
        self.model = model

    def __call__(self, input: Documents) -> list:
        texts = input if isinstance(input, list) else [input]
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [data.embedding for data in response.data]

# 创建集合时注入自定义 Embedding
collection = client.get_or_create_collection(
    name="siliconflow_kb",
    embedding_function=SiliconFlowEmbeddingFunction()  # 使用 BAAI/bge-m3
)
```

## 相似度度量方式

相似度检索，可以在 ChromaDB 创建集合时指定距离函数（创建后不可修改）：

```python
collection = client.create_collection(
    name="cosine_demo",
    metadata={"hnsw:space": "cosine"}  # 可选: cosine / l2 / ip
)
```

| 度量方式 | 公式 | 范围 | 推荐场景 |
|---------|------|------|---------|
| 余弦相似度（cosine） | `A·B / (|A|×|B|)` | [-1, 1] 越大越相似 | 文本语义搜索（默认） |
| 点积（ip） | `Σ(Ai × Bi)` | 无限制 | 向量已归一化时等于余弦 |
| 欧氏距离（l2） | `√Σ(Ai - Bi)²` | [0, +∞) 越小越相似 | 图像/音频特征检索 |

## ANN 索引

向量数据库不可能真的和每个向量逐一计算（100 万文档要算 100 万次），而是使用 **ANN（近似最近邻）** 算法把搜索复杂度从 O(n) 降到 O(log n)：

```
暴力搜索（Brute Force）：O(n)  → 100万文档要算100万次
ANN 搜索：O(log n)  → 使用索引结构加速

常见 ANN 算法：
- HNSW（Hierarchical Navigable Small World）← ChromaDB 默认
- IVF（Inverted File Index）
- PQ（Product Quantization）
```

代价是精度略有损失，但实际检索质量损失通常 <5%，而速度提升数十倍，是工程上非常划算的交换。

## 检索质量评估

检索质量评估的核心原理：用一组预定义的 `query → 期望结果` 测试用例，量化检索系统返回结果的好坏。有两个核心指标：

| 指标 | 含义 | 测什么 |
|------|------|--------|
| **Hit Rate（命中率）** | 期望结果是否出现在 Top-K 中 | "能不能找得到" |
| **MRR（平均倒数排名）** | 第一个相关结果的排序位置倒数取平均 | "找到的排得够不够靠前" |

```python
def evaluate_search(collection, queries_with_expected):
    total = len(queries_with_expected)
    hit_count, mrr = 0, 0

    for query, expected_ids in queries_with_expected:
        results = collection.query(query_texts=[query], n_results=5)
        retrieved_ids = results['ids'][0]

        if any(eid in retrieved_ids for eid in expected_ids):
            hit_count += 1

        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in expected_ids:
                mrr += 1.0 / rank
                break

    return {"hit_rate": hit_count / total, "mrr": mrr / total}

test_cases = [("什么是 RAG？", ["doc_1"]), ("向量数据库怎么用", ["doc_4", "doc_5"])]
metrics = evaluate_search(collection, test_cases)
print(f"Hit Rate: {metrics['hit_rate']:.0%}  MRR: {metrics['mrr']:.3f}")
```

**Hit Rate 看召回有没有漏，MRR 看排序好不好**——这是后续 RAG 调优的量化基础。

## 多知识库组织策略

实际项目通常有多个知识库，ChromaDB 通过两层结构组织：

```
PersistentClient("chroma_db")     ← 一个数据库目录
├── Collection("kb_excerpts")     ← 名言金句库
├── Collection("kb_arch_std")     ← 建筑设计规范库
└── Collection("kb_enum_code")    ← Python 代码库
```

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 不同项目/租户 | 独立 DB 目录 | 完全隔离，可单独迁移/备份 |
| 同一项目不同领域 | 同一 DB 下不同集合 | 共用 Embedding 模型，统一管理 |
| 测试/生产环境 | 独立 DB 目录 | 环境隔离，避免数据污染 |

### 多文件合并知识库

知识库常来自多个文件（JSON、Markdown、Python 代码），用 `metadata` 标记来源后统一索引到同一集合，查询时按来源过滤：

```python
all_docs, all_metas = [], []

# JSON 文件 — 每条独立条目
for e in excerpts:
    all_docs.append(e["content"])
    all_metas.append({"source": "excerpts", "file": "excerpts.json"})

# Markdown 文件 — 按标题分块
for chunk in re.split(r'\n(?=## )', md_text):
    all_docs.append(chunk)
    all_metas.append({"source": "arch_std", "file": "标准.md"})

# Python 文件 — 按 class/def 分块
for chunk in re.split(r'\n(?=(class |def ))', py_text):
    all_docs.append(chunk)
    all_metas.append({"source": "enum_code", "file": "enum.py"})

# 统一索引
col.add(documents=all_docs, metadatas=all_metas, ids=[...])

# 查询时按来源过滤
col.query(query_texts=["建筑设计"], where={"source": "arch_std"})
```
