# P2：向量数据库入门 — ChromaDB

## 目标
安装并掌握 ChromaDB 的基本操作：创建集合、添加文档、相似度搜索

## 什么是向量数据库？

专门存储和检索向量数据的数据库。一条记录可以存储三类内容：

| 内容 | 示例 | 说明 |
|------|------|------|
| **向量** | `[0.12, -0.34, 0.56, ...]` | 文本/图片经过 Embedding 后的语义向量，用于相似度计算 |
| **原始文本** | `"RAG 全称 Retrieval-Augmented Generation"` | 向量的原始内容，检索后直接返回给用户 |
| **元数据** | `{"source": "wiki", "topic": "RAG"}` | 附带的标签/属性，用于过滤筛选 |

核心功能：
- 存储向量 + 原始文本 + 元数据
- 快速近似最近邻搜索（ANN）
- 支持元数据过滤

## 安装 ChromaDB

```bash
pip install chromadb
```

## 基础操作

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

# 3. 添加文档（带向量和元数据）
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

## 相似度搜索

相似度检索是将输入的文本转换为向量，然后在数据库中进行匹配。

相似度搜索的流程为：输入文本 → Embedding 模型（自动）→ 查询向量 → 与库中所有存储向量做距离比较 → 返回最相似的 K 条**原始文本**。

```python
# 4. 搜索（默认使用集合自带的 Embedding 函数）
results = collection.query(
    query_texts=["什么是 RAG？"],
    n_results=3  # 返回最相似的 3 条
)

print("查询结果:")
for i, (doc, metadata, distance) in enumerate(zip(
    results['documents'][0],
    results['metadatas'][0],
    results['distances'][0]
)):
    print(f"\n{i+1}. 文档: {doc}")
    print(f"   来源: {metadata['source']}")
    print(f"   距离: {distance:.4f}")  # 数值越小越相似
```

## 元数据过滤

```python
# 5. 带过滤条件的搜索
results = collection.query(
    query_texts=["什么是向量数据库？"],
    n_results=5,
    where={"topic": "Database"}  # 只搜索 topic 为 Database 的文档
)

print("过滤后的结果（仅 Database 相关）:")
for doc in results['documents'][0]:
    print(f"  - {doc}")
```

## 更新和删除

```python
# 更新文档
collection.update(
    ids=["doc_1"],
    documents=["RAG 是 Retrieval-Augmented Generation 的缩写，2020 年由 Lewis 等人提出"],
    metadatas=[{"source": "wiki", "topic": "RAG", "updated": "2024"}]
)

# 删除文档
collection.delete(ids=["doc_5"])
print(f"删除后集合数量: {collection.count()}")

# 删除集合
# client.delete_collection("my_knowledge_base")
```

## 使用自定义 Embedding 函数

> ChromaDB 默认使用 Sentence Transformers 的 all-MiniLM-L6-v2，这是一个英文模型，对中文的语义理解效果很差。此外，不同 Embedding 模型的维度、精度、语种各有优劣，生产中通常根据场景选择更合适的模型，这就需要在 ChromaDB 中接入自定义的 Embedding 函数。

```python
# 默认 ChromaDB 使用 Sentence Transformers（英文模型，中文效果差）
# 自定义 Embedding 函数，改用 SiliconFlow 的 BAAI/bge-m3（中文更强）：
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
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [data.embedding for data in response.data]

# 使用自定义 Embedding
custom_client = chromadb.PersistentClient(path="./chroma_db_custom")
custom_collection = custom_client.get_or_create_collection(
    name="siliconflow_kb",
    embedding_function=SiliconFlowEmbeddingFunction()  # 使用 BAAI/bge-m3
)
```

## 动手实验

1. 创建 20 条以上的知识文档（关于 AI、编程、科技）
2. 为每条文档添加有意义的元数据（分类、来源、时间）
3. 测试不同查询的搜索效果，观察距离数值
4. 尝试使用 BAAI/bge-m3 替代默认的 Sentence Transformer Embedding

## ChromaDB vs 其他向量数据库

| 特性 | ChromaDB | FAISS | Milvus | Qdrant |
|------|---------|-------|--------|--------|
| 部署难度 | ⭐（pip install） | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 适合场景 | 学习/原型/小规模 | 大规模纯检索 | 生产环境 | 生产环境 |
| 持久化 | 本地文件 | 内存/文件 | 分布式 | 本地/Docker |
| 过滤功能 | 基础 | 无 | 高级 | 高级 |

## 完成标准
- [ ] 安装 ChromaDB 并成功创建持久化数据库
- [ ] 能添加文档、执行搜索、过滤结果
- [ ] 理解集合、文档、元数据的概念
- [ ] 能使用自定义 Embedding 函数

## 下一步 → P3-文档切分策略.md
