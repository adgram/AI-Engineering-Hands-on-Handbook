# 知识库RAG | P1：Embedding 模型与语义向量化

## 前言

大模型虽然能力强大，但它的知识来自训练数据，存在时效性差、不可追溯、容易"幻觉"等问题。**检索增强生成（Retrieval-Augmented Generation, RAG）** 通过先从知识库中检索相关文档，再让模型基于真实资料生成回答，有效缓解了这些问题。

RAG 的核心基础设施是**向量数据库**与 **Embedding 模型**——Embedding 负责将文本转化为语义向量，向量数据库负责高效检索。本节是 RAG 部分的第一节，从 Embedding 原理出发，理解如何让机器"读懂"语义。

本系列是《AI应用工程实战——Prompt / RAG / Agent》的学习笔记，RAG 部分的完整教程和演示代码详见：https://github.com/adgram/AI-Engineering-Hands-on-Handbook。

## 什么是 Embedding

Embedding 是将文本（词/句/段落）映射为固定长度的向量（浮点数数组）的技术。语义相近的文本在向量空间中距离更近。

```
"猫" → [0.12, -0.34, 0.56, ...]  (1024维)
"狗" → [0.15, -0.30, 0.52, ...]  (距离近，语义相似)
"编程" → [-0.45, 0.78, -0.12, ...] (距离远，语义不相关)
```

相比传统关键词匹配，Embedding 实现的是**语义搜索**——搜"轿车"能找到"汽车"，搜"如何部署应用"能找到"上线流程"。这是 RAG 能"理解"用户问题的基石。

## 调用 Embedding API

这里使用 SiliconFlow 提供的 OpenAI 兼容 API 接入 BAAI（智源研究院）的 `bge-m3` 模型，它是目前中文+英文均表现优秀的多语言 Embedding 模型：

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

response = client.embeddings.create(
    model="BAAI/bge-m3",
    input="深度学习是机器学习的一个分支"
)

embedding = response.data[0].embedding
print(f"向量维度: {len(embedding)}")
print(f"前 10 个值: {embedding[:10]}")
print(f"总 Token 消耗: {response.usage.total_tokens}")
```

返回的数据如下：

```python
CreateEmbeddingResponse(
    data=[
            Embedding(
                embedding=[
                        -0.018415380269289017, -0.06950321048498154, ......
                    ],
                index=0,
                object='embedding'
            )
    ],
    model='BAAI/bge-m3',
    object='list',
    usage=Usage(
        prompt_tokens=11,
        total_tokens=11,
        completion_tokens=0
    )
)
```

其中 `CreateEmbeddingResponse` 的核心字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `data[].embedding` | `list[float]` | 文本对应的向量数组（维度由模型决定） |
| `data[].index` | `int` | 输入文本在请求中的索引位置 |
| `model` | `str` | 实际使用的 Embedding 模型名称 |
| `usage.prompt_tokens` | `int` | 输入消耗的 Token 数 |
| `usage.total_tokens` | `int` | 总 Token 数（Embedding 无输出 Token） |

## 计算语义相似度

文本被向量化后，就可以用向量距离衡量语义接近程度。RAG 检索中最常用的是**余弦相似度**——夹角越小（值越接近 1）语义越相近，夹角越大（值越接近 0）语义越无关。

```python
import numpy as np

def get_embedding(text: str) -> list:
    response = client.embeddings.create(model="BAAI/bge-m3", input=text)
    return response.data[0].embedding

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

texts = ["我爱编程", "我喜欢写代码", "今天天气真好", "Python 是一种编程语言", "早上出门记得带伞"]
embeddings = [get_embedding(t) for t in texts]

# "我爱编程" vs 其他四句的相似度
for t, e in zip(texts[1:], embeddings[1:]):
    sim = cosine_similarity(embeddings[0], e)
    print(f"{sim:.2f}  我爱编程  vs  {t}")
```

可以看到"我爱编程"与"我喜欢写代码"相似度最高，与"今天天气真好"相似度最低——这就是语义理解的力量。

## 主流 Embedding 模型对比

| 模型 | 维度 | 适用语言 | API 来源 | 特点 |
|------|------|---------|---------|------|
| BAAI/bge-m3 | 1024 | 多语言 | SiliconFlow | **中文+英文均优，本系列推荐** |
| text-embedding-3-small | 1536 | 多语言 | OpenAI | 性价比高，生态成熟 |
| text-embedding-3-large | 3072 | 多语言 | OpenAI | 效果最好，维度可压缩 |
| BAAI/bge-large-zh-v1.5 | 1024 | 中文为主 | SiliconFlow / 本地 | 纯中文场景优秀 |
| Qwen/Qwen3-Embedding-0.6B | 4096（可调） | 多语言 | SiliconFlow | 支持动态维度 [64..4096] |

### 中文 RAG 的特殊处理

中文不像英文有空格分隔，且一词多义更常见。实践中需要注意：

1. **分词**：用 Jieba 等中文分词器预处理，并为 AI 领域专业词添加自定义词典（如"大语言模型""检索增强生成""向量数据库"）
2. **切分边界**：按中文标点（。！？；，）切分，避免在虚词处截断
3. **模型选择**：英文模型对中文效果差，务必选中文特化或多语言模型

BGE 系列模型在使用时需要加指令前缀，例如 `bge-large-zh-v1.5` 查询时需拼接"为这个句子生成表示以用于检索相关文章："，这是它训练时的约定。

## 批量 Embedding

模型支持一次传入多个文本，比循环调用单条效率高得多：

```python
texts = ["文本一", "文本二", "文本三", "文本四", "文本五"]
response = client.embeddings.create(model="BAAI/bge-m3", input=texts)

for data in response.data:
    print(f"索引 {data.index}: 维度={len(data.embedding)}")
```

批量处理时建议单批不超过 64 条，避免请求超时或触发 API 限流。
