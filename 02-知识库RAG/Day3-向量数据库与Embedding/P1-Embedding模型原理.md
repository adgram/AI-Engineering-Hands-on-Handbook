# P1：Embedding 模型原理

> **检索增强生成（Retrieval-Augmented Generation, RAG）** 是一种将外部知识库检索与大语言模型生成能力相结合的架构。它先通过语义检索从知识库中找到与用户问题相关的文档片段，再将这些片段作为上下文注入提示词，让模型基于真实资料生成回答，从而有效缓解大模型的"幻觉"问题。RAG 的核心基础设施是向量数据库与 Embedding 模型——Embedding 负责将文本转化为语义向量，向量数据库负责高效检索。本部分将从 Embedding 原理出发，逐步构建一个完整的 RAG 知识库问答系统。

## 目标
理解文本向量化的原理，掌握 Embedding API 的调用

## 前置准备

在 `.env` 文件中添加 SiliconFlow API Key（使用 SiliconFlow 的 BAAI/bge-m3）：

```ini
SILICONFLOW_API_KEY=你的SiliconFlow密钥
```

```bash
pip install openai python-dotenv numpy
```

## 什么是 Embedding？

Embedding 是将文本（词/句/段落）映射为固定长度的向量（浮点数数组）的技术。语义相近的文本在向量空间中距离更近。

```
"猫" → [0.12, -0.34, 0.56, ...]  (768维)
"狗" → [0.15, -0.30, 0.52, ...]  (距离近，语义相似)
"编程" → [-0.45, 0.78, -0.12, ...] (距离远，语义不相关)
```

## 为什么需要 Embedding？

| 场景 | 传统方法 | Embedding 方法 |
|------|---------|---------------|
| 搜索 | 关键词匹配（搜不到同义词） | 语义搜索（"轿车"能搜到"汽车"） |
| 聚类 | 需要人工特征 | 自动语义聚类 |
| 推荐 | 标签匹配 | 语义相似度推荐 |

## 调用 Embedding API（SiliconFlow + BAAI/bge-m3）

这里使用 SiliconFlow 提供的 OpenAI 兼容 API 接入 BAAI（智源研究院）的 BAAI/bge-m3 模型：

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),           # SiliconFlow API Key
    base_url="https://api.siliconflow.cn/v1"            # SiliconFlow API 地址
)

response = client.embeddings.create(
    model="BAAI/bge-m3",                                # 多语言 Embedding 模型
    input="深度学习是机器学习的一个分支"
)

embedding = response.data[0].embedding
print(f"向量维度: {len(embedding)}")
print(f"前 10 个值: {embedding[:10]}")
print(f"总 Token 消耗: {response.usage.total_tokens}")
```

## 返回数据解析

client.embeddings.create 返回数据如下：

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

上面打印的 `response` 是一个 `CreateEmbeddingResponse` 对象，核心字段如下：

| 字段                 | 类型          | 说明                                   |
| -------------------- | ------------- | -------------------------------------- |
| `data[].embedding`   | `list[float]` | 文本对应的向量数组（维度由模型决定）   |
| `data[].index`       | `int`         | 输入文本在请求中的索引位置             |
| `model`              | `str`         | 实际使用的 Embedding 模型名称          |
| `object`             | `str`         | 返回类型，固定为 `list`                |
| `usage.prompt_tokens`| `int`         | 输入消耗的 Token 数                    |
| `usage.total_tokens` | `int`         | 总 Token 数（Embedding 无输出 Token）  |

## 计算语义相似度

**语义相似度**衡量两段文本在含义上的接近程度。基于 Embedding 的语义相似度则将文本映射到向量空间，用向量来衡量语义距离，这里使用余弦相似度进行演示。相似度的深入理解，详见P4部分。

- 夹角越小（余弦值接近 1）→ 语义越相近
- 夹角越大（余弦值接近 0 或负数）→ 语义越无关

余弦相似度公式：`cos(a, b) = (a·b) / (|a|·|b|)`

```python
import numpy as np

def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        model="BAAI/bge-m3",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: list, b: list) -> float:
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 测试语义相似度
texts = [
    "我爱编程",
    "我喜欢写代码",
    "今天天气真好",
    "Python 是一种编程语言",
    "早上出门记得带伞"
]

embeddings = [get_embedding(t) for t in texts]

print("语义相似度矩阵:")
print(f"{'':20}", end="")
for t in texts:
    print(f"{t[:10]:10}", end="")
print()

for i, t1 in enumerate(texts):
    print(f"{t1[:20]:20}", end="")
    for j in range(len(texts)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{sim:.2f}      ", end="")
    print()
```

## 常见 Embedding 模型对比

| 模型 | 维度 | 适用语言 | API 来源 | 特点 |
|------|------|---------|---------|------|
| BAAI/bge-m3 | 1024 | 多语言（中文最强） | SiliconFlow | **本教程推荐，中文+英文均优** |
| text-embedding-3-small | 1536 | 多语言 | OpenAI | 性价比高，生态成熟 |
| text-embedding-3-large | 3072 | 多语言 | OpenAI | 效果最好，维度可压缩 |
| BAAI/bge-large-zh-v1.5 | 1024 | 中文为主 | SiliconFlow / 本地 | 纯中文场景优秀 |
| m3e-base | 768 | 中文为主 | 本地运行 | 轻量级中文 |

## 批量 Embedding

模型支持一次传入多个文本进行向量计算：

```python
# 一次传入多个文本（更高效）
texts = ["文本一", "文本二", "文本三", "文本四", "文本五"]
response = client.embeddings.create(
    model="BAAI/bge-m3",
    input=texts
)

for data in response.data:
    print(f"索引 {data.index}: 维度={len(data.embedding)}")
```

## 工业实践：向量精度与维度控制

| 场景 | 推荐维度 | 说明 |
|------|---------|------|
| 通用场景 | 768 | 准确率与存储成本最优平衡 |
| 高精度（法律/金融） | 1024 | 需要精细区分语义 |
| 边缘端低资源 | 512 | 牺牲精度换速度 |

| 精度方案 | 存储 | 速度 | 准确率损失 |
|---------|------|------|-----------|
| fp32 | 1x | 1x | 0% |
| fp16 | 0.5x | 1.3x | <2% |
| int8 | 0.25x | 2x | <5%（需校准） |

**经验法则**：768维 + fp16 是生产中最常见的组合。

## 动手实验

1. 计算以下句子对之间的相似度，排序：
   - "苹果发布了新款手机" vs "华为推出折叠屏"
   - "苹果发布了新款手机" vs "今天天气很好"
   - "苹果发布了新款手机" vs "iPhone 16 性能提升"
2. 找出和"人工智能"语义最相近的 3 个词
3. 对比 BAAI/bge-m3 和 text-embedding-3-small（如果可用）的中文语义效果
4. 测试不同向量维度（384 vs 768 vs 1024）对相似度排序的影响

## 完成标准
- [ ] 理解 Embedding 的基本原理
- [ ] 能调用 Embedding API 获取文本向量
- [ ] 实现了余弦相似度计算
- [ ] 完成语义相似度对比实验
- [ ] 了解向量维度选择和精度压缩的工业实践

## 下一步 → [P2-向量数据库ChromaDB](P2-向量数据库ChromaDB.md)

