import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

_output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

response = client.embeddings.create(
    model="BAAI/bge-m3",                                # 多语言 Embedding 模型
    input="深度学习是机器学习的一个分支"
)

with open(_output_file, "w", encoding="utf-8") as _f:
    embedding = response.data[0].embedding
    _f.write(f"向量维度: {len(embedding)}\n")
    _f.write(f"前 10 个值: {embedding[:10]}\n")
    _f.write(f"总 Token 消耗: {response.usage.total_tokens}\n")

# === Code Block 2 ===

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

result_matrix = "=== Code Block 2 ===\n\n语义相似度矩阵:\n"

result_matrix += f"{'':20}"
for t in texts:
    result_matrix += f"{t[:10]:10}"
result_matrix += "\n"

for i, t1 in enumerate(texts):
    result_matrix += f"{t1[:20]:20}"
    for j in range(len(texts)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        result_matrix += f"{sim:.2f}      "
    result_matrix += "\n"

with open(_output_file, "a", encoding="utf-8") as _f:
    _f.write(result_matrix)


# === Code Block 3 ===

# 一次传入多个文本（更高效）
texts = ["文本一", "文本二", "文本三", "文本四", "文本五"]
response = client.embeddings.create(
    model="BAAI/bge-m3",
    input=texts
)

result_text = "=== Code Block 3 ===\n\n"

for data in response.data:
    result_text += f"索引 {data.index}: 维度={len(data.embedding)}\n"

with open(_output_file, "a", encoding="utf-8") as _f:
    _f.write(result_text)
print(f"结果已写入 {_output_file}")
