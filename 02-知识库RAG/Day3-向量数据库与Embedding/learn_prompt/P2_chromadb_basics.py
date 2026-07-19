from pathlib import Path
import chromadb, os
from chromadb import Documents, EmbeddingFunction
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 写入结果文件
_output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

# 1. 创建客户端（持久化存储到本地）
client = chromadb.PersistentClient(
    path = Path(__file__).parent / "chroma_db",  # 数据存储目录
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

with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write(f"集合中已有 {collection.count()} 条文档\n")

# === Code Block 2 ===

# 4. 搜索（默认使用集合自带的 Embedding 函数）
results = collection.query(
    query_texts=["什么是 RAG？"],
    n_results=3  # 返回最相似的 3 条
)

query_result = "查询结果:\n"
for i, (doc, metadata, distance) in enumerate(zip(
    results['documents'][0],
    results['metadatas'][0],
    results['distances'][0]
)):
    query_result = f"\n{i+1}. 文档: {doc}\n"
    query_result = f"   来源: {metadata['source']}\n"
    query_result = f"   距离: {distance:.4f}\n"  # 数值越小越相似

with open(_output_file, "a", encoding="utf-8") as _f:
    _f.write(query_result)

# === Code Block 3 ===

# 5. 带过滤条件的搜索
results = collection.query(
    query_texts=["什么是向量数据库？"],
    n_results=5,
    where={"topic": "Database"}  # 只搜索 topic 为 Database 的文档
)

query_result2 = "过滤后的结果（仅 Database 相关）:\n"

for doc in results['documents'][0]:
    query_result2 += f"  - {doc}\n"

with open(_output_file, "a", encoding="utf-8") as _f:
    _f.write(query_result2)

# === Code Block 4 ===

# 更新文档
collection.update(
    ids=["doc_1"],
    documents=["RAG 是 Retrieval-Augmented Generation 的缩写，2020 年由 Lewis 等人提出"],
    metadatas=[{"source": "wiki", "topic": "RAG", "updated": "2024"}]
)

# 删除文档
collection.delete(ids=["doc_5"])

with open(_output_file, "a", encoding="utf-8") as _f:
    _f.write(f"删除后集合数量: {collection.count()}")


# 删除集合
# client.delete_collection("my_knowledge_base")

# === Code Block 5 ===

# 默认 ChromaDB 使用 Sentence Transformers
# 也可以用自定义的 Embedding 函数：

class SiliconFlowEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: str = "BAAI/bge-m3"):
        self.client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
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
custom_client = chromadb.PersistentClient(path = Path(__file__).parent /"chroma_db_custom")
custom_collection = custom_client.get_or_create_collection(
    name="siliconflow_kb",
    embedding_function=SiliconFlowEmbeddingFunction()  # 使用 BAAI/bge-m3
)


with open(_output_file, "a", encoding="utf-8") as _f:
    _f.write(f"删除后集合数量: {collection.count()}")


print(f"结果已写入 {_output_file}")
