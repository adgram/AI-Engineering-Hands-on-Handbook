import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from common.llm_client import LLMClient
from common.rag_client.embedding import SiliconFlowEmbedding
client = LLMClient()
import hashlib
import json
import diskcache
import chromadb

class EmbeddingCache:
    def __init__(self, cache_dir: Path):
        self.cache = diskcache.Cache(cache_dir)
    
    def _key(self, text: str, model: str) -> str:
        return hashlib.md5(f"{model}:{text}".encode()).hexdigest()
    
    def get(self, text: str, model: str = "BAAI/bge-m3") -> list:
        key = self._key(text, model)
        return self.cache.get(key)
    
    def set(self, text: str, embedding: list, model: str = "BAAI/bge-m3"):
        key = self._key(text, model)
        self.cache.set(key, embedding)
    
    def get_or_compute(self, text: str, compute_fn, model: str = "BAAI/bge-m3") -> list:
        cached = self.get(text, model)
        if cached is not None:
            return cached
        embedding = compute_fn(text)
        self.set(text, embedding, model)
        return embedding

# 使用
embed_cache = EmbeddingCache(Path(__file__).parent/"cache/embeddings")

_ef = SiliconFlowEmbedding()

def get_embedding(text):
    return _ef([text])[0]

# 第一次：调用 API
emb1 = embed_cache.get_or_compute("RAG 是什么？", get_embedding)
# 第二次：直接返回缓存
emb2 = embed_cache.get_or_compute("RAG 是什么？", get_embedding)
print(f"相同结果: {emb1 == emb2}")

# === Code Block 2 ===

class LLMResponseCache:
    def __init__(self, cache_dir: Path):
        self.cache = diskcache.Cache(cache_dir)
    
    def _key(self, messages: list, model: str, temperature: float) -> str:
        data = json.dumps({"messages": messages, "model": model, "temperature": temperature}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()
    
    def get_or_compute(self, messages: list, compute_fn, model: str = "deepseek-v4-flash", temperature: float = 0) -> str:
        key = self._key(messages, model, temperature)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = compute_fn(messages)
        self.cache.set(key, result)
        return result

# 使用
llm_cache = LLMResponseCache(Path(__file__).parent/"cache/llm")

def call_llm(messages):
    response = client.chat(messages=messages, temperature=0)
    return response.choices[0].message.content

messages = [{"role": "user", "content": "RAG 是什么？"}]

# 第一次：调用 API
ans1 = llm_cache.get_or_compute(messages, call_llm)
# 第二次：直接返回缓存
ans2 = llm_cache.get_or_compute(messages, call_llm)
print(f"缓存命中: {ans1 == ans2}")

# === Code Block 3 ===

class SemanticCache:
    def __init__(self, collection, threshold: float = 0.3):
        self.collection = collection
        self.threshold = threshold
    
    def lookup(self, query: str) -> str:
        results = self.collection.query(query_texts=[query], n_results=1)
        if results['documents'][0] and results['distances'][0][0] < self.threshold:
            return results['metadatas'][0][0].get('answer')
        return None
    
    def store(self, query: str, answer: str):
        self.collection.add(
            documents=[query],
            metadatas=[{"answer": answer}],
            ids=[hashlib.md5(query.encode()).hexdigest()]
        )

# 测试
sem_cache = SemanticCache(
    chromadb.PersistentClient(path=Path(__file__).parent/"chroma_db_semantic").get_or_create_collection("qa_cache")
)

q = "RAG 的优点是什么？"
answer = sem_cache.lookup(q)
if answer:
    print(f"语义缓存命中: {answer}")
else:
    answer = "减少幻觉、支持最新知识、可追溯"
    sem_cache.store(q, answer)
    print(f"已缓存")

# "RAG 有哪些好处？" → 应该命中缓存
similar = "RAG 有哪些好处？"
answer = sem_cache.lookup(similar)
print(f"相似查询{'命中' if answer else '未命中'}: {answer}")

# 写入结果文件
_output_file = str(Path(__file__).parent / f"{Path(__file__).stem}_result.txt")
with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write(f"Embedding缓存一致: {emb1 == emb2}\nLLM缓存命中: {ans1 == ans2}\n语义缓存查询({similar}): {answer}")
print(f"结果已写入 {_output_file}")
