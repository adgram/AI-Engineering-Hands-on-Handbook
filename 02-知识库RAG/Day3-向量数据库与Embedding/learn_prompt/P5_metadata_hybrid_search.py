from pathlib import Path
import json, chromadb, sqlite3, os
from chromadb import Documents, EmbeddingFunction
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_DIR = Path(__file__).parent
_output_file = str(_DIR / f"{Path(__file__).stem}_result.txt")
_DB_NAME = "excerpts"

# --- Embedding API ---
_EMBED_API_KEY = os.getenv("SILICONFLOW_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
_embed_client = OpenAI(api_key=_EMBED_API_KEY, base_url="https://api.siliconflow.cn/v1")
_EMBED_MODEL = "BAAI/bge-m3"

def get_embedding(text: str) -> list:
    resp = _embed_client.embeddings.create(model=_EMBED_MODEL, input=text)
    return resp.data[0].embedding

class _EF(EmbeddingFunction):
    def __call__(self, input: Documents) -> list:
        return [get_embedding(t) for t in (input if isinstance(input, list) else [input])]

# --- 加载数据 ---
_data_dir = _DIR.parent.parent.parent / "common" / "text_data"
with open(_data_dir / "excerpts.json", "r", encoding="utf-8") as f:
    raw = json.load(f)
excerpts = raw["excerpts"]
texts = [e["content"] for e in excerpts if e.get("content")]
topics = []
for e in excerpts:
    tags = e.get("tags", [])
    topics.append(tags[0] if tags else "other")
ids = [f"doc_{i}" for i in range(len(texts))]

with open(_output_file, "w", encoding="utf-8") as f:
    f.write(f"数据: excerpts.json ({len(texts)}条)\n\n")

# --- ChromaDB 元数据过滤（共享 excerpts 库） ---
client = chromadb.PersistentClient(path=str(_DIR / f"chroma_db_{_DB_NAME}"))
existing_names = [c.name for c in client.list_collections()]
col = client.get_collection(name=_DB_NAME, embedding_function=_EF()) if _DB_NAME in existing_names else None

if col is None:
    col = client.create_collection(name=_DB_NAME, embedding_function=_EF())
    col.add(documents=texts, metadatas=[{"topic": t} for t in topics], ids=ids)
    print(f"ChromaDB: {col.count()} 条已索引")
else:
    print(f"ChromaDB: 复用已有集合 ({col.count()} 条)")

out = "--- 元数据过滤 ---\n"
for q_text, topic_filter in [("人生哲理", "哲学散文"), ("文学名句", "古诗词曲"), ("现代诗歌", "现代诗歌")]:
    results = col.query(query_texts=[q_text], n_results=3, where={"topic": topic_filter})
    out += f"查询: {q_text} (topic={topic_filter})\n"
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        out += f"  - {doc[:50]}... [{meta['topic']}]\n"
    out += "\n"

with open(_output_file, "a", encoding="utf-8") as f:
    f.write(out)

# --- 混合检索 ---
class HybridRetriever:
    def __init__(self, col, texts):
        self.col = col
        self.texts = texts
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE kw_index (id TEXT, content TEXT)")
        for i, doc in enumerate(texts):
            self.conn.execute("INSERT INTO kw_index VALUES (?,?)", (f"doc_{i}", doc))

    def hybrid_search(self, query, n_results=5, kw_weight=0.3):
        vec_res = self.col.query(query_texts=[query], n_results=n_results*2)
        kw_res = self.conn.execute("SELECT id FROM kw_index WHERE content LIKE ?", (f"%{query}%",)).fetchall()
        kw_ids = set(r[0] for r in kw_res)
        combined = {}
        for i, doc_id in enumerate(vec_res['ids'][0]):
            combined[doc_id] = (1 - kw_weight) * (1 - i / len(vec_res['ids'][0]))
        for doc_id in kw_ids:
            combined[doc_id] = combined.get(doc_id, 0) + kw_weight
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:n_results]

h = HybridRetriever(col, texts)
out2 = "--- 混合检索 ---\n"
res = h.hybrid_search("人生", n_results=5)
out2 += "查询: 人生\n"
for doc_id, score in res:
    idx = int(doc_id.split("_")[1])
    out2 += f"  {doc_id} score={score:.3f}: {texts[idx][:40]}...\n"

with open(_output_file, "a", encoding="utf-8") as f:
    f.write(out2 + "\n")

# --- RRF 融合 ---
def rrf_fusion(vec_rank, kw_rank, k=60):
    scores = {}
    for rank, doc_id in enumerate(vec_rank, 1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    for rank, doc_id in enumerate(kw_rank, 1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

out3 = "--- RRF 融合 (模拟) ---\n"
for doc_id, score in rrf_fusion(["doc_0","doc_1","doc_3","doc_5","doc_7"], ["doc_3","doc_1","doc_7","doc_10","doc_15"]):
    out3 += f"  {doc_id}: {score:.4f}\n"

with open(_output_file, "a", encoding="utf-8") as f:
    f.write(out3)

print(f"结果已写入 {_output_file}")
