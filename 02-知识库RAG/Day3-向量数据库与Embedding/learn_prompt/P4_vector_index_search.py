from pathlib import Path
import json, numpy as np, os, re
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import Documents, EmbeddingFunction
import chromadb

load_dotenv()

_DIR = Path(__file__).parent
_output_file = str(_DIR / f"{Path(__file__).stem}_result.txt")

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def dot_product(a, b):
    return float(np.dot(np.array(a), np.array(b)))

def euclidean_distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

# --- 加载数据 ---
_data_dir = _DIR.parent.parent.parent / "common/text_data"
with open(_data_dir / "excerpts.json", "r", encoding="utf-8") as f:
    raw = json.load(f)
excerpts = raw["excerpts"]
texts = [e["content"] for e in excerpts if e.get("content")]
metas = [{"source": e.get("source",""), "author": e.get("author",""), "topic": (e.get("tags") or ["other"])[0]} for e in excerpts if e.get("content")]
ids = [f"doc_{i}" for i in range(len(texts))]

with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write(f"数据源: excerpts.json\n条目数: {len(texts)}\n\n")

# --- Embedding API ---
_EMBED_API_KEY = os.getenv("SILICONFLOW_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
_embed_client = OpenAI(api_key=_EMBED_API_KEY, base_url="https://api.siliconflow.cn/v1")
_EMBED_MODEL = "BAAI/bge-m3"

def get_embedding(text: str) -> list:
    resp = _embed_client.embeddings.create(model=_EMBED_MODEL, input=text)
    return resp.data[0].embedding

# --- 相似度比较（纯向量计算） ---
sample_texts = texts[:10]
sample_vecs = [get_embedding(t) for t in sample_texts]

out = "余弦相似度矩阵 (前10条):\n"
for i, t1 in enumerate(sample_texts):
    row = f"[{i}] {t1[:20]:20s}"
    for j in range(len(sample_texts)):
        row += f" {cosine_similarity(sample_vecs[i], sample_vecs[j]):.3f}"
    out += row + "\n"

with open(_output_file, "a", encoding="utf-8") as _f:
    _f.write(out + "\n")

# --- ChromaDB 向量搜索（自带持久化缓存） ---
class _EF(EmbeddingFunction):
    def __call__(self, input: Documents) -> list:
        return [get_embedding(t) for t in (input if isinstance(input, list) else [input])]

_DB_NAME = "excerpts"
client = chromadb.PersistentClient(path=str(_DIR / f"chroma_db_{_DB_NAME}"))
col_name = _DB_NAME
existing = client.get_collection(name=col_name, embedding_function=_EF()) if col_name in [c.name for c in client.list_collections()] else None

if existing is None:
    collection = client.create_collection(name=col_name, embedding_function=_EF(), metadata={"hnsw:space": "cosine"})
    batch_size = 5
    for i in range(0, len(texts), batch_size):
        collection.add(
            documents=texts[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
    print(f"ChromaDB: {collection.count()} 条文档已索引\n")
else:
    collection = existing
    print(f"ChromaDB: 复用已有集合 ({collection.count()} 条)\n")

with open(_output_file, "a", encoding="utf-8") as _f:
    _f.write(f"ChromaDB: {collection.count()} 条文档\n\n")

# --- 搜索评估 ---
queries = ["人的重要性", "关于文学与诗歌", "励志名言", "哲学思考"]
for q in queries:
    result = collection.query(query_texts=[q], n_results=3)
    out = f"查询: {q}\n"
    for doc, meta, dist in zip(result['documents'][0], result['metadatas'][0], result['distances'][0]):
        out += f"  [{dist:.4f}] {doc[:50]}...\n"
    out += "\n"
    with open(_output_file, "a", encoding="utf-8") as _f:
        _f.write(out)

# ===== 多文件合并：同一 ChromaDB 下多个知识库源 =====
_MERGE_DB = "merged"
merge_client = chromadb.PersistentClient(path=str(_DIR / f"chroma_db_{_MERGE_DB}"))

def _chunk_text(text, source, max_len=500):
    chunks, buf = [], ""
    for line in text.split("\n"):
        if len(buf) + len(line) >= max_len:
            if buf.strip():
                chunks.append(buf.strip())
            buf = line
        else:
            buf += "\n" + line if buf else line
    if buf.strip():
        chunks.append(buf.strip())
    return chunks

all_docs, all_metas, all_ids = [], [], []
counter = 0

# 1) excerpts.json — 每条独立
for e in excerpts:
    c = e.get("content", "").strip()
    if c:
        all_docs.append(c)
        all_metas.append({"source": "excerpts", "file": "excerpts.json", "topic": (e.get("tags") or ["other"])[0]})
        all_ids.append(f"s{counter}")
        counter += 1

# 2) 民用建筑设计统一标准.md — 按 ## 标题分块
with open(_data_dir / "民用建筑设计统一标准.md", "r", encoding="utf-8") as f:
    md_text = f.read()
md_chunks = re.split(r'\n(?=## )', md_text)
for chunk in md_chunks:
    chunks = _chunk_text(chunk, "arch_std", 600)
    for c in chunks:
        all_docs.append(c)
        all_metas.append({"source": "arch_std", "file": "民用建筑设计统一标准.md"})
        all_ids.append(f"s{counter}")
        counter += 1

# 3) enum.py — 按 class / def 分块
with open(_data_dir / "enum.py", "r", encoding="utf-8") as f:
    py_text = f.read()
py_chunks = re.split(r'\n(?=(class |def |# ))', py_text)
for chunk in py_chunks:
    if len(chunk.strip()) < 20:
        continue
    sub_chunks = _chunk_text(chunk, "enum_py", 400)
    for c in sub_chunks:
        all_docs.append(c)
        all_metas.append({"source": "enum_code", "file": "enum.py"})
        all_ids.append(f"s{counter}")
        counter += 1

mcol_name = "all_sources"
m_existing = merge_client.get_collection(name=mcol_name, embedding_function=_EF()) if mcol_name in [c.name for c in merge_client.list_collections()] else None

if m_existing is None:
    mcol = merge_client.create_collection(name=mcol_name, embedding_function=_EF())
    for i in range(0, len(all_docs), 10):
        mcol.add(documents=all_docs[i:i+10], metadatas=all_metas[i:i+10], ids=all_ids[i:i+10])
    print(f"\n多文件合并: {mcol.count()} 条 (excerpts:{sum(1 for m in all_metas if m['source']=='excerpts')}, arch_std:{sum(1 for m in all_metas if m['source']=='arch_std')}, enum:{sum(1 for m in all_metas if m['source']=='enum_code')})")
else:
    mcol = m_existing
    print(f"\n多文件合并: 复用已有集合 ({mcol.count()} 条)")

out = "\n=== 多文件合并 — 按源过滤查询 ===\n"
for filt_source, q in [("excerpts", "人生哲理"), ("arch_std", "建筑设计要求"), ("enum_code", "枚举类")]:
    r = mcol.query(query_texts=[q], n_results=2, where={"source": filt_source})
    out += f"查询「{q}」限定 source={filt_source}\n"
    for doc, meta in zip(r['documents'][0], r['metadatas'][0]):
        out += f"  [{meta['source']}] {doc[:40]}...\n"
    out += "\n"

with open(_output_file, "a", encoding="utf-8") as _f:
    _f.write(out)

print(out)
print(f"结果已写入 {_output_file}")
