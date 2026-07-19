from pathlib import Path
import json, os, jieba, jieba.analyse, chromadb
from chromadb import Documents, EmbeddingFunction
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_DIR = Path(__file__).parent
_output_file = str(_DIR / f"{Path(__file__).stem}_result.txt")
_DB_NAME = "excerpts"

# --- Embedding API ---
_EMBED_API_KEY = os.getenv("SILICONFLOW_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
_EMBED_BASE = "https://api.siliconflow.cn/v1"

# --- 加载数据 ---
_data_dir = _DIR.parent.parent.parent / "common" / "text_data"
with open(_data_dir / "excerpts.json", "r", encoding="utf-8") as f:
    raw = json.load(f)
excerpts = raw["excerpts"][:20]
texts = [e["content"] for e in excerpts if e.get("content")]

with open(_output_file, "w", encoding="utf-8") as f:
    f.write(f"数据: excerpts.json ({len(texts)}条)\n\n")

# --- 中文分词 ---
out = "--- jieba 分词 ---\n"
sample = texts[0]
words = jieba.lcut(sample)
out += f"原文: {sample[:40]}...\n"
out += f"精确模式: {words[:15]}\n"
keywords = jieba.analyse.extract_tags(sample, topK=5, withWeight=True)
out += "关键词:\n"
for w, weight in keywords:
    out += f"  {w}: {weight:.4f}\n"
out += "\n"

with open(_output_file, "a", encoding="utf-8") as f:
    f.write(out)

# --- Embedding 模型定义 ---
class SiliconFlowEF(EmbeddingFunction):
    def __init__(self, model="BAAI/bge-m3", dimensions=None, query_prefix=""):
        self.model = model
        self.dimensions = dimensions
        self.query_prefix = query_prefix
        self.client = OpenAI(api_key=_EMBED_API_KEY, base_url=_EMBED_BASE)

    def __call__(self, input: Documents) -> list:
        inputs = [self.query_prefix + t for t in (input if isinstance(input, list) else [input])]
        kwargs = dict(model=self.model, input=inputs)
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        resp = self.client.embeddings.create(**kwargs)
        return [d.embedding for d in resp.data]

models = {
    "BGE-m3": SiliconFlowEF("BAAI/bge-m3"),
    "BGE-large-zh": SiliconFlowEF("BAAI/bge-large-zh-v1.5", query_prefix="为这个句子生成表示以用于检索相关文章："),

    "Qwen3-Embedding-0.6B": SiliconFlowEF("Qwen/Qwen3-Embedding-0.6B", dimensions=1024),
}

# --- 模型对比（共用 excerpts 库下的不同 collection） ---
out = "--- Embedding 模型对比 ---\n"
client = chromadb.PersistentClient(path=str(_DIR / f"chroma_db_{_DB_NAME}"))

test_queries = ["人生哲理", "文学与诗歌", "励志名言"]

for name, emb_fn in models.items():
    try:
        col_name = name.replace("(","").replace(")","").replace(".","_").replace(" ","_")
        existing = [c.name for c in client.list_collections()]
        col = client.get_collection(name=col_name, embedding_function=emb_fn) if col_name in existing else None

        if col is None:
            col = client.create_collection(name=col_name, embedding_function=emb_fn)
            col.add(documents=texts, ids=[f"doc_{i}" for i in range(len(texts))])

        out += f"\n  [{name}]\n"
        for q_text in test_queries:
            r = col.query(query_texts=[q_text], n_results=1)
            doc_id = r['ids'][0][0]
            doc = r['documents'][0][0]
            out += f"    查询「{q_text}」→ {doc_id}: {doc[:30]}...\n"
    except Exception as e:
        out += f"  [{name}] error: {e}\n"
        

with open(_output_file, "a", encoding="utf-8") as f:
    f.write(out + "\n")

# ===== 多模型接入：同一 chroma_db 下，不同维度共存 =====
out = "--- 多模型接入（同一库不同维度集合） ---\n"
client = chromadb.PersistentClient(path=str(_DIR / f"chroma_db_{_DB_NAME}"))
all_cols = [c.name for c in client.list_collections()]
out += f"chroma_db_{_DB_NAME} 下共有 {len(all_cols)} 个集合: {all_cols}\n\n"

dim_models = {
    "BGE-m3(1024d)": (SiliconFlowEF("BAAI/bge-m3"), 1024),
    "Qwen3-512d": (SiliconFlowEF("Qwen/Qwen3-Embedding-0.6B", dimensions=512), 512),
    "Qwen3-256d": (SiliconFlowEF("Qwen/Qwen3-Embedding-0.6B", dimensions=256), 256),
}

test_texts = ["RAG 是检索增强生成", "向量数据库用于存储和检索高维向量"]
out += "每个模型对同一段文本的向量维度:\n"
for name, (fn, _) in dim_models.items():
    vecs = fn(test_texts)
    out += f"  {name}: {len(vecs[0])}d  前5维: {[f'{v:.4f}' for v in vecs[0][:5]]}\n"
out += "\n"

out += "同一 chroma_db 可共存不同维度的集合:\n"
for name, (fn, dim) in dim_models.items():
    col_name = name.replace("(","").replace(")","").replace("-","_").replace(" ","_")
    existing = [c.name for c in client.list_collections()]
    col = client.get_collection(name=col_name, embedding_function=fn) if col_name in existing else None
    if col is None:
        col = client.create_collection(name=col_name, embedding_function=fn)
        col.add(documents=texts[:10], ids=[f"doc_{i}" for i in range(len(texts[:10]))])
    r = col.query(query_texts=["人生哲理"], n_results=1)
    out += f"  {name} ({dim}d collection): 查询「人生哲理」→ {r['ids'][0][0]}: {r['documents'][0][0][:30]}...\n"

with open(_output_file, "a", encoding="utf-8") as f:
    f.write(out + "\n")
print(f"\n结果已写入 {_output_file}")
