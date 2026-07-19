import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from common.rag_client import BaseRAG, load_directory
from dotenv import load_dotenv
load_dotenv()

# 初始化 RAG（从 rag_knowledge 目录加载文档）
_BASE = Path(__file__).parent.parent.parent.parent
data_dir = str(_BASE / "common" / "text_data" / "rag_knowledge")
db_path = str(Path(__file__).parent / "chroma_db_p1")

rag = BaseRAG(persist_dir=db_path, collection_name="context_window_demo")
if rag.store.count() == 0:
    loaded = load_directory(data_dir)
    rag.add_documents(
        documents=[d["content"] for d in loaded],
        metadatas=[{"source": "rag_knowledge", "topic": d.get("topic", "general")} for d in loaded],
        ids=[f"doc_{i+1}" for i in range(len(loaded))]
    )
    print(f"知识库初始化完成，已添加 {rag.store.count()} 条文档")

def threshold_filter(results: dict, min_score: float = None, max_distance: float = None) -> dict:
    """根据分数或距离过滤结果"""
    if not results['documents'][0]:
        return results
    
    filtered = {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
    
    for i, (doc, meta, dist, doc_id) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0],
        results['ids'][0]
    )):
        keep = True
        if max_distance is not None and dist > max_distance:
            keep = False
        if keep:
            filtered['documents'][0].append(doc)
            filtered['metadatas'][0].append(meta)
            filtered['distances'][0].append(dist)
            filtered['ids'][0].append(doc_id)
    
    return filtered

# 使用：只保留距离 < 0.5 的结果
raw_results = rag.store.search("RAG 是什么？", n_results=10)
filtered = threshold_filter(raw_results, max_distance=0.5)
print(f"过滤前: {len(raw_results['documents'][0])} 条")
print(f"过滤后: {len(filtered['documents'][0])} 条")

# === Code Block 2 ===

def dynamic_top_k(results: dict, min_docs: int = 1, max_docs: int = 10, distance_gap: float = 0.1) -> dict:
    """
    动态选择结果数量：
    - 最少返回 min_docs 条
    - 最多返回 max_docs 条
    - 如果下一个结果和上一个的距离差大于 gap，则截断
    """
    if not results['documents'][0]:
        return results
    
    docs = results['documents'][0]
    distances = results['distances'][0]
    
    # 始终保留至少 min_docs 条
    keep_count = min_docs
    
    # 检查距离突变
    for i in range(min_docs, min(len(distances), max_docs)):
        if i >= 1 and distances[i] - distances[i-1] > distance_gap:
            # 距离突变，说明后面的都不相关了
            break
        keep_count = i + 1
    
    return {
        "documents": [docs[:keep_count]],
        "metadatas": [results['metadatas'][0][:keep_count]],
        "distances": [distances[:keep_count]],
        "ids": [results['ids'][0][:keep_count]]
    }

# 测试
raw_results = rag.store.search("RAG 是什么？", n_results=10)
dynamic = dynamic_top_k(raw_results, min_docs=2, max_docs=10, distance_gap=0.15)
print(f"原始 10 条距离: {[f'{d:.4f}' for d in raw_results['distances'][0]]}")
print(f"动态选择 {len(dynamic['documents'][0])} 条")

# === Code Block 3 ===

def relative_distance_filter(results: dict, ratio_threshold: float = 2.0) -> dict:
    """
    仅保留距离 <= 最佳距离 * ratio_threshold 的结果
    例如：最佳距离 0.2，ratio=2.0 → 保留距离 <= 0.4 的所有结果
    """
    if not results['documents'][0]:
        return results
    
    best_distance = results['distances'][0][0]
    cutoff = best_distance * ratio_threshold
    
    filtered = {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
    
    for doc, meta, dist, doc_id in zip(
        results['documents'][0], results['metadatas'][0],
        results['distances'][0], results['ids'][0]
    ):
        if dist <= cutoff:
            filtered['documents'][0].append(doc)
            filtered['metadatas'][0].append(meta)
            filtered['distances'][0].append(dist)
            filtered['ids'][0].append(doc_id)
    
    return filtered


# 写入结果文件
_output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write(f"阈值过滤结果: {filtered}\n动态Top-K结果: {dynamic}")
print(f"结果已写入 {_output_file}")