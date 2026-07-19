"""P6: Rerank 重排序"""

import json, os, sys
import httpx
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from common.rag_base import BaseRAG
from dotenv import load_dotenv

load_dotenv()


def llm_rerank(query: str, documents: list, llm, model: str = "deepseek-v4-flash") -> list:
    scored = []

    for doc in documents:
        prompt = f"""判断以下文档和问题的相关性。

问题：{query}
文档：{doc}

请输出 0-10 的相关性分数：
- 0-3: 不相关
- 4-6: 部分相关
- 7-8: 相关
- 9-10: 高度相关

只输出一个数字。"""

        resp = llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            score = float(resp.choices[0].message.content.strip())
        except:
            score = 5

        scored.append({"doc": doc, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def api_rerank(query: str, documents: list, top_k: int = 5, model: str = "BAAI/bge-reranker-v2-m3") -> list:
    """通过 SiliconFlow API 调用 Rerank 模型"""
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("SILICONFLOW_API_KEY 未设置，无法使用 API Rerank")

    resp = httpx.post(
        "https://api.siliconflow.cn/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "query": query, "documents": documents, "top_n": top_k},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("results", []):
        idx = item["index"]
        results.append((documents[idx], item["relevance_score"]))
    return results


def evaluate_rerank(store, test_cases: list, k_before=10, k_after=5):
    from P3_retrieval_evaluation import SearchEvaluator

    evaluator = SearchEvaluator(store)

    hr_before = evaluator.hit_rate(test_cases, k=k_before)
    mrr_before = evaluator.mrr(test_cases, k=k_before)

    hits_after = 0
    for query, expected_ids in test_cases:
        initial = store.search(query, n_results=k_before)
        if any(eid in initial['ids'][0][:k_after] for eid in expected_ids):
            hits_after += 1

    hr_after = hits_after / len(test_cases)

    return {
        "before": {"k": k_before, "hit_rate": hr_before, "mrr": mrr_before},
        "after": {"k": k_after, "hit_rate": hr_after},
        "improvement": hr_after - hr_before,
    }


if __name__ == "__main__":
    import shutil

    db_path = Path(__file__).parent / "chroma_db_rerank"
    if db_path.exists():
        shutil.rmtree(db_path)

    rag = BaseRAG(
        persist_dir=str(db_path),
        collection_name="rerank_demo",
    )

    if rag.store.count() == 0:
        from common.rag_base import load_directory
        data_dir = str(Path(__file__).parent.parent.parent.parent / "common" / "text_data" / "rag_knowledge")
        loaded = load_directory(data_dir)
        docs = [d["content"] for d in loaded]
        doc_ids = [f"doc_{i+1}" for i in range(len(docs))]
        metas = [{"source": "rag_knowledge", "topic": d.get("topic", "general")} for d in loaded]
        
        rag.add_documents(
            documents=docs,
            metadatas=metas,
            ids=doc_ids
        )
        print(f"知识库初始化完成，已添加 {rag.store.count()} 条文档")

    query = "RAG 的优缺点是什么？"
    initial_results = rag.store.search(query, n_results=5)
    reranked = llm_rerank(query, initial_results['documents'][0], rag.llm)

    print("Rerank 前:")
    for i, doc in enumerate(initial_results['documents'][0]):
        print(f"  #{i+1}: {doc[:60]}...")

    print("\nRerank 后:")
    for i, item in enumerate(reranked):
        print(f"  #{i+1} (分数={item['score']}): {item['doc'][:60]}...")

    print("\n=== API Rerank (SiliconFlow) ===")
    api_documents = [
        "RAG 是检索增强生成，结合检索和 LLM 生成，可以降低幻觉风险",
        "向量数据库使用余弦相似度计算向量间的相关性",
        "Prompt Engineering 是设计和优化提示词的技术",
        "Rerank 模型采用 Cross-Encoder 架构进行深度语义交互",
        "RAG 不需要重新训练模型，支持知识实时更新",
    ]
    api_reranked = api_rerank("RAG 的优势", api_documents, top_k=3)
    for doc, score in api_reranked:
        print(f"  ({score:.4f}) {doc}")

    _output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.json"
    with open(_output_file, "w", encoding="utf-8") as _f:
        json.dump({
            "llm_rerank": [(item['doc'][:40], item['score']) for item in reranked[:3]],
            "api_rerank": [(d[:40], f"{s:.4f}") for d, s in api_reranked],
        }, _f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {_output_file}")