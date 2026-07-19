"""P3: 检索质量评估 — Hit Rate / MRR / NDCG"""

import json, numpy as np, sys
from pathlib import Path
_BASE = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_BASE))
from common.rag_base import BaseRAG
from dotenv import load_dotenv

load_dotenv()


class SearchEvaluator:
    def __init__(self, store):
        self.store = store

    def hit_rate(self, queries_expected: list, k: int = 5) -> float:
        hits = 0
        for query, expected_ids in queries_expected:
            results = self.store.search(query, n_results=k)
            retrieved = results['ids'][0]
            if any(eid in retrieved for eid in expected_ids):
                hits += 1
        return hits / len(queries_expected)

    def mrr(self, queries_expected: list, k: int = 5) -> float:
        reciprocal_ranks = []
        for query, expected_ids in queries_expected:
            results = self.store.search(query, n_results=k)
            retrieved = results['ids'][0]
            found = False
            for rank, rid in enumerate(retrieved, 1):
                if rid in expected_ids:
                    reciprocal_ranks.append(1.0 / rank)
                    found = True
                    break
            if not found:
                reciprocal_ranks.append(0.0)
        return float(np.mean(reciprocal_ranks))

    def ndcg(self, queries_with_grades: list, k: int = 5) -> float:
        scores = []
        for query, id2grade in queries_with_grades:
            results = self.store.search(query, n_results=k)
            retrieved = results['ids'][0]

            dcg = 0
            for i, rid in enumerate(retrieved, 1):
                grade = id2grade.get(rid, 0)
                dcg += (2**grade - 1) / np.log2(i + 1)

            ideal_grades = sorted(id2grade.values(), reverse=True)[:k]
            idcg = sum((2**g - 1) / np.log2(i + 1) for i, g in enumerate(ideal_grades, 1))

            scores.append(dcg / idcg if idcg > 0 else 0)

        return float(np.mean(scores))

    def full_report(self, test_data: dict) -> dict:
        report = {}
        for k in [1, 3, 5, 10]:
            report[f"k={k}"] = {
                "hit_rate": float(self.hit_rate(test_data["binary"], k)),
                "mrr": float(self.mrr(test_data["binary"], k)),
            }
        return report


def find_best_params(store, test_data):
    from itertools import product

    best_score = 0
    best_params = None

    for chunk_size, overlap, k in product([200, 500, 1000], [0, 50, 100], [3, 5, 10]):
        evaluator = SearchEvaluator(store)
        hr = evaluator.hit_rate(test_data, k=k)
        if hr > best_score:
            best_score = hr
            best_params = (chunk_size, overlap, k)

    return {"best_params": best_params, "best_score": best_score}


if __name__ == "__main__":
    import shutil

    db_path = Path(__file__).parent / "chroma_db_eval"
    if db_path.exists():
        shutil.rmtree(db_path)

    rag = BaseRAG(
        persist_dir=str(db_path),
        collection_name="eval_demo",
    )

    if rag.store.count() == 0:
        from common.rag_base import load_directory
        data_dir = str(_BASE / "common" / "text_data" / "rag_knowledge")
        loaded = load_directory(data_dir)
        docs = [d["content"] for d in loaded]
        doc_ids = [f"doc_{i+1}" for i in range(len(docs))]
        metas = [{"source": "rag_knowledge", "topic": d.get("topic", "general")} for d in loaded]
        rag.add_documents(docs, metadatas=metas, ids=doc_ids)
        print(f"知识库初始化完成，共 {rag.store.count()} 条文档")

    # 测试查询——使用非直接匹配的表述，避免关键词完全重叠
    all_ids = rag.store.collection.get()["ids"]
    test_binary = [
        ("如何让 AI 结合外部知识来回答问题？", ["doc_1", "doc_2"]),
        ("怎样高效存储和检索向量数据？", ["doc_3"]),
        ("设计好的提示词有哪些方法论？", ["doc_4"]),
        ("多个 AI 代理如何协同完成复杂任务？", ["doc_5"]),
        ("检索后如何提升结果排序质量？", ["doc_6"]),
        ("文档切分对检索质量有什么影响？", ["doc_7"]),
        ("如何评估 RAG 系统的质量？", ["doc_8"]),
    ]

    evaluator = SearchEvaluator(rag.store)
    report = evaluator.full_report({"binary": test_binary})

    print("评估报告:")
    for k, metrics in report.items():
        print(f"  {k}: HR={metrics['hit_rate']:.3f}, MRR={metrics['mrr']:.3f}")

    _output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.json"
    with open(_output_file, "w", encoding="utf-8") as _f:
        json.dump(report, _f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {_output_file}")