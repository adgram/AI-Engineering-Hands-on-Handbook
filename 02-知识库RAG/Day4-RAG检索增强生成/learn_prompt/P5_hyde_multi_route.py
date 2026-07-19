"""P5: 高级检索 — HyDE 与多路召回"""

from collections import defaultdict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from common.rag_base import BaseRAG
from dotenv import load_dotenv

load_dotenv()



def hyde_search(store, query: str, llm, model: str = "deepseek-v4-flash", k: int = 5) -> dict:
    hyde_prompt = f"""基于你的知识，针对以下问题生成一段假设的、理想的 Wikipedia 风格的回答。

问题：{query}

要求：
- 使用客观、事实性的语言
- 假设回答内容在标准百科全书中
- 长度在 100-200 字之间"""

    response = llm.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": hyde_prompt}],
        temperature=0.3,
    )
    hypothetical_doc = response.choices[0].message.content
    print(f"[HyDE] 假设文档:\n{hypothetical_doc[:150]}...\n")

    results = store.search(hypothetical_doc, n_results=k)
    return results


class MultiRouteRetriever:
    def __init__(self, store, llm, model: str = "deepseek-v4-flash"):
        self.store = store
        self.llm = llm
        self.model = model

    def vector_search(self, query: str, k: int = 5) -> list:
        results = self.store.search(query, n_results=k)
        return list(zip(results['ids'][0], results['documents'][0], results['distances'][0]))

    def keyword_search(self, query: str, k: int = 5) -> list:
        results = self.store.search(query, n_results=k * 2)
        return list(zip(results['ids'][0], results['documents'][0], results['distances'][0]))

    def hyde_search(self, query: str, k: int = 5) -> list:
        resp = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"针对问题生成一段百科式的回答：{query}"}],
        )
        hyde_doc = resp.choices[0].message.content
        results = self.store.search(hyde_doc, n_results=k)
        return list(zip(results['ids'][0], results['documents'][0], results['distances'][0]))

    def multi_route_search(self, query: str, k: int = 5) -> dict:
        routes = {
            "vector": self.vector_search(query, k),
            "hyde": self.hyde_search(query, k),
        }

        K = 60
        scores = defaultdict(float)
        route_details = {}

        for route_name, results in routes.items():
            route_details[route_name] = []
            for rank, (doc_id, doc, dist) in enumerate(results, 1):
                scores[doc_id] += 1 / (K + rank)
                route_details[route_name].append({"id": doc_id, "doc": doc[:80], "rank": rank})

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        final_results = []
        for doc_id, score in ranked:
            doc = self.store.collection.get(ids=[doc_id])
            final_results.append({
                "id": doc_id,
                "content": doc['documents'][0] if doc['documents'] else "",
                "score": score,
            })

        return {
            "query": query,
            "routes": route_details,
            "results": final_results,
        }


if __name__ == "__main__":
    import shutil

    db_path = Path(__file__).parent / "chroma_db_hyde"
    if db_path.exists():
        shutil.rmtree(db_path)

    rag = BaseRAG(
        persist_dir=str(db_path),
        collection_name="hyde_demo",
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

    query = "RAG 和传统微调相比有什么优势？"

    print("=== 普通搜索 ===")
    normal_results = rag.store.search(query, n_results=3)
    for doc in normal_results['documents'][0]:
        print(f"  - {doc[:80]}...")

    print("\n=== HyDE 搜索 ===")
    hyde_results = hyde_search(rag.store, query, rag.llm, k=3)
    for doc in hyde_results['documents'][0]:
        print(f"  - {doc[:80]}...")

    print("\n=== 多路召回 ===")
    retriever = MultiRouteRetriever(rag.store, rag.llm)
    result = retriever.multi_route_search(query, k=3)

    print("各路由结果:")
    for route, docs in result["routes"].items():
        print(f"\n  [{route}]")
        for d in docs:
            print(f"    排名{d['rank']}: {d['doc']}")

    print("\n融合后最终结果:")
    for r in result["results"]:
        print(f"  [{r['id']}] 得分={r['score']:.4f}: {r['content'][:80]}...")

    _output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
    with open(_output_file, "w", encoding="utf-8") as _f:
        _f.write(f"查询: {result['query']}\n融合结果: {result['results']}")
    print(f"\n结果已写入 {_output_file}")