"""入口：CLI 运行实验 / 交互问答 / 网格搜索"""

import json, sys, shutil
from pathlib import Path

_BASE = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_BASE))

from common.rag_base import BaseRAG, load_directory
from dotenv import load_dotenv

from .config import DEFAULT_K
from .dataset import get_test_cases
from .run_experiment import ExperimentRunner

load_dotenv()


def _init_rag(db_path: str) -> BaseRAG:
    if Path(db_path).exists():
        shutil.rmtree(db_path)

    rag = BaseRAG(persist_dir=db_path, collection_name="pipeline_demo")

    if rag.store.count() == 0:
        data_dir = str(_BASE / "common" / "text_data" / "rag_knowledge")
        loaded = load_directory(data_dir)
        docs = [d["content"] for d in loaded]
        doc_ids = [f"doc_{i+1}" for i in range(len(docs))]
        metas = [{"source": "rag_knowledge"} for _ in loaded]
        rag.add_documents(documents=docs, metadatas=metas, ids=doc_ids)
        print(f"知识库初始化完成，已添加 {rag.store.count()} 条文档")

    return rag


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG Pipeline 实验入口")
    parser.add_argument("--ask", type=str, help="提问单个问题")
    parser.add_argument("--config", type=str, default="vector", help="检索策略: vector / hyde / multi_route")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="检索 Top-K")
    parser.add_argument("--rerank", type=str, default="none", help="重排序: none / api")
    parser.add_argument("--grid-search", action="store_true", help="网格搜索最佳参数")
    parser.add_argument("--output", type=str, default=None, help="结果输出路径")

    args = parser.parse_args()

    db_path = str(_BASE / "02-知识库RAG" / "Day4-RAG检索增强生成" / "learn_prompt" / "chroma_db_pipeline")
    rag = _init_rag(db_path)
    runner = ExperimentRunner(rag.store, rag.llm)

    if args.grid_search:
        test_cases = get_test_cases()
        all_results = runner.grid_search(test_cases)
        output = args.output or str(Path(db_path).parent / "grid_search_result.json")
        with open(output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n网格搜索结果已写入 {output}")
        return

    if args.ask:
        config = {"retriever": args.config, "k": args.k, "rerank": args.rerank}
        result = runner.run_config([{"question": args.ask}], config)
        print(f"\n答案: {result['results'][0]['answer']}")
        print(f"忠实度: {result['results'][0]['faithfulness']}/10")
        print(f"相关性: {result['results'][0]['relevance']}/10")
        return

    test_cases = get_test_cases()
    config = {"retriever": args.config, "k": args.k, "rerank": args.rerank}
    result = runner.run_config(test_cases, config)

    print(f"\n实验结果: Overall={result['avg_overall']:.1f}, Latency={result['avg_latency']:.1f}s")

    output = args.output or str(Path(db_path).parent / "experiment_result.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {output}")


if __name__ == "__main__":
    main()
