"""实验管理器：运行单次实验 / 网格搜索最佳配置"""

import json, time

from .config import LLM_MODEL, DEFAULT_K
from common.rag_client import (
    retrieve,
    api_rerank,
    Generator,
    evaluate_faithfulness,
    evaluate_relevance,
)


class ExperimentRunner:
    def __init__(self, store, llm, model: str = LLM_MODEL):
        self.store = store
        self.llm = llm
        self.model = model
        self.generator = Generator(llm, model)

    def run_config(self, test_cases: list, config: dict) -> dict:
        config_name = f"{config['retriever']}_{config.get('k', DEFAULT_K)}_{config.get('rerank', 'none')}"
        results = []

        for case in test_cases:
            start = time.time()

            docs = retrieve(config["retriever"], self.store, self.llm, case["question"], config.get("k", DEFAULT_K))

            if config.get("rerank") == "api":
                docs = [d for d, _ in api_rerank(case["question"], docs, config.get("k", DEFAULT_K))]

            answer = self.generator.generate(case["question"], docs)
            elapsed = time.time() - start

            faithfulness = evaluate_faithfulness(self.llm, "\n".join(docs), answer)
            relevance = evaluate_relevance(self.llm, case["question"], answer)

            results.append({
                "question": case["question"],
                "answer": answer,
                "faithfulness": faithfulness["faithfulness_score"],
                "relevance": relevance["relevance_score"],
                "latency": elapsed,
                "config": config_name,
            })

        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_rel = sum(r["relevance"] for r in results) / len(results)
        avg_latency = sum(r["latency"] for r in results) / len(results)

        return {
            "config": config_name,
            "details": config,
            "avg_faithfulness": avg_faith,
            "avg_relevance": avg_rel,
            "avg_overall": (avg_faith + avg_rel) / 2,
            "avg_latency": avg_latency,
            "results": results,
        }

    def grid_search(self, test_cases: list) -> list:
        configs = [
            {"retriever": "vector", "k": 3, "rerank": "none"},
            {"retriever": "vector", "k": 5, "rerank": "none"},
            {"retriever": "vector", "k": 5, "rerank": "api"},
            {"retriever": "hyde", "k": 5, "rerank": "none"},
            {"retriever": "multi_route", "k": 5, "rerank": "api"},
        ]

        all_results = []
        for cfg in configs:
            print(f"实验: {cfg}")
            result = self.run_config(test_cases, cfg)
            all_results.append(result)
            print(f"  Overall={result['avg_overall']:.1f}, Latency={result['avg_latency']:.1f}s")

        all_results.sort(key=lambda x: x["avg_overall"], reverse=True)

        print("\n最佳配置:")
        print(f"  {all_results[0]['config']}")
        print(f"  Score: {all_results[0]['avg_overall']:.1f}/10")

        return all_results
