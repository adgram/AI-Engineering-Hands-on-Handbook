import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

class RAGEvaluator:
    """RAG 系统自动评估"""

    def evaluate_retrieval(self, queries: list, ground_truths: list, retrieved: list):
        hits = 0
        for q, gt, ret in zip(queries, ground_truths, retrieved):
            if any(gt_item in ret for gt_item in gt):
                hits += 1
        recall = hits / len(queries)

        mrr = 0
        for q, gt, ret in zip(queries, ground_truths, retrieved):
            for rank, doc in enumerate(ret, 1):
                if any(gt_item in doc for gt_item in gt):
                    mrr += 1 / rank
                    break
        mrr /= len(queries)

        return {"Recall": recall, "MRR": mrr}

    def evaluate_generation(self, questions: list, answers: list, contexts: list, llm_judge):
        scores = []
        for q, a, ctx in zip(questions, answers, contexts):
            resp = llm_judge.chat(
                messages=[{"role": "user", "content": f"""评估以下回答是否基于给定资料，没有编造。

资料：{ctx[:500]}

回答：{a}

输出 0-10 分（仅数字）："""}],
                temperature=0.0,
            )
            score = float(resp.choices[0].message.content.strip())
            scores.append(score)
        return {"avg_faithfulness": sum(scores) / len(scores), "scores": scores}


# 写入结果文件
_output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write(f"RAG评估器已配置，支持检索评估和生成评估")
print(f"结果已写入 {_output_file}")
