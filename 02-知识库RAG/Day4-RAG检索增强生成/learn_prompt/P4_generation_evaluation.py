"""P4: 生成质量评估 — Faithfulness / Relevance / LLM-as-Judge"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from common.rag_base import BaseRAG
from dotenv import load_dotenv

load_dotenv()



class RAGEvaluator:
    def __init__(self, llm, model: str = "deepseek-v4-flash"):
        self.llm = llm
        self.model = model

    def check_faithfulness(self, answer: str, context: str) -> dict:
        prompt = f"""你是一个评估助手。判断以下回答是否完全基于给定的上下文，没有编造或幻觉。

上下文：
{context}

回答：
{answer}

请输出 JSON：
{{
    "faithfulness_score": 0-10,
    "is_hallucination": true/false,
    "hallucinated_parts": ["编造的内容1", "编造的内容2"],
    "reason": "简要说明"
}}"""

        resp = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)

    def check_relevance(self, answer: str, question: str) -> dict:
        prompt = f"""判断以下回答是否有效回答了用户的问题。

问题：{question}
回答：{answer}

评分标准：
- 10: 完全正确回答了问题，信息充分
- 7-9: 回答了问题，但可以更完善
- 4-6: 部分相关，但没有直接回答
- 1-3: 几乎不相关
- 0: 完全不相关

输出 JSON：{{"relevance_score": 0-10, "reason": "..."}}"""

        resp = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)

    def evaluate(self, question: str, answer: str, context: str) -> dict:
        faithfulness = self.check_faithfulness(answer, context)
        relevance = self.check_relevance(answer, question)

        return {
            "question": question,
            "answer": answer,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "overall": (faithfulness["faithfulness_score"] + relevance["relevance_score"]) / 2
        }


if __name__ == "__main__":
    rag = BaseRAG(
        persist_dir=str(Path(__file__).parent / "chroma_db_gen_eval"),
        collection_name="gen_eval_demo",
    )

    evaluator = RAGEvaluator(rag.llm)
    
    # 测试用例 1：好回答（忠实于上下文）
    print("=== 测试用例 1：好回答 ===")
    result1 = evaluator.evaluate(
        question="RAG 有什么优势？",
        answer="RAG 可以减少幻觉，支持最新知识，具有可追溯性，不需要重新训练模型。",
        context="RAG 结合了检索和生成，相比纯 LLM，RAG 可以降低幻觉风险，支持最新知识更新，并且回答可以追溯来源。相比微调，RAG 不需要重新训练模型。",
    )
    print(json.dumps(result1, ensure_ascii=False, indent=2))
    
    # 测试用例 2：幻觉回答（编造了上下文中没有的内容）
    print("\n=== 测试用例 2：幻觉回答 ===")
    result2 = evaluator.evaluate(
        question="RAG 有什么优势？",
        answer="RAG 可以提高模型训练速度，减少 GPU 使用量，并且能自动优化模型参数。",
        context="RAG 结合了检索和生成，可以降低幻觉风险，支持最新知识更新。",
    )
    print(json.dumps(result2, ensure_ascii=False, indent=2))
    
    # 测试用例 3：不相关回答
    print("\n=== 测试用例 3：不相关回答 ===")
    result3 = evaluator.evaluate(
        question="RAG 的工作流程是什么？",
        answer="向量数据库使用余弦相似度计算向量间的相关性。",
        context="RAG 的标准工作流程包含文档处理、向量化、检索和生成四个阶段。",
    )
    print(json.dumps(result3, ensure_ascii=False, indent=2))

    _output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
    with open(_output_file, "w", encoding="utf-8") as _f:
        _f.write("测试用例1:\n" + json.dumps(result1, ensure_ascii=False, indent=2) + "\n\n")
        _f.write("测试用例2:\n" + json.dumps(result2, ensure_ascii=False, indent=2) + "\n\n")
        _f.write("测试用例3:\n" + json.dumps(result3, ensure_ascii=False, indent=2))
    print(f"\n结果已写入 {_output_file}")