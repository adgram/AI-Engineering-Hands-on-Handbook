from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, TOP_K

class QAEngine:
    def __init__(self, vector_store, llm_client=None):
        self.vector_store = vector_store
        self.client = llm_client or OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    def ask(self, question: str, n_results: int = TOP_K) -> dict:
        results = self.vector_store.search(question, n_results=n_results)

        if not results['documents'][0]:
            return {"answer": "未找到相关信息", "sources": []}

        context_parts = []
        sources = []
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            context_parts.append(f"[来源 {i+1}] {doc}")
            sources.append({"file": meta['file'], "chunk": doc[:100]})

        context = "\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": "你是知识库问答助手，基于提供的参考资料回答问题。如果信息不足，请如实说明。引用来源时标注 [来源 N]。"},
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"}
        ]

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": sources
        }
