import sys, json
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent/"common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient

client = LLMClient()

class RelevanceScorer:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def score(self, text: str, query: str) -> float:
        text_words = set(text.lower().split())
        query_words = set(query.lower().split())
        if not query_words:
            return 0.5
        return len(text_words & query_words) / len(query_words)

    def filter(self, items: list[dict], query: str, budget: int = None) -> list[dict]:
        for item in items:
            item["relevance"] = self.score(item["content"], query)
        scored = sorted(items, key=lambda x: x["relevance"], reverse=True)
        result = [x for x in scored if x["relevance"] >= self.threshold]
        if budget:
            while len(result) > budget and self.threshold < 1.0:
                self.threshold += 0.1
                result = [x for x in scored if x["relevance"] >= self.threshold]
        return result

def structure_context(system_prompt, definitions, references, user_input):
    messages = [{"role": "system", "content": system_prompt}]
    if definitions:
        messages.append({"role": "system", "content": f"## 关键定义\n{definitions}"})
    if references:
        messages.append({"role": "system", "content": f"## 参考资料\n{references}"})
    messages.append({"role": "user", "content": user_input})
    return messages

class Compressor:
    def compress_summary(self, text: str) -> str:
        resp = client.chat(
            messages=[{"role": "user", "content": f"压缩到 50 字以内：\n{text}"}]
        )
        return resp.choices[0].message.content

    def compress_structured(self, data: dict) -> str:
        essential = {k: v for k, v in data.items() if k not in ["debug_info", "raw_response"]}
        return json.dumps(essential, ensure_ascii=False, sort_keys=True)

    def selective_compress(self, items: list[dict], threshold=0.7) -> list[str]:
        return [
            item["content"] if item.get("relevance", 0) >= threshold
            else f"[压缩] {self.compress_summary(item['content'])}"
            for item in items
        ]

class GSSCPipeline:
    def __init__(self, system_prompt: str, threshold=0.3):
        self.system_prompt = system_prompt
        self.scorer = RelevanceScorer(threshold)
        self.compressor = Compressor()
        self.memory = []

    def add_to_memory(self, content: str):
        self.memory.append(content)

    def build_context(self, query: str, budget: int = None):
        sources = [{"source": "memory", "content": m} for m in self.memory]
        selected = self.scorer.filter(sources, query, budget)

        context_parts = []
        for item in selected:
            if item["relevance"] < 0.5:
                context_parts.append(self.compressor.compress_summary(item["content"]))
            else:
                context_parts.append(item["content"])

        definitions = [c for c in context_parts if len(c) < 100]
        references = [c for c in context_parts if len(c) >= 100]
        return structure_context(self.system_prompt, "\n".join(definitions),
                                 "\n".join(references), query)

    def ask(self, query: str):
        messages = self.build_context(query)
        resp = client.chat(messages=messages)
        self.add_to_memory(f"Q: {query}\nA: {resp.choices[0].message.content}")
        return resp.choices[0].message.content

pipeline = GSSCPipeline("你是一个专业的 AI 助手。")
pipeline.add_to_memory("用户偏好：喜欢简洁回答。")
reply = pipeline.ask("请用一段话解释上下文工程")

output = reply
output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)
print(f"结果已写入 {output_file_name}")
