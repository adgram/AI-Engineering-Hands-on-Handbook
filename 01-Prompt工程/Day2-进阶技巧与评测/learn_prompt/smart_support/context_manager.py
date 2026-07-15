import sys, json
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent.parent / "common"
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
        scored = sorted(items, key=lambda x: self.score(x["content"], query), reverse=True)
        result = [x for x in scored if self.score(x["content"], query) >= self.threshold]
        if budget:
            return result[:budget]
        return result


class ContextManager:
    def __init__(self, system_prompt: str, window_size: int = 10):
        self.system_prompt = system_prompt
        self.window_size = window_size
        self.history: list[dict] = []
        self.scorer = RelevanceScorer()

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

    def build_context(self, query: str, include_history: bool = True) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if include_history and self.history:
            for msg in self.history[-self.window_size:]:
                messages.append(msg)
        messages.append({"role": "user", "content": query})
        return messages

    def get_kv_cache_summary(self) -> dict:
        return {
            "total_messages": len(self.history),
            "window_size": self.window_size,
            "recent_topics": [m["content"][:50] for m in self.history[-3:]],
        }
