from typing import Any, Callable
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    '''大模型客户端'''
    def __init__(
        self,
        api_key: str|None = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        max_retries: int = 3,
    ):
        pass

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int|None = None,
        tools: list|None = None,
        tool_choice: str|None = "auto",
        response_format: dict|None = None,
        stream: bool = False,
        reasoning_effort: str|None = None,
    ) -> Any:
        pass

    def chat_text(self, messages: list[dict], **kwargs) -> str:
        pass

    def chat_json(self, messages: list[dict], **kwargs) -> dict:
        pass

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_executor: Callable,
        max_rounds: int = 5,
        **kwargs
    ) -> str:
        pass

    def get_usage_stats(self) -> dict:
        pass