"""LLM 客户端封装 — 支持对话/流式/工具调用/重试/用量统计"""

import os, json, time, logging
from typing import Any, Callable
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        max_retries: int = 3,
    ):
        self.client = OpenAI(                         # OpenAI 客户端实例
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url
        )
        self.model = model                             # 模型名称
        self.max_retries = max_retries                 # 最大重试次数
        self.total_prompt_tokens = 0                   # 累计输入 Token 数
        self.total_completion_tokens = 0               # 累计输出 Token 数
        self.call_count = 0                            # 累计调用次数
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)      # 日志记录器

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int | None = None,
        tools: list | None = None,
        tool_choice: str | None = "auto",
        response_format: dict | None = None,
        stream: bool = False,
        reasoning_effort: str | None = None,
    ) -> Any:
        """通用对话接口，支持工具调用、流式输出、JSON 模式、思维链"""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format
        if stream:
            kwargs["stream"] = True
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

        # 带指数退避的重试机制
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = self.client.chat.completions.create(**kwargs)
                elapsed = time.time() - start
                self.call_count += 1
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens
                self.logger.info(
                    f"[{self.call_count}] {self.model} | "
                    f"耗时:{elapsed:.1f}s | "
                    f"输入:{response.usage.prompt_tokens} | "
                    f"输出:{response.usage.completion_tokens}"
                )
                return response
            except Exception as e:
                self.logger.warning(f"第{attempt+1}次调用失败: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                else:
                    raise

    def chat_text(self, messages: list[dict], **kwargs) -> str:
        """直接返回文本结果"""
        response = self.chat(messages, **kwargs)
        return response.choices[0].message.content

    def chat_json(self, messages: list[dict], **kwargs) -> dict:
        """返回 JSON 解析后的结果（自动设置 response_format）"""
        kwargs["response_format"] = {"type": "json_object"}
        response = self.chat(messages, **kwargs)
        content = response.choices[0].message.content
        return json.loads(content) if content else {}

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_executor: Callable,
        max_rounds: int = 5,
        **kwargs
    ) -> str:
        """工具调用循环：LLM 自主决策调工具 → 执行 → 继续，直到给出最终回答"""
        current_messages = messages.copy()
        for _ in range(max_rounds):
            response = self.chat(messages=current_messages, tools=tools, **kwargs)
            message = response.choices[0].message
            if not message.tool_calls:
                return message.content
            current_messages.append(message)
            for tc in message.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                self.logger.info(f"调用工具: {func_name}({func_args})")
                try:
                    result = tool_executor(func_name, func_args)
                except Exception as e:
                    result = json.dumps({"error": str(e)})
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                })
        return "达到最大轮数"

    def get_usage_stats(self) -> dict:
        """获取用量统计（调用次数、Token、估算费用）"""
        return {
            "model": self.model,
            "total_calls": self.call_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "estimated_cost_usd": (self.total_prompt_tokens * 0.14 + self.total_completion_tokens * 0.28) / 1_000_000,
        }
