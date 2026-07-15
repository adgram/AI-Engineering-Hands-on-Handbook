"""Agent 基类 — 提供工具注册、工具调用循环、ReAct 单步执行"""

import os, json
from typing import Callable
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class BaseAgent:
    """基础 Agent：System Prompt + 工具注册 + Thought→Action→Observation 循环"""
    def __init__(self, model: str = "deepseek-v4-flash"):
        self.client = OpenAI(                         # OpenAI 客户端实例
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.model = model                             # 模型名称
        self.messages = []                             # 对话消息列表
        self.tools = []                                # 已注册工具列表
        self.call_count = 0                            # 累计调用次数
        self.total_tokens = 0                          # 累计 Token 数

    def add_system_prompt(self, content: str):
        self.messages.append({"role": "system", "content": content})

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_tool(self, name: str, description: str, parameters: dict, func: Callable):
        """注册一个工具：定义 JSON Schema + 绑定执行函数"""
        self.tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        })
        setattr(self, f"_tool_{name}", func)

    def _execute_tool(self, name: str, args: dict) -> str:
        handler = getattr(self, f"_tool_{name}", None)
        if handler:
            try:
                result = handler(**args)
                return json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
            except Exception as e:
                return json.dumps({"error": str(e)})
        return json.dumps({"error": f"未知工具: {name}"})

    def run(self, user_input: str, max_rounds: int = 10) -> str:
        """完整运行：工具调用循环，直到 LLM 给出最终回答"""
        self.add_user_message(user_input)
        for _ in range(max_rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools if self.tools else None,
                tool_choice="auto" if self.tools else None,
            )
            self.call_count += 1
            self.total_tokens += response.usage.total_tokens
            message = response.choices[0].message

            if not message.tool_calls:
                return message.content

            self.messages.append(message)
            for tc in message.tool_calls:
                args = json.loads(tc.function.arguments)
                result = self._execute_tool(tc.function.name, args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

        return "达到最大轮数"

    def reset(self):
        self.messages = []
        self.call_count = 0
        self.total_tokens = 0


class ReActAgent(BaseAgent):
    """ReAct 单步模式：每次 step 只走一轮，便于外部控制循环"""
    def step(self, user_input: str) -> str:
        self.add_user_message(user_input)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools if self.tools else None,
            tool_choice="auto" if self.tools else None,
        )
        self.call_count += 1
        self.total_tokens += response.usage.total_tokens
        message = response.choices[0].message

        if message.content:
            self.messages.append({"role": "assistant", "content": message.content})

        if message.tool_calls:
            self.messages.append(message)
            for tc in message.tool_calls:
                args = json.loads(tc.function.arguments)
                result = self._execute_tool(tc.function.name, args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
            return "[工具调用] " + tc.function.name
        return message.content or ""
