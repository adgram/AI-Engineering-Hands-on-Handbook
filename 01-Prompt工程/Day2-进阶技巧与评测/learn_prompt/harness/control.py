import time, json
from typing import Callable
from .tools import ToolRegistry


class ControlLoop:
    def __init__(
        self,
        llm_call: Callable,
        tool_registry: ToolRegistry,
        max_rounds: int = 10,
        max_tokens_per_call: int = 4096,
    ):
        self.llm_call = llm_call
        self.tool_registry = tool_registry
        self.max_rounds = max_rounds
        self.max_tokens_per_call = max_tokens_per_call
        self.total_rounds = 0
        self.total_tokens = 0

    def run(self, messages: list[dict], **kwargs) -> str:
        current_msgs = messages.copy()

        for round_num in range(self.max_rounds):
            self.total_rounds = round_num + 1

            start = time.time()
            response = self.llm_call(
                messages=current_msgs,
                tools=self.tool_registry.get_all_schemas(),
                max_tokens=self.max_tokens_per_call,
                **kwargs
            )
            elapsed = time.time() - start

            self.total_tokens += response.usage.total_tokens if response.usage else 0
            message = response.choices[0].message

            if not message.tool_calls:
                return message.content

            current_msgs.append(message)

            for tc in message.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                print(f"  [第{round_num+1}轮] {func_name}({func_args}) [耗时:{elapsed:.1f}s]")
                result = self.tool_registry.execute(func_name, **func_args)
                current_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

        return "达到最大推理轮数"

    def get_usage(self) -> dict:
        return {
            "total_rounds": self.total_rounds,
            "total_tokens": self.total_tokens,
        }
