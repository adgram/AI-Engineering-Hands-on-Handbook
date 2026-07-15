import sys, json, time
from datetime import datetime
from pathlib import Path
from typing import Callable
common_path = Path(__file__).parent.parent.parent.parent.parent / "common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient

client = LLMClient()


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name: str, schema: dict, handler: Callable):
        self._tools[name] = {"schema": schema, "handler": handler, "call_count": 0}

    def get_schemas(self) -> list[dict]:
        return [t["schema"] for t in self._tools.values()]

    def execute(self, name: str, **kwargs) -> str:
        tool = self._tools.get(name)
        if not tool:
            return json.dumps({"error": f"未知工具: {name}"})
        tool["call_count"] += 1
        try:
            return tool["handler"].__call__(**kwargs)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_stats(self) -> dict:
        return {name: t["call_count"] for name, t in self._tools.items()}


class ControlLoop:
    def __init__(self, registry: ToolRegistry, max_rounds: int = 10):
        self.registry = registry
        self.max_rounds = max_rounds
        self.rounds_used = 0

    def run(self, messages: list[dict]) -> str:
        current = messages.copy()
        for round_num in range(self.max_rounds):
            self.rounds_used = round_num + 1
            response = client.chat(messages=current, tools=self.registry.get_schemas())
            msg = response.choices[0].message
            if not msg.tool_calls:
                return msg.content
            current.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"  [第{round_num+1}轮] 调用 {name}({args})")
                result = self.registry.execute(name, **args)
                current.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        return "达到最大推理轮数"


class Tracer:
    def __init__(self):
        self.steps = []
        self.start = datetime.now()

    def log(self, step_type: str, content: str, metadata: dict = None):
        self.steps.append({
            "type": step_type,
            "content": content[:200],
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        })

    def summary(self) -> dict:
        elapsed = (datetime.now() - self.start).total_seconds()
        return {
            "total_steps": len(self.steps),
            "duration_seconds": round(elapsed, 2),
            "step_types": {
                t: sum(1 for s in self.steps if s["type"] == t)
                for t in set(s["type"] for s in self.steps)
            },
            "steps": self.steps,
        }
