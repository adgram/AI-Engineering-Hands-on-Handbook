import json
from typing import Callable, Dict, List, Optional


class Tool:
    def __init__(self, name: str, description: str,
                 parameters: Dict, handler: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.call_count = 0

    def to_openai_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def execute(self, **kwargs) -> str:
        self.call_count += 1
        return self.handler(**kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_all_schemas(self) -> List[Dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute(self, name: str, **kwargs) -> str:
        tool = self.get(name)
        if not tool:
            return f"错误：未知工具 '{name}'"
        return tool.execute(**kwargs)

    def get_stats(self) -> Dict:
        return {name: t.call_count for name, t in self._tools.items()}
