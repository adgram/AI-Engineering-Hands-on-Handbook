import json
from datetime import datetime
from typing import Optional


class TraceStep:
    def __init__(self, step_type: str, content: str,
                 metadata: Optional[dict] = None):
        self.step_type = step_type
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        self.duration_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "type": self.step_type,
            "content_preview": self.content[:200],
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


class Tracer:
    def __init__(self):
        self.traces: list[TraceStep] = []
        self.start_time = datetime.now()

    def _add_step(self, step: TraceStep):
        if self.traces:
            prev = self.traces[-1]
            prev.duration_ms = (
                datetime.now() - datetime.fromisoformat(prev.timestamp)
            ).total_seconds() * 1000
        self.traces.append(step)

    def trace_think(self, content: str):
        self._add_step(TraceStep("think", content))

    def trace_tool_call(self, tool_name: str, args: dict, result: str):
        step = TraceStep("tool_call",
                        f"{tool_name}({json.dumps(args, ensure_ascii=False)})",
                        {"tool": tool_name, "args": args, "result_preview": result[:200]})
        self._add_step(step)

    def trace_result(self, content: str):
        self._add_step(TraceStep("result", content))

    def summary(self) -> dict:
        total_ms = (datetime.now() - self.start_time).total_seconds() * 1000
        return {
            "total_steps": len(self.traces),
            "total_duration_ms": total_ms,
            "step_types": {
                t: sum(1 for s in self.traces if s.step_type == t)
                for t in set(s.step_type for s in self.traces)
            },
            "trace": [s.to_dict() for s in self.traces],
        }
