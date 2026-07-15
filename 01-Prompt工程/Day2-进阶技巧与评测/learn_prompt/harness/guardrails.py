import re, json
from typing import Optional


class Guardrail:
    def check(self, content: str) -> Optional[str]:
        raise NotImplementedError


class InputSanitizer(Guardrail):
    def __init__(self):
        self.dangerous_patterns = [
            r"ignore\s+(all\s+)?(previous|above|prior)",
            r"forget\s+(all\s+)?(instructions|directives)",
            r"system\s*(prompt|message|instruction)",
            r"你(是|的).*(系统|设定|指令)",
        ]

    def check(self, content: str) -> Optional[str]:
        for pattern in self.dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return f"输入包含潜在注入模式: {pattern}"
        return None


class OutputValidator(Guardrail):
    def __init__(self, expected_schema: Optional[dict] = None):
        self.expected_schema = expected_schema

    def check_json(self, content: str) -> Optional[str]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return "输出不是合法的 JSON"
        if self.expected_schema:
            for field in self.expected_schema.get("required", []):
                if field not in data:
                    return f"缺少必填字段: {field}"
        return None

    def check_length(self, content: str, max_len: int = 10000) -> Optional[str]:
        if len(content) > max_len:
            return f"输出超长: {len(content)} > {max_len}"
        return None


class GuardrailPipeline:
    def __init__(self):
        self.guardrails: list[Guardrail] = []
        self.rejection_log: list[dict] = []

    def add(self, guardrail: Guardrail):
        self.guardrails.append(guardrail)

    def check_input(self, content: str) -> Optional[str]:
        for g in self.guardrails:
            result = g.check(content)
            if result:
                self.rejection_log.append({
                    "type": type(g).__name__,
                    "reason": result,
                    "content_preview": content[:100]
                })
                return result
        return None

    def get_stats(self) -> dict:
        return {
            "total_rejections": len(self.rejection_log),
            "by_type": {
                name: sum(1 for r in self.rejection_log if r["type"] == name)
                for name in set(r["type"] for r in self.rejection_log)
            }
        }
