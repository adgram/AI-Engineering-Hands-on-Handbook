import sys, re, json
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent.parent / "common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient

client = LLMClient()


class InputSanitizer:
    def __init__(self):
        self.dangerous_patterns = [
            (r"ignore\s+(all\s+)?(previous|above|prior)", "忽略指令攻击"),
            (r"forget\s+(all\s+)?(instructions|directives)", "忘记指令攻击"),
            (r"system\s*(prompt|message|instruction)", "系统提示词攻击"),
            (r"你(是|的).*(系统|设定|指令)", "中文系统提示词攻击"),
            (r"DAN|do\s+anything\s+now", "DAN 攻击"),
        ]

    def check(self, text: str) -> tuple[bool, str]:
        for pattern, label in self.dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"检测到{label}"
        return True, ""

    def sanitize(self, text: str) -> str:
        for pattern, _ in self.dangerous_patterns:
            text = re.sub(pattern, "[已过滤]", text, flags=re.IGNORECASE)
        return text


class OutputValidator:
    def __init__(self, max_length: int = 10000):
        self.max_length = max_length

    def check_sensitive(self, text: str) -> bool:
        patterns = [r"sk-[a-zA-Z0-9]+", r"password", r"API[_-]?[Kk]ey"]
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return False
        return True

    def check_length(self, text: str) -> bool:
        return len(text) <= self.max_length


class DefensePipeline:
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.validator = OutputValidator()
        self.stats = {"input_checks": 0, "output_checks": 0, "blocked": 0}

    def process_input(self, user_input: str) -> str:
        self.stats["input_checks"] += 1
        safe, reason = self.sanitizer.check(user_input)
        if not safe:
            self.stats["blocked"] += 1
            return f"[安全拦截] {reason}，您的输入已被过滤处理。"
        return user_input

    def verify_output(self, output: str) -> str:
        self.stats["output_checks"] += 1
        if not self.validator.check_sensitive(output):
            self.stats["blocked"] += 1
            return "[安全拦截] 输出包含敏感信息，已阻止显示。"
        if not self.validator.check_length(output):
            return output[:self.validator.max_length] + "\n\n[输出已截断]"
        return output

    def get_stats(self) -> dict:
        return dict(self.stats)
