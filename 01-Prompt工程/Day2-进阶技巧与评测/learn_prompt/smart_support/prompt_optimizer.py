import sys, json
from dataclasses import dataclass, field
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent.parent / "common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient

client = LLMClient()


@dataclass
class OptimizerConfig:
    template_sections: dict
    test_cases: list[dict]
    max_rounds: int = 5
    history: list = field(default_factory=list)


class PromptOptimizer:
    def __init__(self, config: OptimizerConfig):
        self.cfg = config
        self.best_template = dict(config.template_sections)
        self.best_score = 0.0

    def assemble(self, sections: dict) -> str:
        return "\n\n".join([f"### {k}\n{v}" for k, v in sections.items()])

    def evaluate(self, template_text: str) -> float:
        correct = 0
        for case in self.cfg.test_cases:
            messages = [
                {"role": "system", "content": template_text},
                {"role": "user", "content": case["input"]},
            ]
            resp = client.chat_text(messages)
            if case.get("expected", "") in resp:
                correct += 1
        return correct / len(self.cfg.test_cases) if self.cfg.test_cases else 0

    def optimize(self):
        for round_idx in range(self.cfg.max_rounds):
            template_text = self.assemble(self.best_template)
            score = self.evaluate(template_text)
            failures = self._collect_failures(template_text)
            error_report = self._analyze_failures(failures) if failures else ""
            if failures:
                new_sections = self._generate_variant(error_report)
                new_score = self.evaluate(self.assemble(new_sections))
                if new_score >= score:
                    self.best_template = new_sections
                    self.best_score = new_score
            self.cfg.history.append({
                "round": round_idx + 1,
                "score": score,
                "failures": len(failures),
            })
            print(f"第{round_idx+1}轮: 得分={score:.0%}, 失败={len(failures)}")
            if not failures:
                break
        return self.cfg.history

    def _collect_failures(self, template_text: str) -> list[dict]:
        failures = []
        for case in self.cfg.test_cases:
            resp = client.chat_text(
                messages=[
                    {"role": "system", "content": template_text},
                    {"role": "user", "content": case["input"]},
                ]
            )
            if case.get("expected", "") not in resp:
                failures.append(case)
        return failures

    def _analyze_failures(self, failures: list[dict]) -> str:
        text = json.dumps(failures[:5], ensure_ascii=False)
        return client.chat_text(messages=[
            {"role": "system", "content": "分析失败模式，用中文总结原因和改进方向。"},
            {"role": "user", "content": text},
        ])

    def _generate_variant(self, error_report: str) -> dict:
        resp = client.chat_json(messages=[
            {
                "role": "system",
                "content": "你是 Prompt 优化专家。根据错误分析返回改进后的完整 template sections JSON。",
            },
            {
                "role": "user",
                "content": f"当前模板：\n{json.dumps(self.best_template, ensure_ascii=False, indent=2)}\n\n错误分析：\n{error_report}",
            },
        ])
        if isinstance(resp, dict):
            return resp
        return dict(self.best_template)
