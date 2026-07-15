import sys, json
from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent/"common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient
client = LLMClient()

@dataclass
class LoopConfig:
    prompt_sections: dict  # {"指令": "...", "规则": "...", "示例": "..."}
    test_cases: list[dict]
    evaluator_fn: Callable
    llm_client: any
    max_rounds: int = 5
    history: list = field(default_factory=list)

class LoopEngine:
    def __init__(self, config: LoopConfig):
        self.cfg = config
        self.best_prompt = dict(config.prompt_sections)
        self.best_score = 0.0

    def assemble_prompt(self, sections: dict) -> str:
        return "\n\n".join([f"### {k}\n{v}" for k, v in sections.items()])

    def generate_variant(self, sections: dict, error_report: str) -> dict:
        """Generator：增量修改，每轮只改 1-2 个 section"""
        self.assemble_prompt(sections)
        resp = self.cfg.llm_client.chat(
            messages=[{"role": "system", "content": "你是一个 Prompt 优化专家。每次只修改 1-2 个 section。"},
                      {"role": "user", "content": f"当前 Prompt sections：\n{json.dumps(sections, ensure_ascii=False, indent=2)}\n\n"
                                                   f"错误分析：\n{error_report}\n\n"
                                                   f"请返回修改后的完整 sections JSON（保持其他 section 不变）。"}]
        )
        return json.loads(resp.choices[0].message.content)

    def optimize(self):
        for round_idx in range(self.cfg.max_rounds):
            prompt_text = self.assemble_prompt(self.best_prompt)
            score = self.cfg.evaluator_fn(prompt_text, self.cfg.test_cases)
            failures = self._collect_failures(prompt_text)

            # Optimizer 分析
            error_report = self._analyze(failures) if failures else ""

            # Generator 生成变体
            if failures:
                new_sections = self.generate_variant(self.best_prompt, error_report)
                new_score = self.cfg.evaluator_fn(self.assemble_prompt(new_sections), self.cfg.test_cases)
                # 棘轮机制：只保留分数不降的版本
                if new_score >= score:
                    self.best_prompt = new_sections
                    self.best_score = new_score

            self.cfg.history.append({"round": round_idx, "score": score, "failures": len(failures)})
            print(f"Round {round_idx+1}: score={score:.0%}, failures={len(failures)}")
            if not failures:
                break

    def _collect_failures(self, prompt_text) -> list[dict]:
        results = []
        for case in self.cfg.test_cases:
            resp = self.cfg.llm_client.chat(
                messages=[{"role": "system", "content": prompt_text},
                          {"role": "user", "content": case["input"]}]
            )
            output = resp.choices[0].message.content
            if case["expected"] not in output:
                results.append({"input": case["input"], "expected": case["expected"], "output": output})
        return results

    def _analyze(self, failures: list[dict]) -> str:
        text = "\n".join([json.dumps(f, ensure_ascii=False) for f in failures[:5]])
        resp = self.cfg.llm_client.chat(
            messages=[{"role": "system", "content": "分析以下失败案例的错误模式，用中文总结。"},
                      {"role": "user", "content": text}]
        )
        return resp.choices[0].message.content



# 1. 定义任务和测试集
test_cases = [
    {"input": "这款手机拍照一流，但续航一般", "expected": "正面"},
    {"input": "用了三天就坏了，质量太差", "expected": "负面"},
    {"input": "价格适中，功能齐全", "expected": "正面"},
    {"input": "一般般吧，没什么特别的", "expected": "中性"},
    {"input": "客服态度差还不给退款", "expected": "负面"},
    {"input": "包装精美，送人很有面子", "expected": "正面"},
    {"input": "功能很多但大部分用不上", "expected": "中性"},
    {"input": "性价比很高，推荐购买", "expected": "正面"},
    {"input": "物流太慢了等了一个星期", "expected": "负面"},
    {"input": "和描述一致，没有惊喜也没有失望", "expected": "中性"},
]

# 2. 定义评估函数
def eval_fn(prompt, cases):
    correct = 0
    for c in cases:
        resp = client.chat(
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": c["input"]}]
        )
        if c["expected"] in resp.choices[0].message.content:
            correct += 1
    return correct / len(cases)

# 3. 手动基线
base_prompt = "分析评论的情感：正面、负面或中性"
base_score = eval_fn(base_prompt, test_cases)
output_lines = [f"基线准确率: {base_score:.0%}"]

# 4. 运行 Loop Engine
config = LoopConfig(
    prompt_sections={
        "角色设定": "你是一个情感分析专家。",
        "指令": "分析以下评论的情感倾向。",
        "规则": "只输出正面、负面或中性",
        "输出格式": "仅输出一个词"
    },
    test_cases=test_cases,
    evaluator_fn=eval_fn,
    llm_client=client,
    max_rounds=3
)
engine = LoopEngine(config)
engine.optimize()

# 5. 查看优化历史
for h in engine.cfg.history:
    output_lines.append(f"轮次 {h['round']+1}: 准确率={h['score']:.0%}, 失败数={h['failures']}")

output = "\n".join(output_lines)
output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)
print(f"结果已写入 {output_file_name}")
