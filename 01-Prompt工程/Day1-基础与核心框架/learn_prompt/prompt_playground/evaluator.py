from typing import Callable
import sys
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent.parent/"common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient

class Evaluator:
    def __init__(self):
        self.client = LLMClient()

    def run_test(
        self,
        test_cases: list[dict],
        system_prompt: str,
        judge_function: Callable[[str, str], bool],
        reasoning_effort: str | None = None,
        verbose: bool = False,
    ) -> dict:
        results = []
        correct = 0
        for case in test_cases:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": case["input"]}
            ]
            output = self.client.chat_text(messages, temperature=0, reasoning_effort=reasoning_effort)
            is_correct = judge_function(output, case["expected"])
            if is_correct:
                correct += 1
            results.append({
                "input": case["input"],
                "output": output,
                "expected": case["expected"],
                "correct": is_correct
            })
            if verbose:
                mark = "[OK]" if is_correct else "[X]"
                print(f"  {mark} 输入: {case['input'][:30]} -> 输出: {output[:20]} | 期望: {case['expected']}")
        accuracy = correct / len(test_cases) if test_cases else 0
        if verbose:
            print(f"准确率: {accuracy:.0%} ({correct}/{len(test_cases)})")
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_cases),
            "results": results
        }

    def compare(
        self,
        test_cases: list[dict],
        prompt_v1: str,
        prompt_v2: str,
        judge_function: Callable
    ) -> dict:
        result_v1 = self.run_test(test_cases, prompt_v1, judge_function)
        result_v2 = self.run_test(test_cases, prompt_v2, judge_function)
        return {
            "v1": result_v1,
            "v2": result_v2,
            "improvement": result_v2["accuracy"] - result_v1["accuracy"]
        }