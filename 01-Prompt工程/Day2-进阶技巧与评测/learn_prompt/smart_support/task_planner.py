import sys, json
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent.parent / "common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient

client = LLMClient()


class TaskPlanner:
    def __init__(self):
        self.plan = []
        self.results = []

    def plan_task(self, request: str) -> list[dict]:
        prompt = f"""你是一个客服任务规划专家。将以下客户请求拆分为可执行的子步骤。

客户请求：{request}

请以 JSON 数组格式返回步骤列表，每步包含 step（步骤编号）和 action（具体操作）。
只返回 JSON，不要其他内容。"""
        resp = client.chat_json(messages=[{"role": "user", "content": prompt}])
        if isinstance(resp, list):
            self.plan = resp
        elif isinstance(resp, dict) and "steps" in resp:
            self.plan = resp["steps"]
        else:
            self.plan = [{"step": 1, "action": request}]
        return self.plan

    def execute_step(self, step: dict, context: dict = None) -> str:
        action = step.get("action", str(step))
        prompt = f"执行以下客服操作：{action}"
        if context:
            prompt += f"\n上下文：{json.dumps(context, ensure_ascii=False)}"
        resp = client.chat_text(messages=[{"role": "user", "content": prompt}])
        self.results.append({"step": step, "result": resp})
        return resp

    def execute_all(self, request: str, context: dict = None) -> list[dict]:
        self.plan_task(request)
        for step in self.plan:
            self.execute_step(step, context)
        return self.results

    def get_summary(self) -> str:
        parts = [f"计划共 {len(self.plan)} 步，已完成 {len(self.results)} 步"]
        for i, r in enumerate(self.results):
            parts.append(f"  步骤 {i+1}: {r['result'][:100]}")
        return "\n".join(parts)
