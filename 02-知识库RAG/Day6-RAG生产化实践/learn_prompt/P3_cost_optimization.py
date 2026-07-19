import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from common.llm_client import LLMClient
client = LLMClient()
import time
from collections import defaultdict
from datetime import datetime
import json

class TokenMonitor:
    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        self.sessions = defaultdict(lambda: {
            "prompt_tokens": 0, "completion_tokens": 0,
            "calls": 0, "total_time": 0
        })
        os.makedirs(log_dir, exist_ok=True)
    
    def log_call(self, session: str, prompt_tokens: int, completion_tokens: int, elapsed: float):
        s = self.sessions[session]
        s["prompt_tokens"] += prompt_tokens
        s["completion_tokens"] += completion_tokens
        s["calls"] += 1
        s["total_time"] += elapsed
    
    def cost_estimate(self, session: str = None) -> dict:
        """估算费用"""
        input_price = 0.14 / 1_000_000  # ¥/token
        output_price = 0.28 / 1_000_000
        
        if session:
            sessions = {session: self.sessions[session]}
        else:
            sessions = dict(self.sessions)
        
        report = {}
        for name, data in sessions.items():
            cost = data["prompt_tokens"] * input_price + data["completion_tokens"] * output_price
            report[name] = {
                **data,
                "estimated_cost_yuan": round(cost, 4),
            }
        
        return report
    
    def summary(self):
        total = self.cost_estimate()
        all_data = self.sessions
        t_prompt = sum(d["prompt_tokens"] for d in all_data.values())
        t_completion = sum(d["completion_tokens"] for d in all_data.values())
        t_calls = sum(d["calls"] for d in all_data.values())
        
        return {
            "total_calls": t_calls,
            "total_prompt_tokens": t_prompt,
            "total_completion_tokens": t_completion,
            "total_tokens": t_prompt + t_completion,
            "total_cost": total
        }

# 集成到 LLMClient
class CostAwareLLMClient(LLMClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = TokenMonitor(Path(__file__).parent/"logs")
        self.current_session = "default"
    
    def chat(self, messages, **kwargs):
        start = time.time()
        response = super().chat(messages, **kwargs)
        elapsed = time.time() - start
        
        self.monitor.log_call(
            self.current_session,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            elapsed
        )
        
        return response
    
    def print_cost_report(self):
        report = self.monitor.summary()
        print("=" * 40)
        print("Token 用量报告")
        print("=" * 40)
        print(f"总调用次数: {report['total_calls']}")
        print(f"总输入 Tokens: {report['total_prompt_tokens']:,}")
        print(f"总输出 Tokens: {report['total_completion_tokens']:,}")
        print(f"总计 Tokens: {report['total_tokens']:,}")
        print(f"估算费用: ¥{report['total_cost']:.2f}")

# === Code Block 2 ===

query = "RAG 是什么？"
# ❌ 冗余
prompt_verbose = f"你是一个非常有经验的并且知识渊博的AI助手，请帮助用户回答以下问题。用户的问题是：'{query}'请给出详细回答。"

# ✅ 精简
prompt_concise = f"回答：{query}"

# === Code Block 3 ===

# ~ 代替全文历史
def compress_history(history: list, max_tokens: int = 200) -> str:
    """用 LLM 压缩对话历史"""
    if not history:
        return ""
    
    text = json.dumps(history, ensure_ascii=False)
    response = client.chat(
        messages=[{"role": "user", "content": f"将以下对话压缩到{max_tokens}tokens以内，保留关键信息：{text}"}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# === Code Block 4 ===

# 简单任务用小模型
def smart_model(question: str) -> str:
    if len(question) < 20:  # 简单问题
        return "deepseek-v4-flash"  # 便宜的
    return "deepseek-v4-flash"  # 默认

# === Code Block 5 ===

# 控制 max_tokens（query 已在上面定义）
response = client.chat(
    messages=[{"role": "user", "content": query}],
    max_tokens=500,  # 限制输出长度
)

# === Code Block 6 ===

def analyze_token_usage(session_data: dict) -> dict:
    """分析 Token 使用效率"""
    total = session_data["prompt_tokens"] + session_data["completion_tokens"]
    
    # 输出/输入比
    ratio = session_data["completion_tokens"] / max(session_data["prompt_tokens"], 1)
    
    # 每次调用的平均 Token
    avg_per_call = total / max(session_data["calls"], 1)
    
    return {
        "output_input_ratio": round(ratio, 2),
        "avg_tokens_per_call": int(avg_per_call),
        "total_tokens": total,
    }

# 写入结果文件
_output_file = str(Path(__file__).parent / f"{Path(__file__).stem}_result.txt")
with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write(f"模块 {os.path.splitext(os.path.basename(__file__))[0]} 已加载：包含TokenMonitor、CostAwareLLMClient、analyze_token_usage")
print(f"结果已写入 {_output_file}")
