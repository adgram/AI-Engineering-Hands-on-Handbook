import sys
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent/"common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient
client = LLMClient()

def plan_and_solve(task: str) -> str:
    plan_prompt = f"""
任务：{task}

请先制定一个执行计划，列出需要完成的子步骤。
格式：
计划：
1. ...
2. ...
3. ...

然后按步骤执行，输出每一步的结果。
"""
    response = client.chat(
        messages=[{"role": "user", "content": plan_prompt}],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    result = plan_and_solve("分析中国新能源汽车市场2024年的竞争格局，并预测2025年趋势")
    output = result
    output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
    with open(output_file_name, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"结果已写入 {output_file_name}")