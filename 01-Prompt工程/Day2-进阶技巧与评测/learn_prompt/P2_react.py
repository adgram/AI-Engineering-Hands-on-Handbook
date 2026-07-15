import sys
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent/"common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient
client = LLMClient()

react_prompt = """
你是一个 AI 助手，使用 ReAct 模式回答问题：
1. Thought: 思考当前需要做什么
2. Action: 执行一个操作（搜索 / 计算 / 查询）
3. Observation: 操作结果
4. 重复以上步骤直到可以给出 Final Answer

问题：北京和上海之间的距离是多少？如果坐高铁需要多久？
"""

response = client.chat(
    messages=[{"role": "user", "content": react_prompt}],
)
output = response.choices[0].message.content
output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)
print(f"结果已写入 {output_file_name}")