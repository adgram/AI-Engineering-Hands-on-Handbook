import sys
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent/"common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient
client = LLMClient()

tot_prompt = """
问题：用 4 条直线一笔连接下图所有 9 个点，不能抬笔。
. . .
. . .
. . .

请用思维树方法解决：
1. 列出至少 3 种不同的解题思路（树的分支）
2. 对每个思路，评估它的可行性
3. 选择最有希望的思路深入推理
4. 如果第一个思路走不通，回溯到其他分支

格式：
分支1：[思路描述] → 可行性评估 → 尝试推理 → 结果
分支2：[思路描述] → 可行性评估 → 尝试推理 → 结果
...
最终选择：分支X → 最终解决方案
"""

response = client.chat(
    messages=[{"role": "user", "content": tot_prompt}],
)
output = response.choices[0].message.content
output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)
print(f"结果已写入 {output_file_name}")