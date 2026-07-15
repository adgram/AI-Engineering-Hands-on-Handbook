import sys, re, io, contextlib
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent/"common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient
client = LLMClient()

pal_prompt = """
你有一台计算器可用，请把问题写成 Python 代码来求解。

问题：一个农场里有鸡和兔子共 35 只，脚共 94 只，鸡和兔子各多少只？

请输出可执行的 Python 代码，并给出运行结果。
"""

resp_code = client.chat(
    messages=[{"role": "user", "content": pal_prompt}]
)
code = resp_code.choices[0].message.content

code_block = re.search(r'```python\n(.*?)\n```', code, re.DOTALL)
output_lines = [f"生成的代码:\n{code_block.group(1) if code_block else code}"]
if code_block:
    f_stdout = io.StringIO()
    with contextlib.redirect_stdout(f_stdout):
        exec(code_block.group(1))
    output_lines.append(f"\n执行结果:\n{f_stdout.getvalue()}")
else:
    output_lines.append("\n(未检测到 Python 代码块)")

output = "\n".join(output_lines)
output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)
print(f"结果已写入 {output_file_name}")