import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

questions = [
    "巴黎是哪个国家的首都？",
    "a = 5, b = 3, a + b = ？",
]

output = ""
for q in questions:
    resp_short = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": f"直接回答，不解释：{q}"}],
        max_tokens=500
    )
    short_len = len(resp_short.choices[0].message.content)

    resp_normal = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": f"回答问题，一步步推理：{q}"}],
        max_tokens=1000
    )
    normal_len = len(resp_normal.choices[0].message.content)

    resp_long = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": f"""请非常详细地推理以下问题：
1. 先写出已知条件
2. 逐步计算，每一步都要解释
3. 检查每一步是否有错误
4. 用另一种方法重新验证答案
5. 给出最终答案

问题：{q}"""}],
        max_tokens=3000
    )
    long_len = len(resp_long.choices[0].message.content)

    output += f"=== 问题: {q} ===\n"
    output += "极短: {} 字符 | 正常: {} 字符 | 过长: {} 字符\n".format(short_len, normal_len, long_len)
    output += "--- 极短 ---\n{}\n\n".format(resp_short.choices[0].message.content.strip())
    output += "--- 正常 ---\n{}\n\n".format(resp_normal.choices[0].message.content.strip())
    output += "--- 过长 ---\n{}\n\n".format(resp_long.choices[0].message.content.strip())

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")
