import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

questions = [
    "某班学生排队，排成 3 行少 1 人，排成 4 行多 3 人，排成 5 行少 1 人，请问这个班最少有多少人？",
    "甲、乙、丙三人中，只有一个人说真话。甲说：不是我做的。乙说：是丙做的。丙说：不是我做的。是谁做的？",
]

output = ""
for q in questions:
    # 直接推理
    resp_direct = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": f"回答问题：{q}"}],
        max_tokens=2000
    )

    # Hi-CoT：先规划，再执行
    resp_hicot = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": f"""先列出解决这个问题需要的所有推理步骤，然后逐一执行每个步骤，最后给出答案。

问题：{q}"""}],
        max_tokens=2000
    )

    output += f"=== 问题: {q} ===\n"
    output += "--- 直接推理 ---\n{}\n\n".format(resp_direct.choices[0].message.content.strip())
    output += "--- Hi-CoT（先规划再执行） ---\n{}\n\n".format(resp_hicot.choices[0].message.content.strip())

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")
