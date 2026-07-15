import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

messages = [
    {"role": "system", "content": "你是一个旅游规划助手。每次回复包含两部分：1. 回答用户的问题2. 紧跟 2-3 个追问来收集偏好（用「🤔 了解你」开头）"}
]

output = ""
print("=== 旅游规划助手（输入 'quit' 退出）===\n")
round_num = 0
while True:
    user_input = input("你: ")
    if user_input.lower() == 'quit':
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        max_tokens=500
    )

    reasoning = getattr(response.choices[0].message, "reasoning_content", "")
    reply = response.choices[0].message.content

    print(f"AI: {reply}\n")
    round_num += 1
    output += f"=== 第 {round_num} 轮 ===\n"
    output += f"用户: {user_input}\n"
    if reasoning:
        output += f"思考过程: {reasoning}\n"
    output += f"AI: {reply}\n\n"

    messages.append({"role": "assistant", "content": reply})

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")