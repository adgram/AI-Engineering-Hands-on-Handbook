import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

# 有序：3 个示例都带 emoji，格式一致
examples_ordered = [
    {"user": "今天天气真好", "assistant": "☀️ 适合出去玩"},
    {"user": "工作好忙啊", "assistant": "💼 加油打工人"},
    {"user": "周末到了", "assistant": "🎉 好好休息"},
]

# 乱序：最后一个示例故意不放 emoji
examples_shuffled = [
    {"user": "周末到了", "assistant": "🎉 好好休息"},
    {"user": "今天天气真好", "assistant": "☀️ 适合出去玩"},
    {"user": "工作好忙啊", "assistant": "工作再忙也要按时吃饭"},
]


def build_few_shot_messages(exs, test_input):
    msgs = [{"role": "system", "content": "对输入的句子做出友好回复"}]
    for ex in exs:
        msgs.append({"role": "user", "content": ex["user"]})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": test_input})
    return msgs


output = ""
for label, exs in [("顺序示例（带emoji）", examples_ordered), ("乱序示例（最后一个无emoji）", examples_shuffled)]:
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=build_few_shot_messages(exs, "晚饭吃什么"),
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        max_tokens=200
    )
    reasoning = getattr(resp.choices[0].message, "reasoning_content", "")
    content = resp.choices[0].message.content
    output += f"=== {label} ===\n"
    if reasoning:
        output += f"思考过程: {reasoning}\n\n"
    output += f"{content}\n\n"

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")
