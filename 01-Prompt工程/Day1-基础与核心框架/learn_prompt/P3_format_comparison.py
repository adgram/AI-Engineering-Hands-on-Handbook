import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

test_text = "张三，男，1995年出生，北京大学计算机系毕业，现在腾讯做后端开发，擅长 Go 和 Python"

few_shot_text = [
    {"role": "system", "content": "提取用户信息"},
    {"role": "user", "content": "李四，女，28岁，本科，设计师"},
    {"role": "assistant", "content": "姓名：李四，性别：女，年龄：28，学历：本科，职业：设计师"},
    {"role": "user", "content": test_text}
]

few_shot_json = [
    {"role": "system", "content": "提取用户信息，输出 JSON"},
    {"role": "user", "content": "李四，女，28岁，本科，设计师"},
    {"role": "assistant", "content": '{"name": "李四", "gender": "女", "age": 28, "education": "本科", "job": "设计师"}'},
    {"role": "user", "content": test_text}
]

few_shot_table = [
    {"role": "system", "content": "提取用户信息，输出 Markdown 表格"},
    {"role": "user", "content": "李四，女，28岁，本科，设计师"},
    {"role": "assistant", "content": "| 姓名 | 性别 | 年龄 | 学历 | 职业 |\n|------|------|------|------|------|\n| 李四 | 女 | 28 | 本科 | 设计师 |"},
    {"role": "user", "content": test_text}
]

output = ""
for label, msgs in [("纯文本格式", few_shot_text), ("JSON格式", few_shot_json), ("Markdown表格", few_shot_table)]:
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=msgs,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        max_tokens=3000
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