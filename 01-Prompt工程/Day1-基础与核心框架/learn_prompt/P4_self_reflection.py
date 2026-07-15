import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

output = ""

# 第一步：生成有缺陷的初始版本
resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "写一个 Python 函数，从 JSON 文件中读取配置，然后连接数据库执行指定的 SQL 查询，返回结果。"}],
    max_tokens=8000
)
initial = resp.choices[0].message.content
output += "=== 第一版（初始实现） ===\n"
output += initial + "\n\n"

# 第二步：让模型自我审查并优化
resp_review = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "user", "content": f"""请严格审查以下代码，找出所有问题（错误处理、安全性、资源管理、代码健壮性）：

{initial}

要求：
1. 列出每个问题及严重程度（高/中/低）
2. 每个问题给出原因
3. 提供修复后的完整版本"""}
    ],
    max_tokens=12000
)
review = resp_review.choices[0].message.content
output += "=== 自我审查结果 ===\n"
output += review + "\n"

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")
