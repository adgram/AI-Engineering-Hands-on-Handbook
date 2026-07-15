import os
import re
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

# 开放生成任务：每次生成的回复天然不同
# 多次运行后统计关键词，看哪些词最常出现
topic = "人工智能对教育的影响"

output = ""
answers = []
for i in range(5):
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": f"第{i+1}轮：用一句话概括{topic}。"}],
        max_tokens=100
    )
    content = resp.choices[0].message.content.strip()
    answers.append(content)
    output += f"第{i+1}次: {content}\n"

# 提取每次回复中的关键词（简单取前两个最长词作为示例）
all_words = []
for a in answers:
    words = re.findall(r'[\u4e00-\u9fff]{2,}', a)
    all_words.extend(words[:3])

word_freq = Counter(all_words).most_common(5)
output += f"\n出现最多的关键词：\n"
for w, c in word_freq:
    output += f"  「{w}」: {c}次\n"

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")
