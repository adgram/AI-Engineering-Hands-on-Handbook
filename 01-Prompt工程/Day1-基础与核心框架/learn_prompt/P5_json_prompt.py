import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

messages = [
    {"role": "system", "content": "你是一个信息提取助手，只输出 JSON 格式"},
    {"role": "user", "content": "张三，35岁，毕业于北京大学计算机系，现为阿里巴巴算法工程师"},
    {"role": "assistant", "content": '{"name": "张三", "age": 35, "education": {"school": "北京大学", "major": "计算机系"}, "job": {"company": "阿里巴巴", "position": "算法工程师"}}'},
    {"role": "user", "content": "李四，28岁，上海交通大学硕士，在字节跳动做产品经理，擅长数据分析"}
]

resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=500
)

reasoning = getattr(resp.choices[0].message, "reasoning_content", "")
content = resp.choices[0].message.content

data = json.loads(content)

base_name = Path(__file__).parent / Path(__file__).stem

# .txt 输出完整内容（含思考过程）
txt_output = f"""思考过程: {reasoning or "（无显式思考过程）"}

回复:
{json.dumps(data, ensure_ascii=False, indent=2)}

姓名: {data['name']}
公司: {data['job']['company']}
"""

with open(f"{base_name}_result.txt", "w", encoding="utf-8") as f:
    f.write(txt_output)

# .json 输出纯 JSON 数据
with open(f"{base_name}_result.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"结果已写入 {base_name}_result.txt 和 {base_name}_result.json")
