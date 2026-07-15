import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

config = {
    "templates": {
        "summary": {
            "system": "你是一个摘要专家。",
            "user": "将以下内容用{max_words}字以内总结：\n\n{text}"
        }
    }
}


def apply_template(name: str, **kwargs) -> list:
    tpl = config["templates"][name]
    return [
        {"role": "system", "content": tpl["system"]},
        {"role": "user", "content": tpl["user"].format(**kwargs)}
    ]


messages = apply_template("summary", text="人工智能正在深刻改变各行各业，从医疗诊断到自动驾驶，从自然语言处理到计算机视觉，AI 技术正在以前所未有的速度渗透到人类生活的方方面面。", max_words="50")

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=500
)

reasoning = getattr(response.choices[0].message, "reasoning_content", "")
content = response.choices[0].message.content

output = f"""思考过程: {reasoning or "（无显式思考过程）"}

回复: {content}
"""

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")