import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "分析用户评论，按以下 json 结构输出：\n{ \"sentiment\": \"正面/负面/neutral/mixed\", \"topics\": [\"主题1\", \"主题2\"], \"intent\": \"抱怨/咨询/赞美/对比\", \"urgency\": \"high/medium/low\" }"},
        {"role": "user", "content": "这个手机电池太不耐用了，半天就没电，但拍照效果还不错"},
    ],
    response_format={"type": "json_object"},
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=500
)

content = response.choices[0].message.content
reasoning = getattr(response.choices[0].message, "reasoning_content", "")

base_name = Path(__file__).parent / Path(__file__).stem

txt_output = f"""思考过程: {reasoning or "（无显式思考过程）"}

"""
if content:
    result = json.loads(content)
    txt_output += f"回复:\n{json.dumps(result, ensure_ascii=False, indent=2)}\n"
else:
    txt_output += "警告：模型返回了空的 content，请调整 Prompt 重试\n"

with open(f"{base_name}_result.txt", "w", encoding="utf-8") as f:
    f.write(txt_output)

if content:
    with open(f"{base_name}_result.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(content), f, ensure_ascii=False, indent=2)

print(f"结果已写入 {base_name}_result.txt{' 和 ' + str(base_name) + '_result.json' if content else ''}")