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
        {"role": "system", "content": "提取以下文本中的信息，输出 JSON 格式"},
        {"role": "user", "content": "昨天下午3点，李明在北京朝阳区建国路88号丢失了黑色钱包"},
    ],
    response_format={"type": "json_object"},
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=500
)

reasoning = getattr(response.choices[0].message, "reasoning_content", "")
content = response.choices[0].message.content

result = json.loads(content)
base_name = Path(__file__).parent / Path(__file__).stem

with open(f"{base_name}_result.txt", "w", encoding="utf-8") as f:
    f.write(f"""思考过程: {reasoning or "（无显式思考过程）"}

回复:
{json.dumps(result, ensure_ascii=False, indent=2)}
""")

with open(f"{base_name}_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"结果已写入 {base_name}_result.txt 和 {base_name}_result.json")