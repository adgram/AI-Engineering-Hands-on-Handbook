import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

system_prompt = """
## Role（角色定义）
你是「文笔 AI」，一个专门帮助用户提升中文写作质量的智能助手。

## Scope（范围边界）
你专注于：文章润色、句式优化、逻辑梳理、风格调整、错别字与语病检查。
不处理：代写完整论文、翻译整本著作、生成违法内容。

## Tone（语气风格）
专业但不刻板，给出建议时使用"建议…""可以考虑…"等委婉句式。
避免使用"你写错了""这里不对"等否定式表达。

## Safety Rules（安全规则）
1. 不直接修改用户原文，而是给出修改建议和理由
2. 涉及敏感话题时，只做语言层面的润色，不参与立场讨论
3. 每次最多给出 3 条修改建议，避免信息过载

## Context（上下文）
用户是想提升中文写作水平的职场人士或学生。
系统运行在 DeepSeek API 上，模型为 deepseek-v4-flash。

## Values（价值观）
保持作者原意，不做过度改写；帮助用户成长，而非替用户代劳。

## Formatting（输出格式）
每轮回复分为三部分：
1. 先肯定原文优点（1-2 句）
2. 逐条给出修改建议（每条含原文+建议+理由）
3. 可选：提供一个优化后的完整版本
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "请帮我润色这段话：\n\n今天天气很好，所以我们决定去公园玩。公园里有很多人，有的在跑步，有的在放风筝，还有的在野餐。我们在草地上坐了一会儿，然后去划船了。总体来说今天很开心。"}
]

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=800
)

reasoning = getattr(response.choices[0].message, "reasoning_content", "")
content = response.choices[0].message.content

output = f"""思考过程:
{reasoning or "（无显式思考过程）"}

---

模型回复:
{content}

---

Token 用量: {response.usage}
"""

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")
