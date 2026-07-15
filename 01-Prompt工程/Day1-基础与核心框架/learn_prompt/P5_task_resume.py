import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

messages = [
    {"role": "system", "content": "你是一个简历解析助手，只输出 JSON。提取姓名、学历、技能、工作经历"},
    {"role": "user", "content": "我叫王磊，2018年毕业于华中科技大学计算机硕士。精通 Python、Java、Docker。先在字节跳动做了3年后端开发，现在在蚂蚁集团做技术架构师。"},
    {"role": "assistant", "content": '{"name": "王磊", "education": {"school": "华中科技大学", "degree": "硕士", "major": "计算机科学与技术"}, "skills": ["Python", "Java", "Docker"], "experience": [{"company": "字节跳动", "position": "后端开发", "years": 3}, {"company": "蚂蚁集团", "position": "架构师"}]}'},
    {"role": "user", "content": "我是张薇，本科毕业于复旦新闻系，有5年新媒体运营经验。熟悉PS、PR、公众号排版。现在是一家MCN机构的内容主管。"}
]

resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    response_format={"type": "json_object"},
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=500
)

content = resp.choices[0].message.content
reasoning = getattr(resp.choices[0].message, "reasoning_content", "")

base_name = Path(__file__).parent / Path(__file__).stem

txt_output = f"""思考过程: {reasoning or "（无显式思考过程）"}

"""
if content:
    data = json.loads(content)
    txt_output += f"回复:\n{json.dumps(data, ensure_ascii=False, indent=2)}\n"
else:
    txt_output += "模型返回空 content，请检查 Prompt 是否包含 'json' 关键词\n"

with open(f"{base_name}_result.txt", "w", encoding="utf-8") as f:
    f.write(txt_output)

if content:
    with open(f"{base_name}_result.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(content), f, ensure_ascii=False, indent=2)

print(f"结果已写入 {base_name}_result.txt{' 和 ' + str(base_name) + '_result.json' if content else ''}")