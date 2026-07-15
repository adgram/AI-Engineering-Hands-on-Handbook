import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from jinja2 import Template

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

template_str = """
你是一个{{ role }}领域的专家。

{% if style %}
请用{{ style }}的风格回答。
{% endif %}

{% if context %}
## 参考信息
{{ context }}
{% endif %}

{% if examples %}
## 示例
{% for ex in examples %}
输入：{{ ex.input }}
输出：{{ ex.output }}
{% endfor %}
{% endif %}

## 问题
{{ question }}
"""

template = Template(template_str)

prompt_text = template.render(
    role="医疗",
    style="通俗易懂",
    context="""头痛（headache）是临床最常见的症状之一，通常将局限于头颅上半部，包括眉弓、耳轮上缘和枕外隆突连线以上部位的疼痛统称头痛。头痛是最常见的神经系统疾病之一，据统计全球有47%的成年人至少出现过1次头痛症状。
头痛病因繁多，可分为原发性头痛和继发性头痛，原发性头痛指不明病因引起的头痛，临床上主要以原发性头痛为主，其中紧张型头痛､偏头痛最常见｡继发性头痛包括各种颅内病变引起的头痛，如脑血管疾病、颅内感染、颅脑外伤等全身性疾病如发热、内环境紊乱以及滥用精神活性药物。头痛不仅影响患者的日常生活和工作能力，还可能导致情绪障碍、睡眠障碍等。因此，及时诊断和治疗头痛非常重要。
头痛以病因治疗为主，包括抗感染治疗、降颅压、颅内肿瘤手术切除等，对于病因不能立即纠正的头痛，给予止痛等对症治疗，慢性头痛呈反复发作者应给予适当的预防性治疗。大部分头痛是良性的，常常病因不明，原发性头痛易反复发作，严重影响生活质量，需要药物控制症状，但某些继发性头痛可能相当严重，有时甚至危及生命。""",
    examples=[
        {"input": "发烧怎么办", "output": "建议测量体温，如果超过38.5℃..."},
        {"input": "咳嗽吃什么药", "output": "咳嗽需要区分干咳或湿咳..."}
    ],
    question="头痛应该挂什么科？"
)

messages = [
    {"role": "system", "content": "你是一个医疗健康助手"},
    {"role": "user", "content": prompt_text}
]

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