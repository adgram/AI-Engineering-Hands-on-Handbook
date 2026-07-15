import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

# 使用特殊格式，模型不会自然采用，必须通过示例学会
task = "将产品描述转为营销短文案，格式：【品类】核心卖点1 | 核心卖点2，20字以内"

# 每组测试用不同产品描述，避免缓存命中相同输入
test_cases = [
    "采用 AI 降噪技术的无线蓝牙耳机，续航 30 小时",
    "纯天然有机冷萃咖啡液，0糖0脂，每盒10条装",
]

examples = [
    {"user": "有机认证的云南咖啡豆，500g 装", "assistant": "【咖啡豆】有机云南 | 500g装"},
    {"user": "纯棉圆领 T 恤，黑白灰三色可选", "assistant": "【T恤】纯棉圆领 | 三色可选"},
    {"user": "不锈钢保温杯，500ml，12小时保温", "assistant": "【保温杯】不锈钢 | 12h保温"},
    {"user": "USB-C 转 HDMI 转换器，4K 60Hz", "assistant": "【转换器】USB-C转HDMI | 4K高清"},
    {"user": "运动跑鞋，网面透气，柔软缓震", "assistant": "【跑鞋】网面透气 | 柔软缓震"},
]

output = ""

for test_input in test_cases:
    output += f"\n{'='*40}\n测试输入: {test_input}\n{'='*40}\n\n"

    for label, msg_list in [
        ("Zero-shot", [{"role": "system", "content": task}, {"role": "user", "content": test_input}]),
        ("1-shot", [{"role": "system", "content": task},
                    {"role": "user", "content": examples[0]["user"]},
                    {"role": "assistant", "content": examples[0]["assistant"]},
                    {"role": "user", "content": test_input}]),
        ("3-shot", [{"role": "system", "content": task}] + [m for ex in examples[:3] for m in ({"role": "user", "content": ex["user"]}, {"role": "assistant", "content": ex["assistant"]})] + [{"role": "user", "content": test_input}]),
        ("5-shot", [{"role": "system", "content": task}] + [m for ex in examples for m in ({"role": "user", "content": ex["user"]}, {"role": "assistant", "content": ex["assistant"]})] + [{"role": "user", "content": test_input}]),
    ]:
        resp = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=msg_list,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
            max_tokens=1000
        )
        reasoning = getattr(resp.choices[0].message, "reasoning_content", "")
        content = resp.choices[0].message.content
        output += f"--- {label} ---\n"
        if reasoning:
            output += f"思考过程: {reasoning}\n\n"
        output += f"回复: {content}\n\n"

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")
