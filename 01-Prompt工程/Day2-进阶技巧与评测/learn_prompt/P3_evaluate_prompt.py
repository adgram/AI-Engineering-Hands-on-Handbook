import sys
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent/"common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient
client = LLMClient()

test_cases = [
    {"input": "这款手机拍照一流，但续航一般", "expected": "正面"},
    {"input": "用了三天就坏了，质量太差", "expected": "负面"},
    {"input": "价格适中，功能齐全", "expected": "正面"},
    {"input": "一般般吧，没什么特别的", "expected": "中性"},
    {"input": "客服态度差还不给退款", "expected": "负面"},
    {"input": "包装精美，送人很有面子", "expected": "正面"},
    {"input": "功能很多但大部分用不上", "expected": "中性"},
    {"input": "性价比很高，推荐购买", "expected": "正面"},
    {"input": "物流太慢了等了一个星期", "expected": "负面"},
    {"input": "和描述一致，没有惊喜也没有失望", "expected": "中性"}
]

def evaluate_prompt(system_prompt, test_cases):
    correct = 0
    results = []
    tag = len(system_prompt)
    for case in test_cases:
        resp = client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'{tag}. {case["input"]}'}
            ],
        )
        output = resp.choices[0].message.content.strip()
        is_correct = case["expected"] in output
        if is_correct:
            correct += 1
        results.append({"input": case["input"], "output": output, "expected": case["expected"], "correct": is_correct})

    accuracy = correct / len(test_cases)
    output_str = f"准确率: {accuracy:.0%} ({correct}/{len(test_cases)})\n"
    for r in results:
        if not r["correct"]:
            output_str += f"  ❌ 期望={r['expected']}, 得到={r['output']}\n"
    output_cache["results"] = output_str
    return accuracy

output_cache = {}

v1 = "分析用户评论的情感倾向，只输出正面、负面或中性"
output_lines = ["=== Prompt V1 ==="]
acc1 = evaluate_prompt(v1, test_cases)
output_lines.append(output_cache["results"])

v2 = "你是一个情感分析专家。分析以下评论的情感倾向。\n规则：\n1. 只输出正面、负面或中性\n2. 如果评论中既有正面也有负面，以整体倾向为准\n3. 不要输出任何解释\n\n"
output_lines.append("\n=== Prompt V2 ===")
acc2 = evaluate_prompt(v2, test_cases)
output_lines.append(output_cache["results"])

v3 = v2 + "\n\n示例：\n评论：'性价比很高，推荐' → 正面\n评论：'质量太差了' → 负面\n评论：'还行吧，能用' → 中性"
output_lines.append("\n=== Prompt V3 ===")
acc3 = evaluate_prompt(v3, test_cases)
output_lines.append(output_cache["results"])

output_lines.append(f"\n=== 总结 ===")
output_lines.append(f"V1: {acc1:.0%} → V2: {acc2:.0%} → V3: {acc3:.0%}")

output = "\n".join(output_lines)
output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)
print(f"结果已写入 {output_file_name}")
