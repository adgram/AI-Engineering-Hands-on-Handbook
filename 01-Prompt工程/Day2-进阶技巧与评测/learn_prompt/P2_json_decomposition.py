import sys, json
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent/"common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient
client = LLMClient()

task = "为一家新开的咖啡店制定开业营销方案"

messages = [
    {"role": "system", "content": "将用户的任务分解为子任务列表，输出 JSON 格式：{\"subtasks\": [{\"id\": 1, \"name\": \"...\", \"description\": \"...\", \"priority\": \"high/medium/low\"}]}"},
    {"role": "user", "content": task}
]

response = client.chat(
    messages=messages,
    response_format={"type": "json_object"},
)

plan = json.loads(response.choices[0].message.content)
output = f"总共 {len(plan['subtasks'])} 个子任务:\n"
for st in plan['subtasks']:
    output += f"  [{st['priority']}] {st['id']}. {st['name']}: {st['description']}\n"

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)
print(f"结果已写入 {output_file_name}")