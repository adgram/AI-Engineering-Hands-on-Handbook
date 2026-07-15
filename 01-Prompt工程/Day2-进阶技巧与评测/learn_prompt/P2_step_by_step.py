import sys
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent/"common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient
client = LLMClient()

def step_by_step(task: str) -> str:
    messages = [
        {"role": "system", "content": "你是一个任务分解专家。将用户的任务拆解为 3-5 个独立的子任务。"},
        {"role": "user", "content": task}
    ]
    resp = client.chat(messages=messages)
    subtasks = resp.choices[0].message.content
    output_str = "=== 任务分解 ===\n" + subtasks + "\n"
    messages_2 = [
        {"role": "system", "content": "逐步执行以下子任务，输出每个步骤的结果。"},
        {"role": "user", "content": subtasks}
    ]
    resp2 = client.chat(messages=messages_2)
    output_str += "\n=== 执行结果 ===\n" + resp2.choices[0].message.content
    return output_str

if __name__ == "__main__":
    result = step_by_step("撰写一篇关于 AI Agent 的博客大纲，包含引言、三个主体段落和结论")

    output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
    with open(output_file_name, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"结果已写入 {output_file_name}")