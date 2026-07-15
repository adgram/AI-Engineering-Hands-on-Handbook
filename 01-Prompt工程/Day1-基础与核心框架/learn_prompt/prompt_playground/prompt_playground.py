import sys, os, json
from pathlib import Path
_PROJECT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT.parent.parent.parent.parent / "common"))
from llm_client import LLMClient
from prompt_manager import PromptManager
from evaluator import Evaluator

client = LLMClient()
pm = PromptManager()
evaluator = Evaluator()
TEST_CASES_PATH = _PROJECT / "test_cases.json"

def load_test_cases(path=None):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(TEST_CASES_PATH):
        with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def interactive_mode():
    print("Prompt Playground v2.0")
    print(f"默认测试用例: {TEST_CASES_PATH} ({len(load_test_cases())}条)")
    print("命令: /templates /new /eval /usage /save /help /benchmark /quit")

    while True:
        cmd = input("\n> ").strip()
        if cmd == "/quit":
            break
        elif cmd == "/templates":
            for name in pm.list_templates():
                print(f"  - {name}")
        elif cmd == "/new":
            name = input("模板名: ")
            system = input("System Prompt: ")
            user_tpl = input("User 模板 (用 {var} 做变量): ")
            pm.register(name, system, user_tpl)
            pm.save()
            print(f"模板 '{name}' 已保存")
        elif cmd == "/eval":
            template_name = input("测试哪个模板?: ")
            if template_name not in pm.list_templates():
                print("模板不存在")
                continue
            test_file = input("测试用例 JSON 路径 (留空默认): ")
            test_cases = load_test_cases(test_file.strip() if test_file.strip() else None)
            if not test_cases:
                print("无测试用例，跳过")
                continue
            system_prompt = pm.templates[template_name]["system"]
            evaluator.run_test(
                test_cases,
                system_prompt,
                judge_function=lambda out, exp: exp in out,
                verbose=True
            )
        elif cmd == "/benchmark":
            names = pm.list_templates()
            if not names:
                print("无模板")
                continue
            test_file = input("测试用例 JSON 路径 (留空默认): ")
            test_cases = load_test_cases(test_file.strip() if test_file.strip() else None)
            if not test_cases:
                print("无测试用例，跳过")
                continue
            for name in names:
                sp = pm.templates[name]["system"]
                result = evaluator.run_test(test_cases, sp, lambda out, exp: exp in out)
                print(f"  [{name}] 准确率: {result['accuracy']:.0%} ({result['correct']}/{result['total']})")
        elif cmd == "/usage":
            stats = client.get_usage_stats()
            print(json.dumps(stats, indent=2, ensure_ascii=False))
        elif cmd == "/help":
            print("命令列表:")
            print("  /templates  - 列出所有模板")
            print("  /new        - 创建新模板")
            print("  /eval       - 测试单个模板效果")
            print("  /benchmark  - 对比所有模板效果")
            print("  /usage      - 查看用量统计")
            print("  /save       - 保存配置")
            print("  /quit       - 退出")
        elif cmd.startswith("/"):
            print(f"未知命令: {cmd}")
        else:
            if pm.list_templates():
                name = pm.list_templates()[0]
                tpl = pm.templates.get(name, {})
                user_tpl = tpl.get("user", "{input}")
                if "{input}" in user_tpl or "{text}" in user_tpl:
                    messages = pm.render(name, input=cmd, text=cmd, context=cmd, question=cmd)
                else:
                    messages = [{"role": "user", "content": cmd}]
            else:
                messages = [{"role": "user", "content": cmd}]
            reply = client.chat_text(messages)
            print(f"\n{reply}")

if __name__ == "__main__":
    interactive_mode()
