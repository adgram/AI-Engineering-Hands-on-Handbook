import sys, json
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent.parent / "common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient
from context_manager import ContextManager
from task_planner import TaskPlanner
from prompt_optimizer import PromptOptimizer, OptimizerConfig
from guardrails import DefensePipeline
from tools import TOOL_SCHEMAS, TOOL_HANDLERS
from harness import ToolRegistry, ControlLoop, Tracer

client = LLMClient()
tracer = Tracer()


def setup_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for schema in TOOL_SCHEMAS:
        name = schema["function"]["name"]
        handler = TOOL_HANDLERS[name]
        registry.register(name, schema, handler)
    return registry


def handle_request(request: str) -> str:
    tracer.log("input", request)

    defense = DefensePipeline()
    safe_input = defense.process_input(request)
    tracer.log("guardrail", f"输入检查: {'通过' if safe_input == request else '拦截'}")

    ctx = ContextManager("你是一个智能客服助手。请礼貌、专业地回复客户。")
    messages = ctx.build_context(safe_input)

    planner = TaskPlanner()
    plan = planner.plan_task(safe_input)
    tracer.log("plan", json.dumps(plan, ensure_ascii=False))

    registry = setup_registry()
    loop = ControlLoop(registry)
    reply = loop.run(messages)
    tracer.log("result", reply)

    safe_output = defense.verify_output(reply)
    if safe_output != reply:
        tracer.log("guardrail", "输出被拦截")

    ctx.add_message("assistant", safe_output)
    return safe_output


def interactive_mode():
    print("=" * 50)
    print("  智能客服工作台 v1.0")
    print("  集成: 上下文管理 | 任务分解 | 安全防护 | 工具调用")
    print("=" * 50)
    print("命令: /plan /optimize /trace /quit")
    print()

    ctx = ContextManager("你是一个智能客服助手。请礼貌、专业地回复客户。")

    while True:
        user_input = input("\n客户: ").strip()
        if user_input == "/quit":
            break
        elif user_input == "/trace":
            s = tracer.summary()
            print(json.dumps(s, indent=2, ensure_ascii=False))
            continue
        elif user_input == "/plan":
            request = input("请输入需要拆解的客户请求: ")
            planner = TaskPlanner()
            plan = planner.plan_task(request)
            print(f"拆解为 {len(plan)} 步:")
            for p in plan:
                print(f"  - {p.get('action', str(p))}")
            continue
        elif user_input == "/optimize":
            template_text = input("请输入需要优化的回复模板: ")
            test_input = input("请输入测试输入: ")
            expected = input("请输入期望输出关键词: ")
            config = OptimizerConfig(
                template_sections={"回复模板": template_text},
                test_cases=[{"input": test_input, "expected": expected}],
                max_rounds=3,
            )
            optimizer = PromptOptimizer(config)
            optimizer.optimize()
            print(f"优化后的模板:\n{optimizer.assemble(optimizer.best_template)}")
            continue

        reply = handle_request(user_input)
        print(f"\n客服: {reply}")


if __name__ == "__main__":
    interactive_mode()
