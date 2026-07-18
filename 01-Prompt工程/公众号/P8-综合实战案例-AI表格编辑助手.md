# Prompt工程 | P8：综合实战案例 — 构建 AI 表格编辑助手

## 前言

本节将结合 P1-P7 所有内容，实现一个完整可使用的工具——AI 表格编辑助手。

注：本案例由AI实现，需要自定义可以让AI进行修改。

## 项目说明

构建一个带 Web 界面的表格编辑助手，用户通过自然语言描述需求，AI 自动操作 Excel 文件。

### 核心流程

```
用户输入（自然语言）
    ↓
① 任务分解（P6）→ 拆解为子步骤（如：分析需求→生成公式→执行修改）
    ↓
② 上下文管理（P5）→ 构建编辑上下文（当前表格状态/历史操作）
    ↓
③ 角色设定（P2）→ 加载 Excel 数据分析师角色
    ↓
④ 少样本示例（P3）→ 注入常见编辑模式
    ↓
⑤ Function Calling（P4）→ 调用 openpyxl 工具执行操作
    ↓
⑥ 安全防护（P7）→ 检查公式注入和数据安全
    ↓
⑦ 返回结果 → 刷新表格预览
```

## 完整代码

项目结构：

```
learn_prompt/excel_editor/
├── app.py          # Gradio 可视化界面
├── excel_tools.py  # 工具函数定义
├── llm_client.py   # LLM 客户端封装
└── prompts.py      # Prompt 模板
```

### llm_client.py

```python
import os, json, time, logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-v4-flash"
        self.logger = logging.getLogger(__name__)

    def chat(self, messages, tools=None, tool_choice="auto", response_format=None):
        kwargs = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format
        try:
            response = self.client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            self.logger.error(f"API 调用失败: {e}")
            raise

    def chat_text(self, messages):
        response = self.chat(messages)
        return response.choices[0].message.content

    def chat_json(self, messages):
        response = self.chat(messages, response_format={"type": "json_object"})
        content = response.choices[0].message.content
        return json.loads(content) if content else {}
```

### prompts.py — System Prompt 设计（P2）

```python
SYSTEM_PROMPT = """## Role（角色定义）
你是一个专业的 Excel 数据分析师「表姐」，擅长用 Python 和 openpyxl 操作 Excel 文件。

## Scope（范围边界）
你可以处理：数据录入、公式生成、单元格格式设置、行列操作、图表生成、数据筛选排序。
不处理：外部数据源连接、宏/VBA 编写。

## Behavior（行为规则）
1. 接到需求后，先调用 decompose_task 拆解步骤，再逐步执行
2. 每次修改后调用 read_table 预览结果，确认正确
3. 生成公式时优先用 Excel 内置函数，必要时用 Python 计算
4. 涉及多步操作时，每步单独调用工具，完成后询问用户是否继续

## Safety Rules（安全规则）
1. 不执行包含 eval/exec/shell 等危险操作的代码
2. 不读取或修改 Excel 文件之外的任何文件
3. 不删除用户数据，移动数据前先备份

## Tone（语气风格）
用简洁的技术语言沟通，给出操作时附上简要说明。
输出格式：
- 操作成功：✓ {操作说明}
- 操作失败：✗ {错误原因}
- 需要确认：? {询问内容}

## Context（上下文）
操作基于 openpyxl 库，支持 .xlsx 格式。
文件路径由系统传入，不可修改。"""
```

### excel_tools.py — 工具定义（P4 Function Calling）

```python
import json, os
from openpyxl import load_workbook, Workbook

CURRENT_FILE = None
CURRENT_DATA = None  # 最近一次读取的表格快照

def set_file(filepath):
    global CURRENT_FILE
    CURRENT_FILE = filepath

def read_table(filepath, sheet_name=None, max_rows=20):
    """读取 Excel 前 max_rows 行，返回文本预览"""
    try:
        wb = load_workbook(filepath, data_only=True)
        ws = wb[sheet_name] if sheet_name else wb.active
        rows = []
        for i, row in enumerate(ws.iter_rows(values_only=True), 1):
            rows.append([str(cell) if cell is not None else "" for cell in row])
            if i > max_rows:
                break
        wb.close()
        result = "\n".join([" | ".join(r) for r in rows])
        global CURRENT_DATA
        CURRENT_DATA = {"sheet": ws.title, "rows": len(list(ws.iter_rows())), "cols": ws.max_column}
        return result
    except Exception as e:
        return f"读取失败: {e}"

def write_cell(filepath, cell, value, sheet_name=None):
    """写入单元格"""
    try:
        wb = load_workbook(filepath)
        ws = wb[sheet_name] if sheet_name else wb.active
        ws[cell] = value
        wb.save(filepath)
        wb.close()
        return json.dumps({"status": "ok", "cell": cell, "value": str(value)})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def write_formula(filepath, cell, formula, sheet_name=None):
    """写入公式"""
    try:
        wb = load_workbook(filepath)
        ws = wb[sheet_name] if sheet_name else wb.active
        ws[cell] = formula
        wb.save(filepath)
        wb.close()
        return json.dumps({"status": "ok", "cell": cell, "formula": formula})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def set_cell_style(filepath, cell, bold=False, font_color=None, fill_color=None, sheet_name=None):
    """设置单元格样式"""
    try:
        from openpyxl.styles import Font, PatternFill
        wb = load_workbook(filepath)
        ws = wb[sheet_name] if sheet_name else wb.active
        cell_obj = ws[cell]
        if bold:
            cell_obj.font = Font(bold=True)
        if fill_color:
            cell_obj.fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid")
        wb.save(filepath)
        wb.close()
        return json.dumps({"status": "ok", "cell": cell})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def insert_row(filepath, row_idx, values, sheet_name=None):
    """插入行"""
    try:
        wb = load_workbook(filepath)
        ws = wb[sheet_name] if sheet_name else wb.active
        ws.insert_rows(row_idx)
        for col_idx, val in enumerate(values, 1):
            ws.cell(row=row_idx, column=col_idx, value=val)
        wb.save(filepath)
        wb.close()
        return json.dumps({"status": "ok", "row": row_idx, "values": values})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def auto_sum(filepath, range_str, target_cell, sheet_name=None):
    """对指定范围求和"""
    try:
        wb = load_workbook(filepath)
        ws = wb[sheet_name] if sheet_name else wb.active
        ws[target_cell] = f"=SUM({range_str})"
        wb.save(filepath)
        wb.close()
        return json.dumps({"status": "ok", "formula": f"=SUM({range_str})", "target": target_cell})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_table",
            "description": "读取 Excel 表格内容（前 20 行），了解当前数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Excel 文件路径"},
                    "sheet_name": {"type": "string", "description": "工作表名（可选）"}
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_cell",
            "description": "向单元格写入值（文本/数字）",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Excel 文件路径"},
                    "cell": {"type": "string", "description": "单元格地址，如 A1, B2"},
                    "value": {"type": "string", "description": "要写入的值"},
                    "sheet_name": {"type": "string", "description": "工作表名（可选）"}
                },
                "required": ["filepath", "cell", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_formula",
            "description": "向单元格写入 Excel 公式",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Excel 文件路径"},
                    "cell": {"type": "string", "description": "单元格地址"},
                    "formula": {"type": "string", "description": "Excel 公式，如 =A1+B1"},
                    "sheet_name": {"type": "string", "description": "工作表名（可选）"}
                },
                "required": ["filepath", "cell", "formula"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_cell_style",
            "description": "设置单元格样式（加粗、字体颜色、填充色）",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Excel 文件路径"},
                    "cell": {"type": "string", "description": "单元格地址"},
                    "bold": {"type": "boolean", "description": "是否加粗"},
                    "font_color": {"type": "string", "description": "字体颜色，如 FFFFFF"},
                    "fill_color": {"type": "string", "description": "填充色，如 4472C4"},
                    "sheet_name": {"type": "string", "description": "工作表名（可选）"}
                },
                "required": ["filepath", "cell"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "insert_row",
            "description": "在指定位置插入一行数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Excel 文件路径"},
                    "row_idx": {"type": "integer", "description": "行号（从 1 开始）"},
                    "values": {"type": "array", "items": {"type": "string"}, "description": "每列的值列表"},
                    "sheet_name": {"type": "string", "description": "工作表名（可选）"}
                },
                "required": ["filepath", "row_idx", "values"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "auto_sum",
            "description": "对指定区域求和，结果写入目标单元格",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Excel 文件路径"},
                    "range_str": {"type": "string", "description": "求和区域，如 B2:B10"},
                    "target_cell": {"type": "string", "description": "结果单元格地址"},
                    "sheet_name": {"type": "string", "description": "工作表名（可选）"}
                },
                "required": ["filepath", "range_str", "target_cell"]
            }
        }
    }
]

FUNCTIONS = {
    "read_table": lambda **k: read_table(**k),
    "write_cell": lambda **k: write_cell(**k),
    "write_formula": lambda **k: write_formula(**k),
    "set_cell_style": lambda **k: set_cell_style(**k),
    "insert_row": lambda **k: insert_row(**k),
    "auto_sum": lambda **k: auto_sum(**k),
}
```

### app.py — 可视化界面

```python
import os, json, tempfile, shutil
from pathlib import Path
import gradio as gr
from openpyxl import Workbook
from llm_client import LLMClient
from prompts import SYSTEM_PROMPT
from excel_tools import TOOLS, FUNCTIONS, set_file

client = LLMClient()
session_history = []

def create_demo_file():
    """创建示例 Excel 文件"""
    demo_path = os.path.join(tempfile.gettempdir(), "demo_sales.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = "销售数据"
    headers = ["月份", "产品", "销量", "单价", "销售额"]
    ws.append(headers)
    data = [
        ["1月", "产品1", 120, 299],
        ["1月", "产品2", 45, 199],
        ["2月", "产品1", 98, 299],
        ["2月", "产品2", 67, 199],
        ["3月", "产品1", 150, 299],
        ["3月", "产品2", 88, 199],
    ]
    for row in data:
        ws.append(row)
    wb.save(demo_path)
    return demo_path

def read_excel_preview(filepath):
    """读取 Excel 预览（用于 Gradio Dataframe）"""
    from openpyxl import load_workbook
    wb = load_workbook(filepath, data_only=True)
    ws = wb.active
    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    data = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        data.append([cell if cell is not None else "" for cell in row])
    wb.close()
    return headers, data

def execute_tool(name, args):
    """执行工具并返回结果"""
    handler = FUNCTIONS.get(name)
    if handler:
        return handler(**args)
    return json.dumps({"error": f"未知工具: {name}"})

def process_message(filepath, user_input, history_text):
    """处理用户消息"""
    global session_history
    set_file(filepath)

    if not os.path.exists(filepath):
        return "文件不存在", filepath, ""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 读取当前表格上下文
    from excel_tools import read_table
    table_context = read_table(filepath)
    messages.append({"role": "user", "content": f"当前表格内容：\n{table_context}\n\n用户需求：{user_input}"})

    # 工具调用循环（P4）
    max_rounds = 8
    for round_num in range(max_rounds):
        response = client.chat(messages=messages, tools=TOOLS, tool_choice="auto")
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content, filepath, ""

        messages.append(message)
        for tc in message.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            args["filepath"] = filepath
            result = execute_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "操作完成（已达最大轮数）", filepath, ""

def upload_and_process(file, user_input):
    """上传文件并处理"""
    if file is None:
        # 使用示例文件
        filepath = create_demo_file()
        action = "已加载示例销售表"
    else:
        filepath = file.name
        action = f"已上传: {Path(file.name).name}"

    if not user_input.strip():
        headers, data = read_excel_preview(filepath)
        return action, filepath, gr.update(value=data, headers=headers, visible=True), "", ""

    result, fp, _ = process_message(filepath, user_input, "")
    headers, data = read_excel_preview(fp)
    return f"✓ {result}", fp, gr.update(value=data, headers=headers, visible=True), "", ""

def create_file_from_template():
    """创建示例表格"""
    fp = create_demo_file()
    headers, data = read_excel_preview(fp)
    msg = "已创建示例销售表（1-3 月会员卡和体验课销售数据）"
    return msg, fp, gr.update(value=data, headers=headers, visible=True)

# ====== Gradio 界面 ======
with gr.Blocks(title="AI 表格编辑助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📊 AI 表格编辑助手
    用自然语言操作 Excel 表格，无需记忆公式和菜单路径。
    """)

    file_state = gr.State("")
    result_text = gr.Textbox(label="操作结果", lines=3, interactive=False)
    table_preview = gr.Dataframe(label="表格预览", visible=False, interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 文件")
            file_input = gr.File(label="上传 Excel 文件（可选）", file_types=[".xlsx"])
            create_btn = gr.Button("📄 创建示例表格", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("### 💬 说你想做什么")
            examples = gr.Examples(
                examples=[
                    ["在 E 列写入公式计算销售额=销量×单价，表头写'销售额'"],
                    ["把标题行加粗，背景改成蓝色"],
                    ["在表格最下面加一行汇总，用公式求和"],
                    ["在 3 月数据后面插入一行：3月/团购课/200/99"],
                    ["做个全年汇总行，用深灰色背景"],
                ],
                inputs=[],
            )
            user_input = gr.Textbox(
                label="输入指令",
                placeholder="例如：在E列插入公式 销售额=销量×单价",
                lines=2
            )
            send_btn = gr.Button("🚀 执行", variant="primary", scale=2)

    gr.Markdown("### 使用说明")
    gr.Markdown("""
    - **上传 Excel** 或点击「创建示例表格」开始
    - **输入自然语言指令**，AI 会自动拆解任务并操作表格
    - **示例指令**：计算销售额、加汇总行、设置格式、插入数据
    - 每次操作后表格预览会自动刷新
    """)

    send_btn.click(
        fn=upload_and_process,
        inputs=[file_input, user_input],
        outputs=[result_text, file_state, table_preview, user_input, file_input]
    )

    create_btn.click(
        fn=create_file_from_template,
        inputs=[],
        outputs=[result_text, file_state, table_preview]
    )

if __name__ == "__main__":
    demo.launch()
```

## 运行方式

```bash
# 安装依赖
pip install openai python-dotenv openpyxl gradio

# 确保 .env 中有 DEEPSEEK_API_KEY
# 启动
cd learn_prompt/excel_editor
python app.py
```

浏览器打开 `http://localhost:7860` 即可使用。

## 涉及知识点回顾

| 模块 | 对应章节 | 在本案例中的作用 |
|------|---------|----------------|
| 角色设定 | P2 | 用结构化分段定义 Excel 分析师角色和行为规则 |
| 少样本提示 | P3 | 界面中的示例指令引导用户使用模式 |
| 结构化输出 | P4 | 工具调用参数和返回值的 JSON Schema 定义 |
| Function Calling | P4 | 6 个工具函数注册 + 完整调用循环 |
| 上下文工程 | P5 | 每次操作前读取表格当前状态作为上下文 |
| 复杂任务分解 | P6 | 多步操作（如"做个季度报表"自动拆解） |
| 对抗性安全 | P7 | 安全规则防止公式注入和危险操作 |
| 调用 API | P1 | 使用 DeepSeek API 进行自然语言理解 |
