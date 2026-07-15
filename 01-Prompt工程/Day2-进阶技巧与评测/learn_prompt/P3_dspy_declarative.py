import sys, os, dspy
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent/"common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient
client = LLMClient()

# ====== 传统手写方式 ======
patient_text = "患者：李某，男，45岁。主诉：咳嗽两周，痰黄。用药：阿莫西林胶囊 0.5g tid，复方甘草口服液 10ml tid。"

prompt = """从以下患者记录中提取药物名称。
规则：
- 只提取药物名称，不提取剂量、频次
- 商品名和化学名都算
- 中药方剂每味药材单独提取

患者记录：{text}
药物："""

response = client.chat(
    messages=[{"role": "user", "content": prompt.format(text=patient_text)}]
)
handwritten_result = response.choices[0].message.content
print(f"[手写方式] 提取结果: {handwritten_result}")

# ====== DSPy 3.0 声明式方式 ======
lm = dspy.LM("openai/deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), api_base="https://api.deepseek.com")
dspy.configure(lm=lm)

class DrugExtraction(dspy.Signature):
    """从患者记录中提取所有药物名称（包括化学名、商品名、中药药材、疫苗）。"""
    patient_note: str = dspy.InputField(desc="患者记录文本")
    drugs: list[str] = dspy.OutputField(desc="提取的药物名称列表")

class DrugExtractor(dspy.Module):
    def __init__(self):
        self.extractor = dspy.Predict(DrugExtraction)

    def forward(self, patient_note):
        return self.extractor(patient_note=patient_note)

# 准备训练数据 & 验证函数
trainset = [
    dspy.Example(patient_note="患者服用头孢克肟和布洛芬", drugs=["头孢克肟", "布洛芬"]).with_inputs("patient_note"),
    dspy.Example(patient_note="处方：阿奇霉素、氯雷他定", drugs=["阿奇霉素", "氯雷他定"]).with_inputs("patient_note"),
]

def validate_drugs(example, pred, trace=None):
    return set(example.drugs) == set(pred.drugs)

# 编译器自动优化
extractor = DrugExtractor()
optimizer = dspy.teleprompt.BootstrapFewShot(metric=validate_drugs)
optimized = optimizer.compile(extractor, trainset=trainset)

# 测试优化后的模型
test_note = "患者服用了阿莫西林和头孢克肟，同时开了布洛芬退烧"
result = optimized(patient_note=test_note)
print(f"[声明式-优化前] 预测: {extractor(patient_note=test_note).drugs}")
print(f"[声明式-优化后] 预测: {result.drugs}")

output = f"手写结果: {handwritten_result}\n声明式(优化前): {extractor(patient_note=test_note).drugs}\n声明式(优化后): {result.drugs}"
output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)
print(f"结果已写入 {output_file_name}")
