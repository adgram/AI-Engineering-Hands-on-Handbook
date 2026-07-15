import os, yaml
from pathlib import Path

_DIR = Path(__file__).parent

class PromptManager:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = str(_DIR / "prompts.yaml")
        self.config_path = config_path
        self.templates = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.templates = yaml.safe_load(f) or {}

    def register(self, name: str, system: str, user_template: str):
        self.templates[name] = {
            "system": system,
            "user": user_template
        }

    def render(self, name: str, **kwargs) -> list[dict]:
        tpl = self.templates.get(name)
        if not tpl:
            raise ValueError(f"模板 '{name}' 不存在")
        return [
            {"role": "system", "content": tpl["system"]},
            {"role": "user", "content": tpl["user"].format(**kwargs)}
        ]

    def list_templates(self) -> list[str]:
        return list(self.templates.keys())

    def delete(self, name: str):
        self.templates.pop(name, None)

    def save(self, config_path: str = None):
        if config_path is None:
            config_path = self.config_path
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.templates, f, allow_unicode=True)