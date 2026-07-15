import os, json
from datetime import datetime
from typing import Optional


class Workspace:
    def __init__(self, root_dir: str = "./agent_workspace"):
        self.root_dir = root_dir
        self.memory: dict[str, any] = {}
        self._ensure_dirs()

    def _ensure_dirs(self):
        for sub in ["files", "output", "data"]:
            os.makedirs(os.path.join(self.root_dir, sub), exist_ok=True)

    def write_file(self, path: str, content: str) -> str:
        full_path = os.path.join(self.root_dir, "files", path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {path}"

    def read_file(self, path: str) -> Optional[str]:
        full_path = os.path.join(self.root_dir, "files", path)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def save_memory(self, key: str, value: any):
        self.memory[key] = value

    def load_memory(self, key: str, default: any = None) -> any:
        return self.memory.get(key, default)

    def get_status(self) -> dict:
        return {
            "files": os.listdir(os.path.join(self.root_dir, "files")),
            "memory_keys": list(self.memory.keys()),
            "memory_size": len(json.dumps(self.memory, ensure_ascii=False)),
        }


class StateManager:
    def __init__(self):
        self.states: dict[str, any] = {}
        self.history: list[dict] = []

    def set(self, key: str, value: any):
        self.states[key] = value
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "key": key,
            "value": value
        })

    def get(self, key: str, default: any = None) -> any:
        return self.states.get(key, default)

    def get_recent_history(self, n: int = 5) -> list[dict]:
        return self.history[-n:]
