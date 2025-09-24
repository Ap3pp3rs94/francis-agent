from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

REFLECTIONS_PATH = Path(r"C:\Francis\data\reflections.json")
REFLECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class Reflection:
    timestamp: str
    goal: str
    plan: str
    outcome: Optional[str]
    reflection: str


class ReflectionMemory:
    def __init__(self, path: Path = REFLECTIONS_PATH, max_items: int = 24):
        self.path = path
        self.max_items = max_items
        self._cache: List[Reflection] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._cache = [Reflection(**item) for item in data][-self.max_items :]
            except Exception:
                self._cache = []
        else:
            self._cache = []

    def save(self):
        self.path.write_text(
            json.dumps([asdict(r) for r in self._cache][-self.max_items :], indent=2),
            encoding="utf-8",
        )

    def add(
        self, goal: str, plan: str, reflection_text: str, outcome: Optional[str] = None
    ):
        self._cache.append(
            Reflection(
                timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                goal=goal.strip(),
                plan=plan.strip(),
                outcome=(outcome or "").strip(),
                reflection=reflection_text.strip(),
            )
        )
        self.save()

    def tail(self, n: int = 3) -> List[Reflection]:
        return self._cache[-n:]

    def as_bullets(self, n: int = 3) -> str:
        items = self.tail(n)
        if not items:
            return "None."
        return "\n".join(
            f"- [{r.timestamp}] Goal: {r.goal}\n  Plan: {r.plan}\n  Reflection: {r.reflection}"
            for r in items
        )

    def clear(self):
        self._cache = []
        self.save()

    def export(self, export_path: Path):
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(
            json.dumps([asdict(r) for r in self._cache], indent=2), encoding="utf-8"
        )
