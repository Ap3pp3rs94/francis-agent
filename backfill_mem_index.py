from pathlib import Path
from datetime import datetime
import json, re

MEMORY_DIR = Path(r"C:\Francis\memory")
INDEX_FILE = Path(r"C:\Francis\data\long_memory.json")
INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

index = []
if INDEX_FILE.exists():
    try:
        index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except Exception:
        index = []

seen = {(item.get("file") or "") for item in index}

for p in sorted(MEMORY_DIR.glob("memory_*.txt")):
    if str(p) in seen:
        continue
    txt = p.read_text(encoding="utf-8", errors="ignore")
    goal = re.search(r"^Goal:\s*(.*)", txt, re.M)
    goal = goal.group(1) if goal else ""
    plan = re.search(r"^Plan:\s*(.*)", txt, re.M)
    plan = plan.group(1) if plan else ""
    refl = re.search(r"^Reflection:\s*(.*)", txt, re.M)
    reflection = refl.group(1) if refl else ""
    ts_match = re.search(r"memory_(\d{8}_\d{6})\.txt$", p.name)
    ts = ts_match.group(1) if ts_match else datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    index.append({"timestamp": ts, "file": str(p), "goal": goal, "plan": plan, "reflection": reflection})

INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")
print(f"Indexed {len(index)} entries -> {INDEX_FILE}")
