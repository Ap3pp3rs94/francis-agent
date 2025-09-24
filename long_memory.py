from __future__ import annotations
import json, csv, re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Iterable

MEMORY_DIR = Path(r"C:\Francis\memory")
DATA_DIR = Path(r"C:\Francis\data")
INDEX_FILE = DATA_DIR / "long_memory.json"

MEMORY_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_index() -> List[Dict[str, Any]]:
    if INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_index(index: List[Dict[str, Any]]):
    INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _normalize_tags(tags: Iterable[str] | None) -> List[str]:
    if not tags:
        return []
    out = []
    for t in tags:
        t = (t or "").strip()
        if t:
            out.append(t.lower())
    # de-dup preserving order
    seen = set()
    res = []
    for t in out:
        if t not in seen:
            seen.add(t)
            res.append(t)
    return res


def add_memory(goal: str, plan: str, reflection: str, tags: List[str] | None = None):
    """
    Save a raw note to a file and record it in the index.
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"memory_{ts}.txt"
    fpath = MEMORY_DIR / fname
    tags = _normalize_tags(tags)

    content = (
        f"Goal: {goal}\n"
        f"Plan: {plan}\n"
        f"Reflection: {reflection}\n"
        f"Tags: {', '.join(tags)}\n"
    )
    fpath.write_text(content, encoding="utf-8")

    index = _load_index()
    index.append(
        {
            "timestamp": ts,
            "file": str(fpath),
            "goal": goal,
            "plan": plan,
            "reflection": reflection,
            "tags": tags,
        }
    )
    _save_index(index)


def list_memories(n: int = 5) -> List[Dict[str, Any]]:
    index = _load_index()
    return index[-n:]


def search_memories(keyword: str) -> List[Dict[str, Any]]:
    kw = keyword.lower().strip()
    if not kw:
        return []
    results = []
    for item in _load_index():
        hay = " ".join(
            [
                item.get("goal", ""),
                item.get("plan", ""),
                item.get("reflection", ""),
                " ".join(item.get("tags", [])),
            ]
        ).lower()
        if kw in hay:
            results.append(item)
    return results


def summarize_recent(n: int = 5) -> str:
    items = list_memories(n)
    if not items:
        return "No past memories yet."
    lines = []
    for item in items:
        ts = item.get("timestamp", "")
        goal = item.get("goal", "")
        refl = item.get("reflection", "")
        tags = ", ".join(item.get("tags", []))
        line = f"- [{ts}] Goal: {goal}\n  Reflection: {refl}"
        if tags:
            line += f"\n  Tags: {tags}"
        lines.append(line)
    return "\n".join(lines)


def summarize_search(keyword: str, n: int = 6) -> str:
    results = search_memories(keyword)
    if not results:
        return f"No memories found for '{keyword}'."
    results = results[-n:]
    lines = []
    for item in results:
        ts = item.get("timestamp", "")
        goal = item.get("goal", "")
        refl = item.get("reflection", "")
        tags = ", ".join(item.get("tags", []))
        line = f"- [{ts}] Goal: {goal}\n  Reflection: {refl}"
        if tags:
            line += f"\n  Tags: {tags}"
        lines.append(line)
    return "\n".join(lines)


def export_json(path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_index()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def export_csv(path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_index()
    cols = ["timestamp", "file", "goal", "plan", "reflection", "tags"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in data:
            r = dict(row)
            r["tags"] = ", ".join(r.get("tags", []))
            w.writerow({k: r.get(k, "") for k in cols})
    return str(path)


def prune_keep_last(k: int = 200):
    data = _load_index()
    if len(data) <= k:
        return 0
    new = data[-k:]
    _save_index(new)
    return len(data) - k


def backfill_from_folder():
    """
    One-time helper if you already had memory_*.txt files created previously.
    Safely idempotent: won't duplicate entries already present in index.
    """
    idx = _load_index()
    seen = {(item.get("file") or "") for item in idx}
    added = 0
    for p in sorted(MEMORY_DIR.glob("memory_*.txt")):
        sp = str(p)
        if sp in seen:
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        goal = re.search(r"^Goal:\s*(.*)", txt, re.M)
        plan = re.search(r"^Plan:\s*(.*)", txt, re.M)
        refl = re.search(r"^Reflection:\s*(.*)", txt, re.M)
        tags = re.search(r"^Tags:\s*(.*)", txt, re.M)
        ts = re.search(r"memory_(\d{8}_\d{6})\.txt$", p.name)
        item = {
            "timestamp": (
                ts.group(1) if ts else datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            ),
            "file": sp,
            "goal": goal.group(1) if goal else "",
            "plan": plan.group(1) if plan else "",
            "reflection": refl.group(1) if refl else "",
            "tags": (
                [t.strip() for t in (tags.group(1).split(","))]
                if tags and tags.group(1).strip()
                else []
            ),
        }
        idx.append(item)
        added += 1
    _save_index(idx)
    return added
