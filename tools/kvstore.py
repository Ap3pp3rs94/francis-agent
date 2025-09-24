# tools/kvstore.py
"""
KVStoreTool — tiny JSON key–value store for Francis

Keys map to JSON files under:  <files.root>/data/<key>.json

Supported ops:
  - get key="profile"
  - set key="profile" value='{"name":"Ap3pp","level":"ultra"}'
  - merge key="profile" value='{"level":"omega"}'         # shallow merge dicts
  - incr key="counter" field="requests" by="2"            # increments integer in an object
  - delete key="profile"
  - list prefix="pro"                                     # lists keys starting with prefix
  - exists key="profile"

Notes:
  • All values must be valid JSON (object/array/number/string/bool/null)
  • Keys are sanitized to [A-Za-z0-9._-] and max 80 chars
  • Enforces files.root sandbox from Francis config
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from __main__ import Tool, ToolResult, CFG
except Exception:

    class Tool: ...

    class ToolResult: ...

    CFG = type("CFG", (), {"data": {}})()

_VALID_KEY = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


class KVStoreTool(Tool):
    name = "kvstore"
    description = "Persistent JSON KV store under data/."

    def _root(self) -> Path:
        return Path(CFG.data["files"]["root"]).resolve() / "data"

    def _path_for(self, key: str) -> Path:
        if not key or not _VALID_KEY.match(key):
            raise ValueError("Invalid key: use [A-Za-z0-9._-] up to 80 chars")
        root = self._root()
        p = (root / f"{key}.json").resolve()
        if not str(p).startswith(str(root.resolve())):
            raise PermissionError("Path escapes files.root/data")
        return p

    def _read_json(self, p: Path) -> Any:
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def _write_json(self, p: Path, value: Any):
        p.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(value, ensure_ascii=False, indent=2)
        # size guard (1MB)
        if len(text.encode("utf-8")) > 1_000_000:
            raise ValueError("Value too large (>1MB)")
        p.write_text(text, encoding="utf-8")

    def run(
        self,
        op: str,
        key: str = "",
        value: str = "",
        prefix: str = "",
        field: str = "",
        by: str = "1",
    ) -> ToolResult:
        op = (op or "").lower().strip()
        try:
            if op == "list":
                root = self._root()
                root.mkdir(parents=True, exist_ok=True)
                keys: List[str] = []
                for fp in root.glob("*.json"):
                    k = fp.stem
                    if prefix and not k.startswith(prefix):
                        continue
                    keys.append(k)
                return ToolResult(
                    True,
                    "\n".join(sorted(keys)),
                    {"count": len(keys), "prefix": prefix},
                )

            if op == "exists":
                if not key:
                    return ToolResult(False, "Missing 'key'", {})
                p = self._path_for(key)
                return ToolResult(
                    True, str(p.exists()).lower(), {"exists": p.exists(), "key": key}
                )

            if op == "get":
                if not key:
                    return ToolResult(False, "Missing 'key'", {})
                p = self._path_for(key)
                data = self._read_json(p)
                if data is None:
                    return ToolResult(False, "not found", {"key": key})
                body = json.dumps(data, ensure_ascii=False, indent=2)
                # respect output clamp
                body = body[: CFG.data["policy"]["max_output_chars"]]
                return ToolResult(True, body, {"key": key, "path": str(p)})

            if op == "set":
                if not key:
                    return ToolResult(False, "Missing 'key'", {})
                try:
                    parsed = json.loads(value)
                except Exception as e:
                    return ToolResult(False, f"Invalid JSON for 'value': {e}", {})
                p = self._path_for(key)
                self._write_json(p, parsed)
                return ToolResult(True, f"OK set {key}", {"key": key, "path": str(p)})

            if op == "merge":
                if not key:
                    return ToolResult(False, "Missing 'key'", {})
                try:
                    patch = json.loads(value)
                except Exception as e:
                    return ToolResult(False, f"Invalid JSON for 'value': {e}", {})
                if not isinstance(patch, dict):
                    return ToolResult(False, "merge requires object JSON", {})
                p = self._path_for(key)
                base = self._read_json(p) or {}
                if not isinstance(base, dict):
                    return ToolResult(
                        False, "existing value is not an object; cannot merge", {}
                    )
                base.update(patch)
                self._write_json(p, base)
                return ToolResult(True, f"OK merge {key}", {"key": key, "path": str(p)})

            if op == "incr":
                # incr a numeric field inside an object
                if not key or not field:
                    return ToolResult(False, "Missing 'key' or 'field'", {})
                p = self._path_for(key)
                base = self._read_json(p) or {}
                if not isinstance(base, dict):
                    return ToolResult(
                        False, "existing value is not an object; cannot incr", {}
                    )
                try:
                    delta = int(by)
                except Exception:
                    return ToolResult(False, "Invalid 'by' (int required)", {})
                cur = base.get(field, 0)
                if not isinstance(cur, int):
                    return ToolResult(False, "Target field is not an int", {})
                base[field] = cur + delta
                self._write_json(p, base)
                return ToolResult(
                    True,
                    f"{field} = {base[field]}",
                    {"key": key, "field": field, "value": base[field]},
                )

            if op == "delete":
                if not key:
                    return ToolResult(False, "Missing 'key'", {})
                p = self._path_for(key)
                if p.exists():
                    p.unlink()
                    return ToolResult(True, f"deleted {key}", {"key": key})
                return ToolResult(False, "not found", {"key": key})

            return ToolResult(
                False,
                f"Unsupported op: {op}",
                {
                    "supported": [
                        "get",
                        "set",
                        "merge",
                        "incr",
                        "delete",
                        "list",
                        "exists",
                    ]
                },
            )

        except Exception as e:
            return ToolResult(False, f"{type(e).__name__}: {e}", {"op": op, "key": key})
