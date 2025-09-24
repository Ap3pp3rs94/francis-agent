#!/usr/bin/env python3
"""
Francis — Phase 3.6 (Ultra+ Task Graph, full)
---------------------------------------------
• Phase 3.5 core (config, memory, tools, plugins, planner, safety)
• Direct tool invocation: hello who=You note="Hi"
• Task Graph runner: `runplan <path>` with ${var} substitution
• JSONL audit, dry-run gating, risk scoring, profiling toggle
• Long-term memory in SQLite with FTS5, short-term JSON
• Plugin loader (tools/*.py)
"""

from __future__ import annotations

import datetime as dt
import gzip
import importlib.util
import io
import json
import os
import re
import shlex
import sqlite3
import subprocess
import textwrap
import time
import traceback
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional deps
try:
    import tomllib  # Python 3.11+
except Exception:
    tomllib = None
try:
    import requests  # type: ignore
except Exception:
    requests = None

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
EXPORT_DIR = APP_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True)
TOOLS_DIR = APP_DIR / "tools"
TOOLS_DIR.mkdir(exist_ok=True)
MEMORY_JSON = APP_DIR / "memory.json"
DB_PATH = APP_DIR / "francis.sqlite"
AUDIT_PATH = LOG_DIR / f"audit-{dt.datetime.now().strftime('%Y%m%d')}.jsonl"

DEFAULT_CFG = {
    "app": {
        "name": "Francis",
        "max_short_term": 800,
        "dry_run": False,
        "profile": False,
    },
    "memory": {"enable_long_term": True, "review_window": 15},
    "policy": {
        # raw strings to avoid \s warnings
        "redact_patterns": [
            r"(?i)(api[_-]?key|token|password)\s*[:=]\s*([A-Za-z0-9\-_.]+)",
            r"(?i)secret\s*[:=]\s*([^\s]+)",
        ],
        "max_output_chars": 16000,
    },
    "shell": {
        "timeout_sec": 25,
        "cwd": str(APP_DIR),  # override in config.toml
        "allow": [
            "echo",
            "dir",
            "ls",
            "type",
            "cat",
            "python",
            "pip",
            "git",
            "ipconfig",
            "whoami",
        ],
        "deny": [
            "rm",
            "del",
            "rmdir",
            "format",
            "shutdown",
            "mkfs",
            "reg ",
            "sc ",
            "bcdedit",
        ],
        "env_allow": ["PATH", "PYTHONPATH"],
    },
    "files": {"root": str(APP_DIR), "allow_write": True},
    "web": {
        "timeout_sec": 18,
        "user_agent": "FrancisBot/1.0 (+local)",
        "max_bytes": 900_000,
        "retries": 2,
        "backoff": 0.6,
    },
}


# ---------- Config ----------
class Config:
    def __init__(self, defaults: dict):
        self._cfg = json.loads(json.dumps(defaults))  # deep copy
        self._load_files()
        self._apply_env_overrides()
        self._validate()

    @property
    def data(self) -> dict:
        return self._cfg

    def _load_files(self):
        path = APP_DIR / "config.toml"
        if tomllib and path.exists():
            try:
                with path.open("rb") as f:
                    data = tomllib.load(f)
                for k, v in data.items():
                    if isinstance(v, dict) and k in self._cfg:
                        self._cfg[k].update(v)
                    else:
                        self._cfg[k] = v
            except Exception as e:
                log_console(f"[warn] Failed to load config.toml: {e}")

    def _apply_env_overrides(self):
        # Example: FRANCIS_app__dry_run=true
        for k, v in os.environ.items():
            if not k.startswith("FRANCIS_"):
                continue
            path = k[len("FRANCIS_") :].lower().split("__")
            try:
                ref = self._cfg
                for p in path[:-1]:
                    ref = ref[p]
                leaf = path[-1]
                try:
                    val = json.loads(v)
                except Exception:
                    val = v
                ref[leaf] = val
            except Exception:
                pass

    def _validate(self):
        assert isinstance(self._cfg["app"]["dry_run"], bool)
        assert 1 <= int(self._cfg["app"]["max_short_term"]) <= 5000


# ---------- Logging ----------
def log_console(msg: str):
    print(msg)


def log_audit(event: str, payload: dict):
    rec = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # rotate old audit logs (keep 10)
    files = sorted(LOG_DIR.glob("audit-*.jsonl"))
    while len(files) > 10:
        old = files.pop(0)
        try:
            old.unlink()
        except Exception:
            pass


# ---------- Memory ----------
@dataclass
class STEntry:
    id: str
    timestamp: str
    goal: str
    plan: List[str]
    outcome: str
    reflection: str


class ShortTermMemory:
    def __init__(self, path: Path, max_items: int):
        self.path = path
        self.max_items = max_items
        self.data: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:
                log_console("[warn] memory.json corrupted; starting fresh")
                self.data = []

    def _save(self):
        if len(self.data) > self.max_items:
            self.data = self.data[-self.max_items :]
        self.path.write_text(json.dumps(self.data, indent=2))

    def append(self, entry: STEntry):
        self.data.append(asdict(entry))
        self._save()

    def dump(self) -> List[Dict[str, Any]]:
        return list(self.data)


class LongTermMemory:
    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                kind TEXT NOT NULL,
                ref_id TEXT,
                extra JSON
            );
            """
        )
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
                id, content, tokenize='porter'
            );
            """
        )
        self.conn.commit()

    def upsert(
        self,
        *,
        id: str,
        timestamp: str,
        kind: str,
        ref_id: str,
        content: str,
        extra: dict | None = None,
    ):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO entries (id, timestamp, kind, ref_id, extra) VALUES (?,?,?,?,?)",
            (id, timestamp, kind, ref_id, json.dumps(extra or {})),
        )
        cur.execute(
            "INSERT OR REPLACE INTO entries_fts (id, content) VALUES (?,?)",
            (id, content),
        )
        self.conn.commit()

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        kinds: Optional[List[str]] = None,
        since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        q = query.replace('"', '""')
        sql = """
        SELECT e.id, e.timestamp, e.kind, e.ref_id,
               snippet(entries_fts, 1, '[', ']', ' … ', 8)
        FROM entries_fts f JOIN entries e ON e.id = f.id
        WHERE entries_fts MATCH ?
        """
        params = [q]
        if kinds:
            ph = ",".join(["?"] * len(kinds))
            sql += f" AND e.kind IN ({ph})"
            params += kinds
        if since:
            sql += " AND e.timestamp >= ?"
            params.append(since)
        sql += " ORDER BY e.timestamp DESC LIMIT ?"
        params.append(limit)
        cur.execute(sql, params)
        rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "kind": r[2],
                "ref_id": r[3],
                "snippet": r[4],
            }
            for r in rows
        ]

    def backup(self, dest: Path):
        self.conn.commit()
        with sqlite3.connect(f"file:{dest}?mode=rwc", uri=True) as target:
            self.conn.backup(target)


# ---------- Tools ----------
@dataclass
class ToolResult:
    ok: bool
    output: str
    meta: Dict[str, Any]


class Tool:
    name: str = "tool"
    description: str = ""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self, **kwargs) -> ToolResult:
        return ToolResult(False, "unimplemented", {})

    def _truncate(self, s: str, max_chars: int) -> Tuple[str, bool]:
        return (
            (s, False)
            if len(s) <= max_chars
            else (s[:max_chars] + "\n… [truncated]", True)
        )

    def _redact(self, s: str, patterns: List[str]) -> str:
        for pat in patterns:
            try:
                s = re.sub(pat, "[REDACTED]", s)
            except re.error:
                pass
        return s


class ShellTool(Tool):
    name = "shell"
    description = "Execute safe shell commands with allow/deny lists and timeouts."

    def run(self, cmd: str) -> ToolResult:
        policy = CFG.data["policy"]
        shcfg = CFG.data["shell"]
        allow, deny = set(shcfg["allow"]), set(shcfg["deny"])
        timeout = int(shcfg["timeout_sec"])
        cwd = Path(shcfg["cwd"]).resolve()
        base = Path(shlex.split(cmd, posix=False)[0]).name.lower() if cmd else ""
        if base in deny or (allow and base not in allow):
            return ToolResult(
                False, f"Command '{base}' not allowed by policy.", {"cmd": cmd}
            )
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            out = self._redact(out, policy["redact_patterns"])
            out, truncated = self._truncate(out, policy["max_output_chars"])
            return ToolResult(
                proc.returncode == 0,
                out.strip(),
                {
                    "returncode": proc.returncode,
                    "cwd": str(cwd),
                    "truncated": truncated,
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"Timed out after {timeout}s", {"cmd": cmd})
        except Exception as e:
            return ToolResult(False, str(e), {"cmd": cmd})


class FileTool(Tool):
    name = "files"
    description = "Read/write text files within configured root."

    def _resolve(self, path: str) -> Path:
        root = Path(CFG.data["files"]["root"]).resolve()
        p = (root / path).resolve()
        if not str(p).startswith(str(root)):
            raise PermissionError("Path escapes root")
        return p

    def run(self, op: str, path: str, content: Optional[str] = None) -> ToolResult:
        try:
            p = self._resolve(path)
            if op == "read":
                data = p.read_text(encoding="utf-8")
                data = self._redact(data, CFG.data["policy"]["redact_patterns"])
                data, truncated = self._truncate(
                    data, CFG.data["policy"]["max_output_chars"]
                )
                return ToolResult(True, data, {"path": str(p), "truncated": truncated})
            if op == "write":
                if not CFG.data["files"]["allow_write"]:
                    return ToolResult(
                        False, "Writes disabled by policy.", {"path": str(p)}
                    )
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content or "", encoding="utf-8")
                return ToolResult(
                    True, f"Wrote {p}", {"bytes": len((content or "").encode("utf-8"))}
                )
            return ToolResult(False, "Unsupported op (read|write)", {"op": op})
        except Exception as e:
            return ToolResult(False, str(e), {"op": op, "path": path})


class WebTool(Tool):
    name = "web"
    description = "Fetch a URL and return text content with retry/backoff."

    def run(self, url: str) -> ToolResult:
        cfg = CFG.data["web"]
        timeout = int(cfg["timeout_sec"])
        ua = cfg["user_agent"]
        max_bytes = int(cfg["max_bytes"])
        retries = int(cfg.get("retries", 1))
        backoff = float(cfg.get("backoff", 0.5))
        err: Optional[str] = None
        for attempt in range(retries + 1):
            try:
                if requests:
                    resp = requests.get(
                        url, timeout=timeout, headers={"User-Agent": ua}, stream=True
                    )
                    resp.raise_for_status()
                    data = resp.raw.read(max_bytes, decode_content=True)
                    text = data.decode(resp.encoding or "utf-8", errors="ignore")
                else:
                    from urllib.request import Request, urlopen

                    req = Request(url, headers={"User-Agent": ua})
                    with urlopen(req, timeout=timeout) as r:  # type: ignore
                        text = r.read(max_bytes).decode("utf-8", errors="ignore")
                text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.I)
                text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                text, truncated = self._truncate(
                    text, CFG.data["policy"]["max_output_chars"]
                )
                return ToolResult(True, text, {"truncated": truncated})
            except Exception as e:
                err = str(e)
                if attempt < retries:
                    time.sleep(backoff * (2**attempt))
        return ToolResult(False, err or "unknown error", {"url": url})


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    def load_plugins(self):
        for path in TOOLS_DIR.glob("*.py"):
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if not spec or not spec.loader:
                continue
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)  # type: ignore
                for obj in vars(mod).values():
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, Tool)
                        and obj is not Tool
                    ):
                        self.register(obj(CFG.data))
            except Exception as e:
                log_console(f"[warn] Failed to load plugin {path.name}: {e}")


# ---------- Francis Core ----------
def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def hrule(ch: str = "-") -> str:
    return ch * 70


class Francis:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.app_name = cfg.data["app"]["name"]
        self.st = ShortTermMemory(MEMORY_JSON, cfg.data["app"]["max_short_term"])
        self.lt = (
            LongTermMemory(DB_PATH) if cfg.data["memory"]["enable_long_term"] else None
        )
        self.tools = ToolRegistry()
        self.tools.register(ShellTool(cfg.data))
        self.tools.register(FileTool(cfg.data))
        self.tools.register(WebTool(cfg.data))
        self.tools.load_plugins()
        log_console(
            f"[{now_iso()}] {self.app_name} ready. Tools: {', '.join(self.tools.names())}"
        )
        log_audit("startup", {"tools": self.tools.names()})

    # ---- Planner helpers ----
    def _parse_tool_invocation(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Accepts direct tool calls like:
          hello who=Ap3pp note="Plugins are live"
          search pattern="Francis" glob="**/*.py" max_matches=10
          gitops op=status
        """
        toks = shlex.split(text)
        if not toks:
            return None
        t0 = toks[0].lower()
        if t0 not in self.tools.names():
            return None
        args: Dict[str, Any] = {}
        for tok in toks[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                args[k] = v
        return {"tool": t0, "args": args, "explain": f"{t0} with args"}

    # ---- Planner v2.1 ----
    def plan(self, goal: str) -> List[Dict[str, Any]]:
        g = goal.strip()
        actions: List[Dict[str, Any]] = []
        parts = [p.strip() for p in re.split(r"\s*&&\s*", g) if p.strip()]
        for p in parts:
            parsed = self._parse_tool_invocation(p)
            if parsed:
                actions.append(parsed)
                continue
            pl = p.lower()
            if pl.startswith("open ") and (
                pl.endswith(".txt") or pl.endswith(".md") or pl.endswith(".py")
            ):
                path = p.split(" ", 1)[1]
                actions.append(
                    {
                        "tool": "files",
                        "args": {"op": "read", "path": path},
                        "explain": f"Read file: {path}",
                    }
                )
            elif pl.startswith("write ") and ":" in p:
                _, rest = p.split(" ", 1)
                path, content = rest.split(":", 1)
                actions.append(
                    {
                        "tool": "files",
                        "args": {
                            "op": "write",
                            "path": path.strip(),
                            "content": content,
                        },
                        "explain": f"Write file: {path.strip()}",
                    }
                )
            elif pl.startswith("run "):
                cmd = p.split(" ", 1)[1]
                actions.append(
                    {"tool": "shell", "args": {"cmd": cmd}, "explain": f"Shell: {cmd}"}
                )
            elif pl.startswith("fetch ") and p.split(" ", 1)[1].startswith("http"):
                url = p.split(" ", 1)[1].strip()
                actions.append(
                    {
                        "tool": "web",
                        "args": {"url": url},
                        "explain": f"Fetch URL: {url}",
                    }
                )
            else:
                if any(k in pl for k in ("git ", "python ", "pip ")):
                    actions.append(
                        {
                            "tool": "shell",
                            "args": {"cmd": p},
                            "explain": f"Shell (keyword): {p}",
                        }
                    )
                else:
                    actions.append(
                        {
                            "tool": "shell",
                            "args": {"cmd": f"echo {shlex.quote(p)}"},
                            "explain": "Echo for traceability",
                        }
                    )
        return actions

    # ---- Risk & confirm ----
    def risk_score(self, action: Dict[str, Any]) -> Tuple[int, str]:
        t = action.get("tool")
        a = action.get("args", {})
        score, reason = 1, "low"
        if t == "shell":
            cmd = a.get("cmd", "").lower()
            if any(
                d in cmd
                for d in [
                    "rm",
                    "del",
                    "rmdir",
                    "format",
                    "shutdown",
                    "mkfs",
                    "reg ",
                    "sc ",
                    "bcdedit",
                ]
            ):
                return 9, "dangerous shell verb"
            if "git " in cmd:
                score, reason = 5, "git side-effects"
        if t == "files" and a.get("op") == "write":
            score, reason = 4, "file write"
        if t == "web":
            score, reason = max(score, 2), "network fetch"
        return score, reason

    def confirm(self, action: Dict[str, Any], score: int, reason: str) -> bool:
        if self.cfg.data["app"]["dry_run"]:
            log_console(
                f"[dry-run] {action['tool']} {action['args']} (risk={score}:{reason})"
            )
            return False
        if score >= 5:
            print(hrule())
            print(f"High-risk action (score={score}, reason={reason}) → confirm:")
            print(json.dumps(action, indent=2))
            ans = input("Proceed? [y/N]: ").strip().lower()
            return ans == "y"
        return True

    # ---- Reflection ----
    def reflect(
        self, goal: str, actions: List[Dict[str, Any]], results: List[ToolResult]
    ) -> str:
        ok = all(r.ok for r in results) if results else True
        tips = []
        if not ok:
            tips.append("Some actions failed; consider allowlists/timeouts.")
        if any(a["tool"] == "files" for a in actions):
            tips.append("Files are sandboxed to 'files.root'.")
        if any(a["tool"] == "shell" for a in actions):
            tips.append("Shell obeys allow/deny lists.")
        return f"Run {len(actions)} actions | All OK: {ok} | " + (
            "; ".join(tips) or "Indexed outputs to long-term memory."
        )

    # ---- Memory writes ----
    def _write_short(
        self, goal: str, plan_lines: List[str], outcome: str, reflection: str
    ) -> STEntry:
        eid = uuid.uuid4().hex[:12]
        entry = STEntry(eid, now_iso(), goal, plan_lines, outcome, reflection)
        self.st.append(entry)
        return entry

    def _write_long(self, entry: STEntry, tool_logs: List[Dict[str, Any]]):
        if not self.lt:
            return
        self.lt.upsert(
            id=f"{entry.id}:goal",
            timestamp=entry.timestamp,
            kind="goal",
            ref_id=entry.id,
            content=entry.goal,
            extra={},
        )
        self.lt.upsert(
            id=f"{entry.id}:plan",
            timestamp=entry.timestamp,
            kind="plan",
            ref_id=entry.id,
            content="\n".join(entry.plan),
            extra={},
        )
        self.lt.upsert(
            id=f"{entry.id}:outcome",
            timestamp=entry.timestamp,
            kind="outcome",
            ref_id=entry.id,
            content=entry.outcome,
            extra={},
        )
        self.lt.upsert(
            id=f"{entry.id}:reflection",
            timestamp=entry.timestamp,
            kind="reflection",
            ref_id=entry.id,
            content=entry.reflection,
            extra={},
        )
        for i, log in enumerate(tool_logs, 1):
            body = f"[{log['tool']}] ok={log['ok']} meta={json.dumps(log.get('meta', {}))}\n{log['output']}"
            if len(body) > 5000:
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode="w") as gz:
                    gz.write(body.encode("utf-8"))
                body = f"<gzipped:{len(buf.getvalue())}B>"
            self.lt.upsert(
                id=f"{entry.id}:tool:{i}",
                timestamp=entry.timestamp,
                kind="tool",
                ref_id=entry.id,
                content=body,
                extra={},
            )

    # ---- Task Graph runner ----
    _VAR_RX = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def _subst_vars(self, obj: Any, vars: Dict[str, Any]) -> Any:
        if isinstance(obj, str):

            def repl(m):
                return str(vars.get(m.group(1), m.group(0)))

            return self._VAR_RX.sub(repl, obj)
        if isinstance(obj, list):
            return [self._subst_vars(x, vars) for x in obj]
        if isinstance(obj, dict):
            return {k: self._subst_vars(v, vars) for k, v in obj.items()}
        return obj

    def execute_task_graph(self, graph: Dict[str, Any]):
        vars = graph.get("vars", {}) or {}
        steps = graph.get("steps", []) or []
        if not isinstance(steps, list) or not steps:
            print("TaskGraph: no steps provided.")
            return

        plan_lines = [
            f"{i+1}. {step.get('use')} {json.dumps(step.get('args', {}))}"
            for i, step in enumerate(steps)
        ]
        print("Plan (TaskGraph):")
        for ln in plan_lines:
            print("  " + ln)

        tool_logs: List[Dict[str, Any]] = []
        results: List[ToolResult] = []

        for idx, step in enumerate(steps, 1):
            tool_name = str(step.get("use", "")).lower()
            raw_args = step.get("args", {}) or {}
            if not tool_name:
                results.append(ToolResult(False, "Missing 'use' in step", {}))
                continue
            args = self._subst_vars(raw_args, vars)
            action = {
                "tool": tool_name,
                "args": args,
                "explain": f"TaskGraph step {idx}",
            }
            score, reason = self.risk_score(action)
            if not self.confirm(action, score, reason):
                res = ToolResult(False, f"Skipped (risk={score})", {"action": action})
            else:
                try:
                    tool = self.tools.get(tool_name)
                    res = tool.run(**args)
                except Exception as e:
                    res = ToolResult(
                        False,
                        f"{type(e).__name__}: {e}",
                        {"trace": traceback.format_exc()[-500:]},
                    )
            results.append(res)
            tool_logs.append(
                {
                    "tool": tool_name,
                    "ok": res.ok,
                    "output": res.output,
                    "meta": res.meta,
                }
            )
            preview = res.output[:600] + (
                "\n… [truncated]" if len(res.output) > 600 else ""
            )
            print(hrule())
            print(f"[{tool_name}] ok={res.ok} meta={res.meta}")
            print(preview)

        reflection = self.reflect(
            "TaskGraph",
            [{"tool": s.get("use", ""), "args": s.get("args", {})} for s in steps],
            results,
        )
        outcome = (
            "; ".join(
                [f"{i+1}:{'OK' if r.ok else 'ERR'}" for i, r in enumerate(results)]
            )
            or "no actions"
        )
        entry = self._write_short("TaskGraph executed", plan_lines, outcome, reflection)
        self._write_long(entry, tool_logs)
        print(hrule())
        print("Reflection:")
        print("  " + reflection)

    # ---- Commands ----
    def cmd_help(self):
        print(hrule())
        print("Commands:")
        print("  help                           Show this help")
        print("  mem [N]                        Show recent short-term entries")
        print(
            "  review <fts query> [kinds]     Search long-term (kinds: goal,plan,outcome,reflection,tool)"
        )
        print("  export                         Export JSON and SQLite backups")
        print(
            "  prune <days>                   Prune short-term entries older than N days"
        )
        print("  tools                          List registered tools")
        print("  set dry_run on|off             Toggle dry-run mode")
        print("  profile on|off                 Toggle profiling of last goal")
        print("  status                         Summarize current config and files")
        print("  health                         Quick health checks")
        print("  selftest                       Run a tiny end-to-end test")
        print(
            "  sampleplan                     Write _sample_plan.json in project root"
        )
        print("  runplan <path>                 Execute a JSON task graph file")
        print("  exit                           Quit")
        print(hrule())

    def cmd_mem(self, n: Optional[int] = None):
        n = n or int(self.cfg.data["memory"]["review_window"])
        items = self.st.dump()[-n:]
        if not items:
            print("(no memory yet)")
            return
        print(hrule())
        for it in items:
            print(f"[{it['timestamp']}] id={it['id']}")
            print(f"  Goal: {it['goal']}")
            print(f"  Plan: {len(it['plan'])} steps")
            print(f"  Outcome: {it['outcome']}")
            print(f"  Reflection: {it['reflection']}")
            print("  —")
        print(hrule())

    def cmd_review(self, raw: str):
        if not self.lt:
            print("Long-term memory disabled.")
            return
        parts = raw.split()
        query = parts[1] if len(parts) >= 2 else ""
        kinds = parts[2].split(",") if len(parts) >= 3 else None
        if not query:
            print("Usage: review <fts query> [kinds]")
            return
        rows = self.lt.search(query, kinds=kinds)
        if not rows:
            print("(no results)")
            return
        print(hrule("="))
        print(f"Search: {query} kinds={kinds or 'all'}")
        for r in rows:
            print(f"[{r['timestamp']}] {r['kind']} @ {r['ref_id']}")
            print(f"  {r['snippet']}")
            print("  —")
        print(hrule("="))

    def cmd_export(self):
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        st_path = EXPORT_DIR / f"memory_{ts}.json"
        st_path.write_text(json.dumps(self.st.dump(), indent=2))
        if self.lt:
            db_path = EXPORT_DIR / f"francis_{ts}.sqlite"
            self.lt.backup(db_path)
            print(f"Exported: {st_path.name}, {db_path.name}")
        else:
            print(f"Exported: {st_path.name}")

    def cmd_prune(self, days: int):
        cutoff = dt.datetime.now() - dt.timedelta(days=days)
        before = len(self.st.data)

        def tsf(e):
            try:
                return dt.datetime.fromisoformat(e.get("timestamp", ""))
            except Exception:
                return dt.datetime.min

        self.st.data = [e for e in self.st.data if tsf(e) >= cutoff]
        self.st._save()
        print(f"Pruned {before - len(self.st.data)} entries.")

    def cmd_status(self):
        print(hrule())
        print("Status")
        print(f"  App: {self.app_name}")
        print(
            f"  Dry run: {self.cfg.data['app']['dry_run']}  Profile: {self.cfg.data['app']['profile']}"
        )
        print(f"  Tools: {', '.join(self.tools.names())}")
        print(f"  Memory JSON: {'exists' if MEMORY_JSON.exists() else 'missing'}")
        print(f"  SQLite DB: {'exists' if DB_PATH.exists() else 'missing'}")
        print(f"  Audit log: {AUDIT_PATH.name}")
        print(hrule())

    def cmd_health(self):
        issues = []
        if not os.access(APP_DIR, os.W_OK):
            issues.append("App dir not writable")
        if not os.access(EXPORT_DIR, os.W_OK):
            issues.append("Export dir not writable")
        if issues:
            print("Health: FAIL\n - " + "\n - ".join(issues))
        else:
            print("Health: OK")

    def cmd_selftest(self):
        ok = True
        r1 = FileTool(self.cfg.data).run("write", "_selftest.txt", "hello")
        ok &= r1.ok
        r2 = FileTool(self.cfg.data).run("read", "_selftest.txt")
        ok &= r2.ok and "hello" in r2.output
        r3 = ShellTool(self.cfg.data).run("echo selftest")
        ok &= r3.ok
        print("Selftest:", "PASS" if ok else "FAIL")

    def cmd_sampleplan(self):
        plan = {
            "vars": {"msg": "world"},
            "steps": [
                {"use": "shell", "args": {"cmd": "echo Hello ${msg}!"}},
                {
                    "use": "files",
                    "args": {
                        "op": "write",
                        "path": "notes\\hello.txt",
                        "content": "Hi ${msg}",
                    },
                },
                {"use": "web", "args": {"url": "https://example.com"}},
            ],
        }
        dest = APP_DIR / "_sample_plan.json"
        dest.write_text(json.dumps(plan, indent=2))
        print(f"Wrote sample plan: {dest}")

    # ---- Execute goal ----
    def execute_goal(self, goal: str):
        start = time.perf_counter()
        actions = self.plan(goal)
        plan_lines = [
            f"{i+1}. {a['explain']} → {a['tool']}" for i, a in enumerate(actions)
        ]
        print("Plan:")
        for ln in plan_lines:
            print("  " + ln)
        results: List[ToolResult] = []
        tool_logs: List[Dict[str, Any]] = []
        for a in actions:
            score, reason = self.risk_score(a)
            if not self.confirm(a, score, reason):
                res = ToolResult(False, f"Skipped (risk={score})", {"action": a})
            else:
                try:
                    tool = self.tools.get(a["tool"])  # may raise
                    res = tool.run(**a.get("args", {}))
                except Exception as e:
                    res = ToolResult(
                        False,
                        f"{type(e).__name__}: {e}",
                        {"trace": traceback.format_exc()[-500:]},
                    )
            results.append(res)
            tool_logs.append(
                {
                    "tool": a["tool"],
                    "ok": res.ok,
                    "output": res.output,
                    "meta": res.meta,
                }
            )
            preview = res.output[:600] + (
                "\n… [truncated]" if len(res.output) > 600 else ""
            )
            print(hrule())
            print(f"[{a['tool']}] ok={res.ok} meta={res.meta}")
            print(preview)
            log_audit("tool_call", {"tool": a["tool"], "ok": res.ok, "meta": res.meta})
        reflection = self.reflect(goal, actions, results)
        outcome = (
            "; ".join(
                [f"{i+1}:{'OK' if r.ok else 'ERR'}" for i, r in enumerate(results)]
            )
            or "no actions"
        )
        entry = self._write_short(goal, plan_lines, outcome, reflection)
        self._write_long(entry, tool_logs)
        dur = time.perf_counter() - start
        print(hrule())
        print("Reflection:")
        print("  " + reflection)
        print(f"Duration: {dur:.2f}s")
        if self.cfg.data["app"].get("profile"):
            log_console("(profiling) — lightweight timing enabled for last run.")

    # ---- REPL ----
    def run(self):
        try:
            while True:
                raw = input(
                    "\nEnter a goal (or 'help','mem','review','export','prune','tools','set dry_run on|off','profile on|off','status','health','selftest','sampleplan','runplan <path>','exit'): "
                ).strip()
                if not raw:
                    continue
                low = raw.lower()
                if low == "exit":
                    print("Goodbye!")
                    break
                if low == "help":
                    self.cmd_help()
                    continue
                if low.startswith("mem"):
                    parts = raw.split()
                    n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                    self.cmd_mem(n)
                    continue
                if low.startswith("review"):
                    self.cmd_review(raw)
                    continue
                if low == "export":
                    self.cmd_export()
                    continue
                if low.startswith("prune"):
                    parts = raw.split()
                    days = (
                        int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 30
                    )
                    self.cmd_prune(days)
                    continue
                if low == "tools":
                    print(", ".join(self.tools.names()))
                    continue
                if low.startswith("set dry_run"):
                    val = raw.rsplit(" ", 1)[-1].lower() in ("on", "true", "1", "yes")
                    self.cfg.data["app"]["dry_run"] = val
                    print(f"dry_run = {val}")
                    continue
                if low.startswith("profile"):
                    val = raw.rsplit(" ", 1)[-1].lower() in ("on", "true", "1", "yes")
                    self.cfg.data["app"]["profile"] = val
                    print(f"profile = {val}")
                    continue
                if low == "status":
                    self.cmd_status()
                    continue
                if low == "health":
                    self.cmd_health()
                    continue
                if low == "selftest":
                    self.cmd_selftest()
                    continue
                if low == "sampleplan":
                    self.cmd_sampleplan()
                    continue
                if low.startswith("runplan"):
                    _, _, path = raw.partition(" ")
                    p = (
                        (APP_DIR / path)
                        if path and not Path(path).is_absolute()
                        else Path(path)
                    )
                    if not path or not p.exists():
                        print("Usage: runplan <path-to-json>  (file must exist)")
                        continue
                    try:
                        graph = json.loads(p.read_text(encoding="utf-8"))
                        self.execute_task_graph(graph)
                    except Exception as e:
                        print(f"Failed to run plan: {e}")
                    continue
                # Otherwise it's a goal
                self.execute_goal(raw)
        except KeyboardInterrupt:
            print("\n(interrupted) Bye!")
        except Exception as e:
            print(f"[error] {e}")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    CFG = Config(DEFAULT_CFG)
    Francis(CFG).run()


def main() -> None:
    # TODO: call your actual loop here
    print("francis CLI is alive")


if __name__ == "__main__":
    main()
