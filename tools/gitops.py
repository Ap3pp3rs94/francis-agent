# tools/gitops.py
"""
GitOpsTool — safe Git wrapper for Francis
Supports: status, diff, add-commit, snapshot, tag, checkout
"""

import shlex
import subprocess
from pathlib import Path

try:
    from __main__ import Tool, ToolResult, CFG
except Exception:

    class Tool: ...

    class ToolResult: ...

    CFG = type("CFG", (), {"data": {}})()


class GitOpsTool(Tool):
    name = "gitops"
    description = "Safe git wrapper: status, diff, add-commit, snapshot, tag, checkout."

    def _run_git(self, args: str) -> ToolResult:
        cmd = f"git {args}"
        shcfg = CFG.data.get("shell", {})
        cwd = Path(shcfg.get("cwd", ".")).resolve()
        timeout = int(shcfg.get("timeout_sec", 25))
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
            ok = proc.returncode == 0
            return ToolResult(
                ok, out.strip(), {"returncode": proc.returncode, "cwd": str(cwd)}
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"Timed out after {timeout}s", {"cmd": cmd})
        except Exception as e:
            return ToolResult(False, str(e), {"cmd": cmd})

    def run(
        self,
        op: str = "status",
        message: str = "",
        files: str = ".",
        tag: str = "",
        branch: str = "",
        push: str = "false",
    ) -> ToolResult:
        op = (op or "status").lower()

        if op == "status":
            return self._run_git("status --porcelain -b")

        if op == "diff":
            return self._run_git("diff --stat")

        if op == "add-commit":
            msg = message or "chore: snapshot by Francis"
            add = self._run_git(f"add {shlex.quote(files)}")
            if not add.ok:
                return add
            return self._run_git(f"commit -m {shlex.quote(msg)}")

        if op == "snapshot":
            msg = message or "chore: snapshot by Francis"
            steps = []
            for step in [
                f"add {shlex.quote(files)}",
                f"commit -m {shlex.quote(msg)}",
            ]:
                r = self._run_git(step)
                steps.append((step, r.ok))
                if not r.ok:
                    return r
            if tag:
                r = self._run_git(f"tag {shlex.quote(tag)} -f")
                steps.append((f"tag {tag}", r.ok))
                if not r.ok:
                    return r
            if push.lower() in ("true", "1", "yes", "y"):
                if branch:
                    r = self._run_git(f"push origin {shlex.quote(branch)} --tags")
                else:
                    r = self._run_git("push --tags")
                steps.append(("push", r.ok))
                if not r.ok:
                    return r
            summary = "\n".join(f"[OK] {s}" if ok else f"[ERR] {s}" for s, ok in steps)
            return ToolResult(True, summary, {"op": "snapshot"})

        if op == "tag":
            if not tag:
                return ToolResult(False, "Missing 'tag' parameter", {})
            return self._run_git(f"tag {shlex.quote(tag)} -f")

        if op == "checkout":
            if not branch:
                return ToolResult(False, "Missing 'branch' parameter", {})
            return self._run_git(f"checkout {shlex.quote(branch)}")

        return ToolResult(
            False,
            f"Unsupported op: {op}",
            {
                "supported": [
                    "status",
                    "diff",
                    "add-commit",
                    "snapshot",
                    "tag",
                    "checkout",
                ]
            },
        )
