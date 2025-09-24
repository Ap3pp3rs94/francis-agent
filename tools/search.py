import re
from pathlib import Path

try:
    from __main__ import Tool, ToolResult, CFG
except Exception:

    class Tool: ...

    class ToolResult: ...

    CFG = type(
        "CFG",
        (),
        {"data": {"files": {"root": "."}, "policy": {"max_output_chars": 16000}}},
    )()


class CodeSearchTool(Tool):
    name = "search"
    description = "Regex search across files under the configured files.root."

    def run(
        self,
        pattern: str,
        glob: str = "**/*.*",
        max_matches: int = 200,
        ignore_binary: bool = True,
    ):
        root = Path(CFG.data["files"]["root"]).resolve()
        rx = re.compile(pattern)
        results = []
        count = 0
        for p in root.glob(glob):
            if not p.is_file():
                continue
            try:
                if ignore_binary and any(
                    p.suffix.lower() in s
                    for s in [
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".gif",
                        ".pdf",
                        ".zip",
                        ".exe",
                        ".dll",
                    ]
                ):
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for ln, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    results.append(f"{p.relative_to(root)}:{ln}: {line.strip()}")
                    count += 1
                    if count >= max_matches:
                        break
            if count >= max_matches:
                break
        if not results:
            return ToolResult(True, "(no matches)", {"matches": 0})
        out = "\n".join(results)
        max_chars = CFG.data["policy"]["max_output_chars"]
        out = out[:max_chars] + ("\n… [truncated]" if len(out) > max_chars else "")
        return ToolResult(
            True, out, {"matches": count, "glob": glob, "pattern": pattern}
        )
