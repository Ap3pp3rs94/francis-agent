from __future__ import annotations
from __future__ import annotations
from __future__ import annotations

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

# TODO: add the rest of your Francis implementation here
# (imports, classes, functions, etc.)


def main(argv=None) -> int:
    import sys
    from importlib import metadata

    args = sys.argv[1:] if argv is None else list(argv)
    if any(a in ("-V", "--version") for a in args):
        try:
            ver = metadata.version("francis-agent")
            print(f"francis {ver}")
        except Exception:
            print("francis (local)")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
