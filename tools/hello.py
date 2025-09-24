try:
    from __main__ import Tool, ToolResult
except Exception:

    class Tool: ...

    class ToolResult: ...


class HelloTool(Tool):
    name = "hello"
    description = "Greet a user and echo back a short message."

    def run(self, who: str = "world", note: str = ""):
        msg = f"Hello, {who}!"
        if note:
            msg += f" Note: {note}"
        return ToolResult(True, msg, {"who": who, "len": len(msg)})
