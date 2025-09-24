# tools/httpjson.py
"""
HTTPJSONTool — safe HTTP/JSON client for Francis
Usage via direct tool call or task graph:
  httpjson op=GET url="https://httpbin.org/get" query='{"hello":"world"}'
  httpjson op=POST url="https://httpbin.org/post" json='{"name":"francis"}'
  httpjson op=GET url="https://api.example.com/resource" headers='{"Authorization":"Bearer XXX"}'
  httpjson op=GET url="..." save_to="responses\\latest.json"

Supported args:
  - op: GET|POST|PUT|PATCH|DELETE (default GET)
  - url: target URL (required)
  - headers: JSON string of headers (optional)
  - query: JSON string of query params (optional)
  - json: JSON string for request body (optional)
  - data: JSON string for form-encoded body (optional, dict -> x-www-form-urlencoded)
  - timeout: seconds (default from config.web.timeout_sec)
  - max_bytes: clamp response read (fallback to config.web.max_bytes)
  - save_to: relative file path under files.root; saves response text
Notes:
  - Redacts Authorization and any header key matching /(?i)(api[_-]?key|token|password|secret)/
  - Pretty-prints JSON responses, truncates large bodies per policy.max_output_chars
"""

import json as _json
import re as _re
from pathlib import Path

try:
    import requests  # type: ignore
except Exception:
    requests = None

try:
    from __main__ import Tool, ToolResult, CFG
except Exception:

    class Tool: ...

    class ToolResult: ...

    CFG = type("CFG", (), {"data": {}})()


class HTTPJSONTool(Tool):
    name = "httpjson"
    description = "HTTP/JSON client: GET/POST/PUT/PATCH/DELETE with headers/query/body and safe redaction."

    _REDACT_RX = _re.compile(r"(?i)(authorization|api[_-]?key|token|password|secret)")

    def _redact_headers(self, headers: dict) -> dict:
        out = {}
        for k, v in headers.items():
            if self._REDACT_RX.search(k or ""):
                out[k] = "[REDACTED]"
            else:
                out[k] = v
        return out

    def _save_response(self, text: str, rel_path: str):
        root = Path(CFG.data["files"]["root"]).resolve()
        p = (root / rel_path).resolve()
        if not str(p).startswith(str(root)):
            raise PermissionError("save_to escapes files.root")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return str(p)

    def run(
        self,
        op: str = "GET",
        url: str = "",
        headers: str = "",
        query: str = "",
        json: str = "",
        data: str = "",
        timeout: str = "",
        max_bytes: str = "",
        save_to: str = "",
    ) -> ToolResult:
        if requests is None:
            return ToolResult(
                False,
                "The 'requests' package is not available in this environment.",
                {},
            )

        if not url:
            return ToolResult(False, "Missing 'url' parameter", {})

        # Parse JSON-ish inputs safely
        def parse_obj(s: str) -> dict:
            s = (s or "").strip()
            if not s:
                return {}
            try:
                return _json.loads(s)
            except Exception as e:
                raise ValueError(f"Invalid JSON: {e}")

        try:
            hdrs = parse_obj(headers)
            q = parse_obj(query)
            body_json = parse_obj(json)
            body_data = parse_obj(data)
        except ValueError as e:
            return ToolResult(False, str(e), {})

        # Config-derived defaults
        wcfg = CFG.data.get("web", {})
        t_sec = (
            int(timeout)
            if str(timeout).strip().isdigit()
            else int(wcfg.get("timeout_sec", 18))
        )
        clamp = (
            int(max_bytes)
            if str(max_bytes).strip().isdigit()
            else int(wcfg.get("max_bytes", 900_000))
        )
        ua = wcfg.get("user_agent", "FrancisBot/1.0 (+local)")

        # Compose request
        method = (op or "GET").upper().strip()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            return ToolResult(
                False,
                f"Unsupported op: {method}",
                {"supported": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
            )

        # Ensure UA present unless explicitly overridden
        hdrs = {"User-Agent": ua, **hdrs}

        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=hdrs,
                params=q or None,
                json=body_json or None,
                data=body_data or None,
                timeout=t_sec,
                stream=True,
            )
            status = resp.status_code
            # Read up to clamp bytes to avoid huge bodies
            raw = resp.raw.read(clamp, decode_content=True)
            text = raw.decode(resp.encoding or "utf-8", errors="replace")

            # Pretty JSON if possible
            pretty = None
            try:
                pretty = _json.dumps(_json.loads(text), indent=2)[
                    : CFG.data["policy"]["max_output_chars"]
                ]
            except Exception:
                # non-JSON response, fall back to raw text
                pretty = text[: CFG.data["policy"]["max_output_chars"]]

            meta = {
                "status": status,
                "url": url,
                "method": method,
                "headers": self._redact_headers(hdrs),
                "params": q,
                "truncated": len(pretty) < len(text),
            }

            if save_to:
                try:
                    saved = self._save_response(pretty, save_to)
                    meta["saved"] = saved
                except Exception as e:
                    return ToolResult(False, f"save_to failed: {e}", meta)

            ok = 200 <= status < 300
            # Include short tail of body for quick visibility if saved separately
            return ToolResult(ok, pretty, meta)

        except requests.Timeout:
            return ToolResult(
                False, f"Timeout after {t_sec}s", {"url": url, "method": method}
            )
        except Exception as e:
            return ToolResult(
                False, f"{type(e).__name__}: {e}", {"url": url, "method": method}
            )
