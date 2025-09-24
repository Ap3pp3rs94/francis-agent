# tools/zipper.py
"""
ZipperTool — safe zip/unzip within files.root

Ops:
  - zip   src="notes" dest="backups\\notes.zip" include="*.txt;*.md" exclude="*.tmp;__pycache__" level="6"
  - unzip src="backups\\notes.zip" dest="restore\\notes" overwrite="false"
  - list  src="backups\\notes.zip"

Notes:
  • All paths are relative to files.root and sandboxed.
  • Globs (include/exclude) are ;-separated lists, matched case-insensitively.
  • Compression level: 0..9 (default 6). Uses ZIP_DEFLATED when level>0.
  • Unzip refuses to overwrite unless overwrite="true".
"""

import fnmatch
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import zipfile

try:
    from __main__ import Tool, ToolResult, CFG
except Exception:

    class Tool: ...

    class ToolResult: ...

    CFG = type("CFG", (), {"data": {}})()


class ZipperTool(Tool):
    name = "zipper"
    description = "Create and extract zip archives under files.root."

    def _root(self) -> Path:
        return Path(CFG.data["files"]["root"]).resolve()

    def _resolve_in_root(self, rel: str) -> Path:
        root = self._root()
        p = (root / rel).resolve()
        if not str(p).startswith(str(root)):
            raise PermissionError("Path escapes files.root")
        return p

    def _split_patterns(self, s: str) -> List[str]:
        s = (s or "").strip()
        if not s:
            return []
        return [p.strip() for p in s.split(";") if p.strip()]

    def _match_any(self, name: str, patterns: Iterable[str]) -> bool:
        low = name.lower()
        for pat in patterns:
            if fnmatch.fnmatch(low, pat.lower()):
                return True
        return False

    def _walk_collect(
        self, base: Path, include: List[str], exclude: List[str]
    ) -> List[Tuple[Path, Path]]:
        files: List[Tuple[Path, Path]] = []
        base = base.resolve()
        for root, dirs, filenames in os.walk(base):
            # apply directory excludes (by name)
            dirs[:] = [d for d in dirs if not self._match_any(d, exclude)]
            for fn in filenames:
                if include and not self._match_any(fn, include):
                    continue
                if exclude and self._match_any(fn, exclude):
                    continue
                ap = Path(root, fn).resolve()
                rp = ap.relative_to(base)
                files.append((ap, rp))
        return files

    def _zip_dir(
        self,
        src_dir: Path,
        dest_zip: Path,
        include: List[str],
        exclude: List[str],
        level: int,
    ) -> Tuple[bool, str, dict]:
        src_dir = src_dir.resolve()
        count = 0
        mode = zipfile.ZIP_STORED if level <= 0 else zipfile.ZIP_DEFLATED
        dest_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            dest_zip, "w", compression=mode, compresslevel=None if level <= 0 else level
        ) as zf:
            for ap, rp in self._walk_collect(src_dir, include, exclude):
                # normalize to forward slashes inside zip
                zf.write(ap, arcname=str(rp).replace("\\", "/"))
                count += 1
        return (
            True,
            f"Zipped {count} file(s) → {dest_zip}",
            {"files": count, "dest": str(dest_zip)},
        )

    def _zip_file(
        self, src_file: Path, dest_zip: Path, level: int
    ) -> Tuple[bool, str, dict]:
        mode = zipfile.ZIP_STORED if level <= 0 else zipfile.ZIP_DEFLATED
        dest_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            dest_zip, "w", compression=mode, compresslevel=None if level <= 0 else level
        ) as zf:
            zf.write(src_file, arcname=src_file.name)
        return True, f"Zipped 1 file → {dest_zip}", {"files": 1, "dest": str(dest_zip)}

    def _unzip(
        self, src_zip: Path, dest_dir: Path, overwrite: bool
    ) -> Tuple[bool, str, dict]:
        if not src_zip.exists():
            return False, "source zip not found", {}
        with zipfile.ZipFile(src_zip, "r") as zf:
            names = zf.namelist()
            extracted = 0
            for name in names:
                # Guard path traversal attempts
                target = (dest_dir / name).resolve()
                if not str(target).startswith(str(dest_dir.resolve())):
                    return False, f"blocked path traversal: {name}", {}
                if target.exists() and not overwrite:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                extracted += 1
        return (
            True,
            f"Extracted {extracted} file(s) → {dest_dir}",
            {"files": extracted, "dest": str(dest_dir)},
        )

    def _list(self, src_zip: Path) -> Tuple[bool, str, dict]:
        if not src_zip.exists():
            return False, "source zip not found", {}
        with zipfile.ZipFile(src_zip, "r") as zf:
            items = zf.infolist()
            lines = [f"{it.filename}\t{it.file_size}B" for it in items]
        return True, "\n".join(lines), {"count": len(lines)}

    def run(
        self,
        op: str,
        src: str = "",
        dest: str = "",
        include: str = "",
        exclude: str = "",
        level: str = "6",
        overwrite: str = "false",
    ):
        try:
            op = (op or "").lower().strip()
            if op not in {"zip", "unzip", "list"}:
                return ToolResult(
                    False,
                    f"Unsupported op: {op}",
                    {"supported": ["zip", "unzip", "list"]},
                )

            if op == "zip":
                if not src or not dest:
                    return ToolResult(False, "zip requires src and dest", {})
                src_path = self._resolve_in_root(src)
                dest_zip = self._resolve_in_root(dest)
                try:
                    lvl = max(0, min(9, int(level)))
                except Exception:
                    lvl = 6
                if src_path.is_dir():
                    inc = self._split_patterns(include)
                    exc = self._split_patterns(exclude)
                    ok, msg, meta = self._zip_dir(src_path, dest_zip, inc, exc, lvl)
                    return ToolResult(ok, msg, meta)
                elif src_path.is_file():
                    ok, msg, meta = self._zip_file(src_path, dest_zip, lvl)
                    return ToolResult(ok, msg, meta)
                else:
                    return ToolResult(False, "src not found", {"src": str(src_path)})

            if op == "unzip":
                if not src:
                    return ToolResult(False, "unzip requires src", {})
                src_zip = self._resolve_in_root(src)
                dest_dir = self._resolve_in_root(
                    dest or (Path("restore") / Path(src).stem).as_posix()
                )
                ow = str(overwrite).lower() in ("1", "true", "yes", "y")
                ok, msg, meta = self._unzip(src_zip, dest_dir, ow)
                return ToolResult(ok, msg, meta)

            if op == "list":
                if not src:
                    return ToolResult(False, "list requires src", {})
                src_zip = self._resolve_in_root(src)
                ok, msg, meta = self._list(src_zip)
                return ToolResult(ok, msg, meta)

            return ToolResult(False, f"Unsupported op: {op}", {})

        except Exception as e:
            return ToolResult(
                False, f"{type(e).__name__}: {e}", {"op": op, "src": src, "dest": dest}
            )
