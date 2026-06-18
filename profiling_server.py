"""Tiny local server for profiling_form.html — no new dependencies (stdlib only).

The form itself is a static file and can't write to disk on its own (browser
security). This server does two things:
  - GET  /            serves profiling_form.html
  - POST /save        appends the submitted profile block to app/users.md

Usage:
    python profiling_server.py
    -> open http://localhost:8765/   (NOT the .html file directly — file://
       pages can't fetch() to a server, so saving would silently fail)
"""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORM_PATH = ROOT / "profiling_form.html"
USERS_PATH = ROOT / "app" / "users.md"
PORT = 8765


def _existing_names(users_md_text: str) -> set[str]:
    return {m.strip().lower() for m in re.findall(r"^##\s+(.+)$", users_md_text, flags=re.MULTILINE)}


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/profiling_form.html"):
            html = FORM_PATH.read_text(encoding="utf-8").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/save":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Bad JSON"})
            return

        name = (payload.get("name") or "").strip()
        block = (payload.get("block") or "").strip()
        if not name or not block:
            self._send_json(400, {"ok": False, "error": "Missing name or block"})
            return

        current = USERS_PATH.read_text(encoding="utf-8")
        if name.lower() in _existing_names(current):
            self._send_json(409, {"ok": False, "error": f'Profile "{name}" already exists in users.md — rename and try again.'})
            return

        USERS_PATH.write_text(current.rstrip() + "\n\n" + block + "\n", encoding="utf-8")
        self._send_json(200, {"ok": True})

    def log_message(self, format: str, *args) -> None:
        pass  # quiet — this is a throwaway local dev tool


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Profiling form: http://127.0.0.1:{PORT}/  (Ctrl+C to stop)")
    server.serve_forever()
