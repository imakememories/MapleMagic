"""The page's server: its static files, and a small JSON API over one Session."""
from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .session import Session

STATIC = Path(__file__).parent / "static"
# Explicit types: Windows' registry can map .js to text/plain, which browsers refuse as a module.
FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/debug.js": ("debug.js", "text/javascript; charset=utf-8"),
}


def serve(checkpoint: str, deck: str, host: str, port: int, open_browser: bool):
    session = Session(checkpoint, deck)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path not in FILES:
                return self.reply(404, b"not found", "text/plain")
            name, kind = FILES[path]
            self.reply(200, (STATIC / name).read_bytes(), kind)

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path == "/api/new":
                    seed = str(body.get("seed") or "").strip()
                    session.new(body.get("agent", "network"), int(body.get("sims") or 400), str(body.get("seat", "random")),
                                int(seed) if seed else None)
                elif self.path == "/api/choose":
                    session.choose(int(body["i"]))
                elif self.path == "/api/ask":
                    session.ask()
                elif self.path != "/api/state":
                    return self.reply(404, b"not found", "text/plain")
            except (ValueError, KeyError, TypeError, OSError) as err:
                return self.reply(400, json.dumps({"error": str(err)}).encode(), "application/json")
            self.reply(200, json.dumps(session.state(bool(body.get("debug")))).encode(), "application/json")

        def reply(self, code: int, data: bytes, kind: str):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer((host, port), Handler)
    url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}"
    print(f"Open {url} in a browser (Ctrl+C stops the server).", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
