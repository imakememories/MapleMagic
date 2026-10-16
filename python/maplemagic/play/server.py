"""The page's server: its static files, card art, and a small JSON API over one Session."""
from __future__ import annotations

import json
import urllib.parse
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .art import Art
from .session import Session

STATIC = Path(__file__).parent / "static"
# Explicit types: Windows' registry can map .js to text/plain, which browsers refuse as a module.
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}


def api(session: Session, path: str, body: dict) -> tuple[int, dict]:
    """One API call: (HTTP status, JSON reply)."""
    try:
        if path == "/api/new":
            seed = str(body.get("seed") or "").strip()
            session.new(body.get("agent", "network"), int(body.get("sims") or 400), str(body.get("seat", "random")),
                        int(seed) if seed else None)
        elif path == "/api/choose":
            session.choose(int(body["i"]))
        elif path == "/api/ask":
            session.ask()
        elif path != "/api/state":
            return 404, {"error": f"no {path}"}
    except (ValueError, KeyError, TypeError, OSError) as err:
        return 400, {"error": str(err)}
    return 200, session.state(bool(body.get("debug")))


def serve(checkpoint: str, deck: str, host: str, port: int, open_browser: bool, art_dir: Path, device: str = "auto"):
    session = Session(checkpoint, deck, device)
    art = Art(art_dir)
    # Match objects stay on the thread that made them, so every game call runs on this one.
    game = ThreadPoolExecutor(max_workers=1)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            if path.startswith("/art/"):
                data = art.get(urllib.parse.unquote(path[len("/art/"):]))
                if data is None:
                    return self.reply(404, b"no art", "text/plain")
                return self.reply(200, data, "image/jpeg", cache=True)
            name = path.lstrip("/") or "index.html"
            # Only files in the static folder itself, by exact name.
            files = {p.name: p for p in STATIC.iterdir() if p.suffix in TYPES}
            if name not in files:
                return self.reply(404, b"not found", "text/plain")
            self.reply(200, files[name].read_bytes(), TYPES[files[name].suffix])

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except ValueError as err:
                return self.reply(400, json.dumps({"error": str(err)}).encode(), "application/json")
            code, reply = game.submit(api, session, self.path, body).result()
            self.reply(code, json.dumps(reply).encode(), "application/json")

        def reply(self, code: int, data: bytes, kind: str, cache: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            if cache:
                self.send_header("Cache-Control", "max-age=31536000")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}"
    print(f"Open {url} in a browser (Ctrl+C stops the server). Card art is kept in {art_dir}.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
