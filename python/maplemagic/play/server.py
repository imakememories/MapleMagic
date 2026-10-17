"""The page's server: its static files, card art, the models to choose from, and a small JSON API over one Session."""
from __future__ import annotations

import json
import urllib.parse
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import DECK_PAIRS
from .art import Art
from .models import Catalog
from .session import Session

STATIC = Path(__file__).parent / "static"
# Explicit types: Windows' registry can map .js to text/plain, which browsers refuse as a module.
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}


# What a bad request or an unloadable checkpoint raises; anything else is a bug and stays loud.
BAD = (ValueError, KeyError, TypeError, OSError, RuntimeError)


def api(session: Session, path: str, body: dict) -> tuple[int, dict]:
    """One game call: (HTTP status, JSON reply)."""
    try:
        session.prefs(body)
        if path == "/api/new":
            session.new(body)
        elif path == "/api/choose":
            session.choose(int(body["i"]))
        elif path == "/api/attack":
            session.attack(list(body["attackers"]))
        elif path == "/api/block":
            session.block([list(p) for p in body["blocks"]])
        elif path == "/api/pass_turn":
            session.pass_turn()
        elif path == "/api/ask":
            session.ask()
        elif path != "/api/state":
            return 404, {"error": f"no {path}"}
    except BAD as err:
        return 400, {"error": str(err)}
    finally:
        session.progress = {"busy": False, "version": session.progress["version"]}
    return 200, session.state(session.debug)


def progress(session: Session, query: dict) -> dict:
    """While a request runs: the agent's search so far, and the board after the latest move if newer than version v."""
    p = session.progress
    try:
        seen = int((query.get("v") or ["0"])[0])
    except ValueError:
        seen = 0
    return {k: v for k, v in p.items() if k != "snapshot" or p["version"] > seen}


def catalog_api(catalog: Catalog, deck: str, path: str, query: dict) -> tuple[int, dict]:
    """The models to choose from (/api/models), and one checkpoint's details (/api/model?id=run/file)."""
    try:
        if path == "/api/models":
            return 200, {"runs": catalog.runs(), "default": catalog.default_id(), "default_deck": deck,
                         "pairs": [list(p) for p in DECK_PAIRS]}
        if path == "/api/model":
            return 200, catalog.details(query.get("id", [""])[0])
    except BAD as err:
        return 400, {"error": str(err)}
    return 404, {"error": f"no {path}"}


def serve(checkpoint: str, runs: Path, deck: str, host: str, port: int, open_browser: bool, art_dir: Path,
          device: str = "auto"):
    catalog = Catalog(runs, Path(checkpoint))
    session = Session(catalog, device)
    art = Art(art_dir)
    # Match objects stay on the thread that made them, so every game call runs on this one.
    game = ThreadPoolExecutor(max_workers=1)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            path = url.path
            if path == "/api/progress":
                reply = progress(session, urllib.parse.parse_qs(url.query))
                return self.reply(200, json.dumps(reply).encode(), "application/json")
            if path.startswith("/api/"):
                code, reply = game.submit(catalog_api, catalog, deck, path, urllib.parse.parse_qs(url.query)).result()
                return self.reply(code, json.dumps(reply).encode(), "application/json")
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
