"""Card images from Scryfall, fetched once per card and kept on disk."""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .. import CARD_NAMES

URL = "https://api.scryfall.com/cards/named?exact={}&format=image&version=normal"
SEARCH = "https://api.scryfall.com/cards/search?q={}&unique=art"
# Scryfall asks for a User-Agent, an Accept header and 50-100 ms between requests.
HEADERS = {"User-Agent": "MapleMagic/0.1 (local play page)", "Accept": "*/*"}
GAP = 0.1


class Art:
    """Images of the engine's cards by name; None when Scryfall has none or can't be reached."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.names = set(CARD_NAMES)
        self.lock = threading.Lock()
        self.last = 0.0

    def path(self, name: str) -> Path:
        return self.folder / (re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") + ".jpg")

    def get(self, name: str) -> bytes | None:
        if name not in self.names:
            return None
        path = self.path(name)
        missing = path.with_suffix(".missing")
        if path.exists():
            return path.read_bytes()
        if missing.exists():
            return None
        # One fetch at a time.
        with self.lock:
            if path.exists():
                return path.read_bytes()
            try:
                data = self.fetch(name)
            except (urllib.error.HTTPError, LookupError, ValueError) as err:
                # Not found stays not found; a refusal or an outage may pass.
                if not isinstance(err, urllib.error.HTTPError) or err.code == 404:
                    self.folder.mkdir(parents=True, exist_ok=True)
                    missing.touch()
                return None
            except OSError:
                return None
            self.folder.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".part")
            part.write_bytes(data)
            part.replace(path)
            return data

    def request(self, url: str) -> bytes:
        """One request, spaced from the last as Scryfall asks."""
        time.sleep(max(0.0, self.last + GAP - time.monotonic()))
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=15) as r:
                return r.read()
        finally:
            self.last = time.monotonic()

    def fetch(self, name: str) -> bytes:
        """The card's image; for a token (the engine's "Treasure Token"), the art of a Scryfall token of that name."""
        try:
            return self.request(URL.format(urllib.parse.quote(name)))
        except urllib.error.HTTPError as err:
            if err.code != 404 or not name.endswith(" Token"):
                raise
        want = name.removesuffix(" Token")
        found = json.loads(self.request(SEARCH.format(urllib.parse.quote(f't:token !"{want}"'))))["data"]
        # A one-sided token of that name, else the matching side of a two-sided one.
        sides = [c for c in found if "image_uris" in c] + [f for c in found for f in c.get("card_faces", [])]
        side = next((s for s in sides if s.get("name") == want and "image_uris" in s), None)
        if side is None:
            raise LookupError(f"no token named {want}")
        return self.request(side["image_uris"]["normal"])
