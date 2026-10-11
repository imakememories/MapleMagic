"""A barebones page for playing against an agent: `mmz web`, then open the printed address."""
from __future__ import annotations

import json
import random
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from . import Match
from .play import Agent

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ognis vs agent</title>
<style>
body { font-family: system-ui, sans-serif; margin: 12px; max-width: 900px; background: #fcfcfb; color: #0b0b0b; }
pre { white-space: pre-wrap; background: #f0efec; padding: 8px; font-size: 14px; }
button { display: block; width: 100%; text-align: left; margin: 4px 0; padding: 8px; font-size: 15px; }
#log { max-height: 220px; overflow-y: auto; }
@media (prefers-color-scheme: dark) { body { background: #1a1a19; color: #fff; } pre { background: #2c2c2a; } }
</style></head><body>
<form id="new">
  Agent <select name="agent"><option value="network">best network</option><option value="heuristic">heuristic</option></select>
  sims <input name="sims" value="400" size="4">
  seat <select name="seat"><option>random</option><option>0</option><option>1</option></select>
  <input type="submit" value="New game">
</form>
<h3 id="status"></h3>
<pre id="board">Start a new game.</pre>
<div id="choices"></div>
<h4>What happened</h4>
<pre id="log"></pre>
<script>
async function call(path, body) {
  document.getElementById("status").textContent = "The agent is thinking...";
  const r = await fetch(path, {method: "POST", body: JSON.stringify(body || {})});
  show(await r.json());
}
function show(s) {
  document.getElementById("status").textContent = s.status;
  document.getElementById("board").textContent = s.board;
  document.getElementById("log").textContent = s.log.join("\\n");
  const box = document.getElementById("choices");
  box.innerHTML = "";
  s.choices.forEach((c, i) => {
    const b = document.createElement("button");
    b.textContent = c;
    b.onclick = () => call("/choose", {i: i});
    box.appendChild(b);
  });
  const log = document.getElementById("log");
  log.scrollTop = log.scrollHeight;
}
document.getElementById("new").onsubmit = e => {
  e.preventDefault();
  call("/new", Object.fromEntries(new FormData(e.target)));
};
fetch("/state", {method: "POST"}).then(r => r.json()).then(show);
</script></body></html>"""


class Session:
    """One game at a time."""

    def __init__(self, checkpoint: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = checkpoint
        self.agents: dict[tuple, Agent] = {}
        self.m: Match | None = None
        self.human = 0
        self.agent: Agent | None = None
        self.log: list[str] = []

    def new(self, agent: str, sims: int, seat: str):
        key = (agent, sims)
        if key not in self.agents:
            self.agents[key] = Agent(self.checkpoint if agent == "network" else None, sims, self.device)
        self.agent = self.agents[key]
        self.human = random.randrange(2) if seat == "random" else int(seat)
        seed = random.randrange(1 << 31)
        self.m = Match("Ognis", "Ognis", seed)
        self.log = [f"Game {seed}: you are seat {self.human}, against the {agent} agent at {sims} simulations."]
        self.advance()

    def choose(self, i: int):
        if self.m is None or self.m.is_over() or self.m.to_act() != self.human:
            return
        choices = self.m.choices(self.human)
        if 0 <= i < len(choices):
            self.log.append(f"You: {choices[i]}")
            self.m.apply(i)
            self.advance()

    def advance(self):
        """Lets the agent move until it is the person's turn to choose."""
        while not self.m.is_over() and self.m.to_act() != self.human:
            i, _ = self.agent.choose(self.m, random.randrange(1 << 31))
            self.log.append(f"Opponent: {self.m.choices(self.human)[i]}")
            self.m.apply(i)

    def state(self) -> dict:
        if self.m is None:
            return {"status": "Choose an agent and start a game.", "board": "", "choices": [], "log": []}
        if self.m.is_over():
            result = self.m.outcome()
            status = "You won!" if result == f"win {self.human}" else "The agent won." if result.startswith("win") \
                else f"Game over: {result}"
            choices = []
        else:
            status, choices = "Your decision", self.m.choices(self.human)
        return {"status": status, "board": self.m.render(self.human), "choices": choices, "log": self.log[-200:]}


def serve(checkpoint: str, host: str, port: int):
    session = Session(checkpoint)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.reply(PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/new":
                session.new(body.get("agent", "network"), int(body.get("sims") or 400), body.get("seat", "random"))
            elif self.path == "/choose":
                session.choose(int(body.get("i", -1)))
            self.reply(json.dumps(session.state()).encode(), "application/json")

        def reply(self, data: bytes, kind: str):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    print(f"Open http://{host if host != '0.0.0.0' else 'localhost'}:{port} in a browser (Ctrl+C stops the server).",
          flush=True)
    HTTPServer((host, port), Handler).serve_forever()
