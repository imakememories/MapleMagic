// The game: sends the person's actions and shows the status, board, choices and log.
import { showThinking } from "./debug.js";

const $ = id => document.getElementById(id);
let busy = false;

async function call(path, body = {}) {
  if (busy) return;
  busy = true;
  $("status").textContent = "Thinking...";
  try {
    const r = await fetch(path, { method: "POST", body: JSON.stringify({ ...body, debug: $("debug").checked }) });
    const s = await r.json();
    if (s.error) $("status").textContent = "Error: " + s.error;
    else show(s);
  } catch (err) {
    $("status").textContent = "Error: " + err;
  } finally {
    busy = false;
  }
}

function show(s) {
  $("status").textContent = s.status;
  $("board").textContent = s.board;
  $("log").textContent = s.log.join("\n");
  $("choices").replaceChildren(...s.choices.map((c, i) => {
    const b = document.createElement("button");
    b.textContent = c;
    b.onclick = () => call("/api/choose", { i });
    return b;
  }));
  showThinking($("thinking"), s, () => call("/api/ask"));
}

$("new").onsubmit = e => {
  e.preventDefault();
  call("/api/new", Object.fromEntries(new FormData(e.target)));
};
$("debug").onchange = () => call("/api/state");
call("/api/state");
