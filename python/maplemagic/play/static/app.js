// The page: sends the person's actions to the server and redraws the table,
// action bar, log and thinking from each reply. While a request runs, it
// shows the agent's progress and the board after each of its moves.
import { makeActions, renderActionBar } from "./actions.js";
import { combatMode, drawLines } from "./combat.js";
import { showThinking } from "./debug.js";
import { closePopup } from "./popup.js";
import { getPrefs, renderStops } from "./prefs.js";
import { openSetup } from "./setup.js";
import { renderTable } from "./table.js";
import { renderLog, renderRecap } from "./timeline.js";

const $ = id => document.getElementById(id);
let busy = false;
let last = null; // the last state shown
let lines = [];

function showProgress(label, p) {
  const box = $("progress");
  const parts = [];
  if (label) parts.push(Object.assign(document.createElement("div"), { className: "made", textContent: "✔ You: " + label }));
  const t = p && p.thinking;
  if (t) {
    const bar = document.createElement("progress");
    bar.max = t.total;
    bar.value = t.sims;
    const what = `The opponent is thinking: decision ${t.decision}, ${t.sims} of ${t.total} sims` +
      (t.seconds ? ` (${t.seconds.toFixed(1)} s)` : "");
    parts.push(Object.assign(document.createElement("div"), { textContent: what }), bar);
    if (t.leaning) parts.push(Object.assign(document.createElement("div"), {
      className: "leaning", textContent: `Leaning to: ${t.leaning} (${(100 * t.share).toFixed(0)}% of visits)` }));
  } else if (busy) parts.push(Object.assign(document.createElement("div"), { textContent: "Working..." }));
  box.hidden = !parts.length;
  box.replaceChildren(...parts);
}

// While a request runs: the agent's search and each new board.
async function poll(label) {
  let seen = -1;
  while (busy) {
    try {
      const p = await (await fetch("/api/progress?v=" + Math.max(seen, 0))).json();
      if (!busy) break;
      if (p.snapshot && p.version > seen) {
        seen = p.version;
        show(p.snapshot, true);
      }
      showProgress(label, p);
    } catch { /* the request's own reply reports errors */ }
    await new Promise(r => setTimeout(r, 250));
  }
}

// Resolves to the server's reply ({ error } on failure), after showing it. `label`: what the person chose.
async function call(path, body = {}, label = "", el = null) {
  if (busy) return { error: "still working on the last request" };
  busy = true;
  document.body.classList.add("busy");
  if (el) el.classList.add("sent");
  showProgress(label, null);
  poll(label);
  let s;
  try {
    const r = await fetch(path, { method: "POST", body: JSON.stringify({ ...body, ...getPrefs(), debug: $("debug").checked }) });
    s = await r.json();
  } catch (err) {
    s = { error: String(err) };
  } finally {
    busy = false;
    document.body.classList.remove("busy");
    showProgress("", null);
  }
  if (s.error) {
    if (last) show(last);
    const prompt = document.querySelector("#actions .prompt");
    if (prompt) prompt.textContent = "Error: " + s.error;
  } else show(s);
  return s;
}

const choose = (i, el) => call("/api/choose", { i }, last.choices[i], el);
const passTurn = el => call("/api/pass_turn", {}, "Pass until the next turn", el);

// Draws state `s`; `live` while the request that made it still runs.
function show(s, live = false) {
  if (!live) last = s;
  closePopup();
  $("opponent").textContent = s.opponent ? "Against " + s.opponent : "";
  renderLog($("log"), s);
  renderRecap($("recap"), s);
  const actions = makeActions(s.choices, s.board, choose);
  const mode = live ? null : combatMode(s, actions, call, () => show(last));
  const deco = mode || actions;
  if (s.board) renderTable(s, deco, getPrefs().stops);
  renderActionBar($("actions"), live ? { ...s, made: [] } : s, actions, mode, choose, passTurn);
  lines = mode ? mode.lines : [];
  drawLines(lines);
  if (!live) showThinking($("thinking"), s, () => call("/api/ask", {}, "Ask the agent"));
}

renderStops($("stops"), () => { if (last && !busy) show(last); });
addEventListener("resize", () => drawLines(lines));
addEventListener("scroll", () => drawLines(lines), true);
const setup = () => openSetup(opts => call("/api/new", opts, "New game"));
$("setup-open").onclick = setup;
$("debug").onchange = () => call("/api/state");
call("/api/state").then(s => { if (!s.board) setup(); });
