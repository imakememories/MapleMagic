// The page: sends the person's actions to the server and redraws the table,
// action bar, log and thinking from each reply.
import { makeActions, renderActionBar } from "./actions.js";
import { showThinking } from "./debug.js";
import { closePopup } from "./popup.js";
import { openSetup } from "./setup.js";
import { renderTable } from "./table.js";

const $ = id => document.getElementById(id);
let busy = false;

function setStatus(text) {
  const prompt = document.querySelector("#actions .prompt");
  if (prompt) prompt.textContent = text;
  else $("actions").textContent = text;
}

// Resolves to the server's reply ({ error } on failure), after showing it.
async function call(path, body = {}) {
  if (busy) return { error: "still working on the last request" };
  busy = true;
  document.body.classList.add("busy");
  setStatus("Thinking...");
  let s;
  try {
    const r = await fetch(path, { method: "POST", body: JSON.stringify({ ...body, debug: $("debug").checked }) });
    s = await r.json();
  } catch (err) {
    s = { error: String(err) };
  } finally {
    busy = false;
    document.body.classList.remove("busy");
  }
  if (s.error) setStatus("Error: " + s.error);
  else show(s);
  return s;
}

const choose = i => call("/api/choose", { i });

function show(s) {
  closePopup();
  $("opponent").textContent = s.opponent ? "Against " + s.opponent : "";
  $("log").textContent = s.log.join("\n");
  $("log").scrollTop = $("log").scrollHeight;
  const actions = makeActions(s.choices, s.board ? s.board.anchors : [], choose);
  if (s.board) renderTable(s.board, actions, s.debug && s.debug.agent_hand);
  renderActionBar($("actions"), s.status, s.choices, actions, choose);
  showThinking($("thinking"), s, () => call("/api/ask"));
}

const setup = () => openSetup(opts => call("/api/new", opts));
$("setup-open").onclick = setup;
$("debug").onchange = () => call("/api/state");
call("/api/state").then(s => { if (!s.board) setup(); });
