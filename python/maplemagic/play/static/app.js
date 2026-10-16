// The page: sends the person's actions to the server and redraws the table,
// action bar, log and thinking from each reply.
import { makeActions, renderActionBar } from "./actions.js";
import { showThinking } from "./debug.js";
import { closePopup } from "./popup.js";
import { renderTable } from "./table.js";

const $ = id => document.getElementById(id);
let busy = false;

function setStatus(text) {
  const prompt = document.querySelector("#actions .prompt");
  if (prompt) prompt.textContent = text;
  else $("actions").textContent = text;
}

async function call(path, body = {}) {
  if (busy) return;
  busy = true;
  document.body.classList.add("busy");
  setStatus("Thinking...");
  try {
    const r = await fetch(path, { method: "POST", body: JSON.stringify({ ...body, debug: $("debug").checked }) });
    const s = await r.json();
    if (s.error) setStatus("Error: " + s.error);
    else show(s);
  } catch (err) {
    setStatus("Error: " + err);
  } finally {
    busy = false;
    document.body.classList.remove("busy");
  }
}

const choose = i => call("/api/choose", { i });

function show(s) {
  closePopup();
  $("log").textContent = s.log.join("\n");
  $("log").scrollTop = $("log").scrollHeight;
  const actions = makeActions(s.choices, s.board ? s.board.anchors : [], choose);
  if (s.board) renderTable(s.board, actions, s.debug && s.debug.agent_hand);
  renderActionBar($("actions"), s.status, s.choices, actions, choose);
  showThinking($("thinking"), s, () => call("/api/ask"));
}

$("new").onsubmit = e => {
  e.preventDefault();
  call("/api/new", Object.fromEntries(new FormData(e.target)));
};
$("debug").onchange = () => call("/api/state");
call("/api/state");
