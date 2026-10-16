// The new-game dialog: the agent (a network from any run and generation, or
// the heuristic or random agent), how it searches, the decks, seat and seed.
import { renderDetails } from "./model.js";

const $ = id => document.getElementById(id);
let catalog = null; // /api/models
const details = new Map(); // checkpoint id -> /api/model
let trained = null; // the chosen checkpoint's search as trained

async function getJSON(url) {
  const r = await fetch(url);
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}

function option(value, text) {
  const o = document.createElement("option");
  o.value = value;
  o.textContent = text;
  return o;
}

const agentKind = () => document.querySelector('input[name="agent"]:checked').value;
const chosenRun = () => catalog.runs.find(r => r.name === $("run").value);
const chosenId = () => `${$("run").value}/${$("checkpoint").value}`;

const runLabel = r => `${r.name}: ${r.decks.join(", ")}, ${r.trained ?? 0} of ${r.generations} generations` +
  (r.best ? `, best vs heuristic ${r.best.vs_heuristic.toFixed(2)} (gen ${r.best.gen})` : "");

function fillRuns() {
  const groups = new Map();
  for (const r of catalog.runs) groups.set(r.format, [...(groups.get(r.format) || []), r]);
  $("run").replaceChildren(...[...groups].map(([format, runs]) => {
    const g = document.createElement("optgroup");
    g.label = format;
    g.append(...runs.map(r => option(r.name, runLabel(r))));
    return g;
  }));
}

function fillCheckpoints(pick) {
  const run = chosenRun();
  $("checkpoint").replaceChildren(...run.checkpoints.map(c => option(c.file, c.label)));
  if (run.checkpoints.some(c => c.file === pick)) $("checkpoint").value = pick;
  setDecks(run.pairings[0][0], run.pairings[0][1]);
  showDetails();
}

async function showDetails() {
  const id = chosenId(), box = $("model-details");
  box.textContent = "Loading...";
  try {
    if (!details.has(id)) details.set(id, await getJSON("/api/model?id=" + encodeURIComponent(id)));
    if (id !== chosenId()) return; // the choice changed meanwhile
    renderDetails(box, details.get(id));
    trained = details.get(id).search;
    setSearch(trained);
  } catch (err) {
    box.textContent = "Can't read this checkpoint: " + err.message;
    trained = null;
  }
}

function setSearch(s) {
  $("method").value = s.maple_worlds ? "maple" : s.pimc_worlds ? "pimc" : "ismcts";
  $("worlds").value = s.maple_worlds || s.pimc_worlds || 8;
  $("maple_select").value = s.maple_select;
  $("maple_resample").checked = s.maple_resample;
  $("collisions").checked = s.collisions;
  $("perfect_obs").checked = s.perfect_obs;
  $("world_sampler").value = s.world_sampler;
  $("value_mix").value = s.value_mix;
  syncFields();
}

// Shows only the fields that apply to the agent and the search method.
function syncFields() {
  const kind = agentKind(), method = $("method").value;
  $("network-pick").hidden = $("network-search").hidden = kind !== "network";
  $("sims").closest("label").hidden = kind === "random";
  $("worlds").closest("label").hidden = method === "ismcts";
  $("maple_select").closest("label").hidden = $("maple_resample").closest("label").hidden = method !== "maple";
  deckNote();
}

function setDecks(you, agent) {
  $("deck-you").value = you;
  $("deck-agent").value = agent;
  deckNote();
}

function deckNote() {
  const pair = [$("deck-you").value, $("deck-agent").value];
  const run = chosenRun();
  let note = "";
  if (!catalog.pairs.some(p => p[0] === pair[0] && p[1] === pair[1])) note = "These decks can't play each other (different formats).";
  else if (agentKind() === "network" && run && !run.decks.includes(pair[1])) note = `This network never trained with ${pair[1]}.`;
  $("deck-note").textContent = note;
}

function options() {
  const method = $("method").value, worlds = Number($("worlds").value);
  return {
    agent: agentKind(),
    checkpoint: chosenId(),
    sims: $("sims").value,
    maple_worlds: method === "maple" ? worlds : 0,
    pimc_worlds: method === "pimc" ? worlds : 0,
    maple_select: $("maple_select").value,
    maple_resample: $("maple_resample").checked,
    collisions: $("collisions").checked,
    perfect_obs: $("perfect_obs").checked,
    world_sampler: $("world_sampler").value,
    value_mix: $("value_mix").value,
    decks: [$("deck-you").value, $("deck-agent").value],
    seat: $("seat").value,
    seed: $("seed").value,
  };
}

// start(options) starts a game and resolves to the server's reply.
async function build(start) {
  catalog = await getJSON("/api/models");
  fillRuns();
  const decks = [...new Set(catalog.pairs.flat())].sort();
  for (const id of ["deck-you", "deck-agent"]) $(id).replaceChildren(...decks.map(d => option(d, d)));
  const [run, file] = catalog.default.split("/");
  if (catalog.runs.some(r => r.name === run)) $("run").value = run;
  if (catalog.runs.length) {
    fillCheckpoints(file);
  } else {
    const network = document.querySelector('input[name="agent"][value="network"]');
    network.disabled = true;
    document.querySelector('input[name="agent"][value="heuristic"]').checked = true;
    setDecks(catalog.default_deck, catalog.default_deck);
  }
  $("run").onchange = () => fillCheckpoints();
  $("checkpoint").onchange = showDetails;
  $("method").onchange = syncFields;
  $("as-trained").onclick = () => trained && setSearch(trained);
  for (const r of document.querySelectorAll('input[name="agent"]')) r.onchange = syncFields;
  $("deck-you").onchange = $("deck-agent").onchange = deckNote;
  $("setup-cancel").onclick = () => $("setup").close();
  $("setup-form").onsubmit = async e => {
    e.preventDefault();
    $("setup-error").textContent = "Starting...";
    const s = await start(options());
    if (s && s.error) {
      $("setup-error").textContent = "Error: " + s.error;
    } else {
      $("setup-error").textContent = "";
      $("setup").close();
    }
  };
  syncFields();
}

export async function openSetup(start) {
  $("setup").showModal();
  if (catalog) return;
  try {
    await build(start);
  } catch (err) {
    catalog = null;
    $("setup-error").textContent = "Can't list the models: " + err.message;
  }
}
