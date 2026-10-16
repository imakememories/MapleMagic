// One checkpoint's details: its network, how its run trained and searched,
// this generation's numbers, and the run's evaluations by generation.

const fmt = x => x == null ? "" : typeof x === "number" && !Number.isInteger(x) ? x.toFixed(Math.abs(x) >= 10 ? 1 : 3) : String(x);
const words = key => key.replace(/_/g, " ").replace("heldout", "held-out").replace("vs prev", "vs previous");

function el(tag, text) {
  const e = document.createElement(tag);
  if (text != null) e.textContent = text;
  return e;
}

export function describeSearch(s) {
  const method = s.maple_worlds ? `MAPLE, ${s.maple_worlds} worlds (${s.maple_select}${s.maple_resample ? ", resampled" : ""})`
    : s.pimc_worlds ? `PIMC, ${s.pimc_worlds} worlds` : "IS-MCTS";
  const extras = [s.collisions && "collisions", s.perfect_obs && "perfect observation",
    s.world_sampler === "oracle" && "oracle worlds", s.value_mix !== 1 && `value mix ${s.value_mix}`].filter(Boolean);
  return [method, ...extras].join(", ");
}

function facts(d) {
  const n = d.network, t = d.training, g = d.this_gen;
  const rows = [
    ["Checkpoint", `${d.run} / ${d.file}` + (d.gen != null ? ` (generation ${d.gen})` : "")],
    ["Network", `${n.d}-wide, ${n.layers} layer${n.layers === 1 ? "" : "s"}, ${n.heads} heads, ` +
      `${(n.parameters / 1e6).toFixed(2)}M parameters${n.attributes ? ", card attributes" : ""}`],
    ["Trained", `${t.generations} generations of ${t.games_per_gen} games, ${t.sims} sims a move, ` +
      `${t.value_target} value targets, ${t.policy_target} policy targets`],
    ["Decks", t.pairings.map(p => p.join(" vs ")).join("; ")],
    ["Search as trained", describeSearch(d.search)],
  ];
  if (Object.keys(g).length) {
    rows.push(["This generation", Object.entries(g).map(([k, v]) => `${words(k)} ${fmt(v)}`).join(", ")]);
  }
  const dl = el("dl");
  for (const [k, v] of rows) dl.append(el("dt", k), el("dd", v));
  return dl;
}

function evals(d) {
  if (!d.evals.length) return el("p", "No evaluations in this run's log.");
  const tab = el("table");
  const head = el("tr");
  head.append(...["gen", "vs heuristic", "vs previous", "held-out value AUC"].map(h => el("th", h)));
  tab.append(head);
  for (const r of d.evals) {
    const tr = el("tr");
    if (r.gen === d.gen) tr.className = "chosen";
    tr.append(...[r.gen, fmt(r.vs_heuristic), fmt(r.vs_prev), fmt(r.heldout_value_auc)].map(x => el("td", String(x))));
    tab.append(tr);
  }
  return tab;
}

export function renderDetails(box, d) {
  box.replaceChildren(facts(d), el("h5", "Evaluations (score out of 1; the chosen generation highlighted)"), evals(d),
    ...d.notes.map(n => el("p", n)));
}
