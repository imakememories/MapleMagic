// The agent's thinking: what its search made of each decision, as plain
// tables. (Its hand is shown face up on the table meanwhile.)

const pct = x => x == null ? "" : (100 * x).toFixed(1) + "%";
const num = x => x == null ? "" : (x >= 0 ? "+" : "") + x.toFixed(2);

function el(tag, text) {
  const e = document.createElement(tag);
  if (text != null) e.textContent = text;
  return e;
}

function row(tag, cells) {
  const tr = el("tr");
  tr.append(...cells.map(c => el(tag, c)));
  return tr;
}

// One decision's choices; "*" marks the search's pick. The heuristic's priors are uniform, so they're left out.
function table(t, priors) {
  const tab = el("table");
  tab.append(row("th", ["", "choice", ...(priors ? ["prior"] : []), "visits", "Q"]));
  t.rows.forEach((r, i) => tab.append(row("td", [
    i === t.chose ? "*" : "", r.text, ...(priors ? [pct(r.prior)] : []), pct(r.share), num(r.q),
  ])));
  return tab;
}

function summary(t, label) {
  if (!t.sims) return t.note;
  return `${t.sims} sims in ${t.seconds.toFixed(2)} s. Search value ${num(t.value)}, ${label} ${num(t.net_value)} ` +
    "(for the side searching; +1 is a win).";
}

export function showThinking(box, s, ask) {
  box.hidden = !s.debug;
  if (!s.debug) return box.replaceChildren();
  const d = s.debug, priors = s.agent === "network";
  const label = priors ? "network value" : "heuristic value";
  const parts = [];
  if (s.choices.length) {
    const b = el("button", "Ask the agent about your decision");
    b.onclick = ask;
    parts.push(b);
  }
  if (d.advice) parts.push(el("h4", "The agent on your decision"), el("p", summary(d.advice, label)), table(d.advice, priors));
  d.thoughts.forEach((t, i) => parts.push(
    el("h4", `Agent decision ${i + 1} since your last choice`), el("p", summary(t, label)), table(t, priors)));
  if (!d.thoughts.length && !d.advice) parts.push(el("p", "The agent hasn't decided anything since your last choice."));
  box.replaceChildren(...parts);
}
