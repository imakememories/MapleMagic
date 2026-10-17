// The log, turn by turn and step by step: the choices made, and what happened
// (damage and its source, life, draws, cards moving, tokens, attacks). What's
// new since your last decision is marked, and repeated in a recap.
import { preview } from "./art.js";

// The steps of a turn as the page names them; Untap and Cleanup have no priority.
export const STEPS = [
  ["Upkeep", "Upkeep"], ["Draw", "Draw"], ["Main1", "Main 1"], ["BeginCombat", "Begin combat"],
  ["DeclareAttackers", "Attackers"], ["DeclareBlockers", "Blockers"], ["CombatDamage", "Damage"],
  ["EndCombat", "End combat"], ["Main2", "Main 2"], ["End", "End step"],
];
const NAMES = new Map([...STEPS, ["Untap", "Untap"], ["Cleanup", "Cleanup"]]);

function el(tag, cls, t) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (t != null) e.textContent = t;
  return e;
}

// This turn's entries by step.
export function stepEntries(s, turn, active) {
  const out = new Map();
  for (const g of s.timeline || []) {
    if (g.mulligans || g.turn !== turn || g.active !== active) continue;
    out.set(g.step, [...(out.get(g.step) || []), ...g.entries]);
  }
  return out;
}

const turnName = g => g.mulligans ? "Mulligans" : `Turn ${g.turn}: ${g.active === "you" ? "your" : "the opponent's"} turn`;

function entry(e, since) {
  const d = el("div", `entry ${e.kind} ${e.who}` + (e.seq >= since ? " new" : ""));
  if (e.kind === "choice") d.append(el("b", "", e.who === "you" ? "You: " : "Opponent: "));
  d.append(document.createTextNode(e.text));
  if (e.life) d.append(el("span", "life", `${e.life[0]} → ${e.life[1]}`));
  if (e.cards) d.addEventListener("mouseenter", () => preview(e.cards[0]));
  return d;
}

// Groups as headed blocks: a turn header when the turn changes, then each step with its entries.
function blocks(groups, since) {
  const out = [];
  let last = null;
  for (const g of groups) {
    const name = turnName(g);
    if (name !== last) out.push(el("div", "turn-head " + (g.active || ""), name));
    last = name;
    const step = el("div", "step-block");
    if (!g.mulligans) step.append(el("div", "step-head", NAMES.get(g.step) || g.step));
    step.append(...g.entries.map(e => entry(e, since)));
    out.push(step);
  }
  return out;
}

export function renderLog(box, s) {
  const atEnd = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  box.replaceChildren(...(s.intro ? [el("div", "intro", s.intro)] : []), ...blocks(s.timeline || [], s.since));
  if (atEnd || !box.dataset.shown) box.scrollTop = box.scrollHeight;
  box.dataset.shown = "1";
}

// What happened since your last decision, by step.
export function renderRecap(box, s) {
  const groups = (s.timeline || [])
    .map(g => ({ ...g, entries: g.entries.filter(e => e.seq >= s.since) }))
    .filter(g => g.entries.length);
  box.hidden = !groups.length;
  box.replaceChildren(el("h4", "", "Since your last decision"), ...blocks(groups, s.since));
}
