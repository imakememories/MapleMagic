// Where you stop to act instead of passing: per step, on your turns and on
// the opponent's, kept in this browser and sent with every request.
import { STEPS } from "./timeline.js";

const KEY = "maplemagic.stops";
const DEFAULTS = { stops: { you: ["Main1", "Main2"], opponent: ["DeclareAttackers", "End"] }, resolve_own: true };
const copy = v => JSON.parse(JSON.stringify(v));

function load() {
  try {
    const v = JSON.parse(localStorage.getItem(KEY));
    if (v && v.stops && Array.isArray(v.stops.you) && Array.isArray(v.stops.opponent)) return v;
  } catch { /* no storage here */ }
  return copy(DEFAULTS);
}

let prefs = load();

function save() {
  try {
    localStorage.setItem(KEY, JSON.stringify(prefs));
  } catch { /* kept for this page only */ }
}

export const getPrefs = () => prefs;

function checkbox(checked, onchange) {
  const c = document.createElement("input");
  c.type = "checkbox";
  c.checked = checked;
  c.onchange = () => onchange(c.checked);
  return c;
}

// The settings as a table of steps x (your turn, the opponent's); `changed` runs after each change.
export function renderStops(box, changed) {
  const set = (who, step, on) => {
    const s = new Set(prefs.stops[who]);
    if (on) s.add(step);
    else s.delete(step);
    prefs.stops[who] = STEPS.map(([k]) => k).filter(k => s.has(k));
    save();
    changed();
  };
  const table = document.createElement("table");
  table.innerHTML = "<tr><th>Stop at</th><th>Your turn</th><th>Opponent's turn</th></tr>";
  for (const [k, label] of STEPS) {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    name.textContent = label;
    tr.append(name, ...["you", "opponent"].map(who => {
      const td = document.createElement("td");
      td.append(checkbox(prefs.stops[who].includes(k), on => set(who, k, on)));
      return td;
    }));
    table.append(tr);
  }
  const own = document.createElement("label");
  own.append(checkbox(prefs.resolve_own, on => { prefs.resolve_own = on; save(); changed(); }),
    " Let my own spells and abilities resolve (pass while only mine are on the stack)");
  const note = document.createElement("p");
  note.className = "note";
  note.textContent = "In an unchecked step you pass automatically while the stack is empty. You always stop when the " +
    "opponent has something on the stack, and for attacks, blocks, targets and other choices, unless all you could " +
    "do is make mana. Attacker and blocker stops only apply while a creature is attacking.";
  const reset = document.createElement("button");
  reset.type = "button";
  reset.textContent = "Reset to defaults";
  reset.onclick = () => {
    prefs = copy(DEFAULTS);
    save();
    renderStops(box, changed);
    changed();
  };
  box.replaceChildren(table, own, note, reset);
}
