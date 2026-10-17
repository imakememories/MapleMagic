// The person's choices, put on the cards, players and stack items they name;
// the rest (pass, keep, yes/no, modes, and choices only on cards inside a
// pile) are buttons in the action bar.
import { openMenu } from "./popup.js";

// Keys of what's always in view: hands, battlefields, command zones, the stack, the players.
function inView(board) {
  const keys = new Set(["you", "opponent"]);
  for (const side of [board.you, board.opponent]) {
    for (const c of [...side.hand, ...side.battlefield, ...(side.command_zone || [])]) if (c.key) keys.add(c.key);
  }
  for (const item of board.stack) keys.add(item.key);
  return keys;
}

// choices: texts; choose(i, el) makes choice i (el: what was clicked).
export function makeActions(choices, board, choose) {
  const anchors = board ? board.anchors : [];
  const byKey = new Map();
  anchors.forEach((keys, i) => keys.forEach(k => byKey.set(k, [...(byKey.get(k) || []), i])));
  const shown = board ? inView(board) : new Set();
  const loose = choices.map((_, i) => i).filter(i => !(anchors[i] || []).some(k => shown.has(k)));

  // Makes `el` clickable for the choices naming item.key: one choice is made at once, more open a menu.
  function decorate(el, item) {
    const mine = byKey.get(item.key);
    if (!mine) return;
    el.classList.add("can");
    el.title = mine.map(i => choices[i]).join("\n");
    el.onclick = e => {
      e.stopPropagation();
      if (mine.length === 1) choose(mine[0], el);
      else openMenu(e.clientX, e.clientY, mine.map(i => ({ text: choices[i], run: () => choose(i, el) })));
    };
  }

  return { decorate, loose, has: key => byKey.has(key), mark: () => "" };
}

export function button(text, run, cls = "") {
  const b = document.createElement("button");
  b.textContent = text;
  b.className = cls;
  b.onclick = e => run(e.currentTarget);
  return b;
}

function line(cls, text) {
  const d = document.createElement("div");
  d.className = cls;
  d.textContent = text;
  return d;
}

// The prompt, what you just chose, a notice, buttons (the combat selection's, or a button per
// choice not on the board, plus passing to the next turn), and every choice as text.
// `mode`: a combat selection or null; `passTurn`: passes until the next turn.
export function renderActionBar(box, s, actions, mode, choose, passTurn) {
  const choices = s.choices || [];
  const kinds = s.board ? s.board.kinds : [];
  const parts = [line("prompt", mode ? mode.prompt : s.passing ? "Passing to the next turn..." : s.status)];
  for (const m of s.made || []) parts.push(line("made", "✔ You: " + m));
  if (s.notice) parts.push(line("notice", s.notice));
  if (mode && mode.hint) parts.push(line("hint", mode.hint));
  const buttons = document.createElement("div");
  buttons.className = "buttons";
  if (mode) buttons.append(...mode.buttons);
  else {
    buttons.append(...actions.loose.map(i => {
      const pass = kinds[i] === "Pass";
      const b = button(pass ? (s.board.stack.length ? "Pass (resolve the top of the stack)" : "Pass") : choices[i],
        el => choose(i, el), pass ? "pass" : "");
      return b;
    }));
    if (kinds.includes("Pass")) {
      const b = button("Pass until next turn", el => passTurn(el), "secondary");
      b.title = "Pass every step until the next turn starts; still stops when the opponent casts something, for blocks, and for other choices. Skips your attack.";
      buttons.append(b);
    }
  }
  parts.push(buttons);
  if (!mode && actions.loose.length < choices.length) {
    parts.push(line("hint", (actions.loose.length ? "Or click" : "Click") + " a highlighted card."));
  }
  if (choices.length) {
    const all = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `All choices as a list (${choices.length})`;
    all.append(summary, ...choices.map((c, i) => button(c, el => choose(i, el))));
    parts.push(all);
  }
  box.replaceChildren(...parts);
}
