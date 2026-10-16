// The person's choices, put on the cards, players and stack items they name;
// the rest (pass, keep, yes/no, modes...) go on buttons in the action bar.
import { openMenu } from "./popup.js";

// choices: texts; anchors: per choice, the keys it names; choose(i) makes choice i.
export function makeActions(choices, anchors, choose) {
  const byKey = new Map();
  anchors.forEach((keys, i) => keys.forEach(k => byKey.set(k, [...(byKey.get(k) || []), i])));
  const loose = choices.map((_, i) => i).filter(i => !(anchors[i] || []).length);

  // Makes `el` clickable for the choices naming `key`: one choice is made at once, more open a menu.
  function decorate(el, key) {
    const mine = byKey.get(key);
    if (!mine) return;
    el.classList.add("can");
    el.title = mine.map(i => choices[i]).join("\n");
    el.onclick = e => {
      e.stopPropagation();
      if (mine.length === 1) choose(mine[0]);
      else openMenu(e.clientX, e.clientY, mine.map(i => ({ text: choices[i], run: () => choose(i) })));
    };
  }

  return { decorate, loose, has: key => byKey.has(key) };
}

function button(text, run) {
  const b = document.createElement("button");
  b.textContent = text;
  b.onclick = run;
  return b;
}

// The prompt, a button per choice not on the board, and every choice as text for when a card is hard to find.
export function renderActionBar(box, status, choices, actions, choose) {
  const prompt = document.createElement("div");
  prompt.className = "prompt";
  prompt.textContent = status;
  const buttons = document.createElement("div");
  buttons.className = "buttons";
  buttons.append(...actions.loose.map(i => button(choices[i], () => choose(i))));
  const parts = [prompt, buttons];
  if (actions.loose.length < choices.length) {
    const hint = document.createElement("div");
    hint.className = "hint";
    hint.textContent = (actions.loose.length ? "Or click" : "Click") + " a highlighted card.";
    parts.push(hint);
  }
  if (choices.length) {
    const all = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `All choices (${choices.length})`;
    all.append(summary, ...choices.map((c, i) => button(c, () => choose(i))));
    parts.push(all);
  }
  box.replaceChildren(...parts);
}
