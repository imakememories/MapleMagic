// The table: each player's side (life, library, graveyard, exile, command
// zone, battlefield, hand) and the turn and stack between them.
import { abilityElement, cardElement } from "./card.js";
import { openZone } from "./popup.js";

function div(cls, ...children) {
  const d = document.createElement("div");
  d.className = cls;
  d.append(...children);
  return d;
}

function text(cls, t) {
  const d = div(cls);
  d.textContent = t;
  return d;
}

const cards = (list, actions) => list.map(c => {
  const el = cardElement(c);
  if (c.key) actions.decorate(el, c.key);
  return el;
});

// A face-down library, or a graveyard or exile showing its top card; a click lays the pile out.
function pile(label, list, actions, faceDown) {
  const empty = faceDown ? faceDown === "0" : !list.length;
  const top = faceDown || empty ? div("card back" + (empty ? " empty" : "")) : cardElement(list[0]);
  const p = div("pile", top, text("count", `${label} ${faceDown ? faceDown : list.length}`));
  if (!faceDown && list.length) {
    p.classList.add("open");
    if (list.some(c => c.key && actions.has(c.key))) top.classList.add("can");
    p.onclick = e => {
      e.stopPropagation();
      openZone(label, cards(list, actions));
    };
  }
  return p;
}

const MANA = ["W", "U", "B", "R", "G", "C"];

function info(p, key, actions) {
  const life = div("life", text("number", p.life), text("label", key === "you" ? "You" : "Opponent"));
  actions.decorate(life, key);
  const lines = [];
  if (p.commander) {
    const c = p.commander;
    lines.push(`${c.name}: cast ${c.casts} time${c.casts === 1 ? "" : "s"}`);
    if (c.damage_taken) lines.push(`${c.damage_taken} commander damage taken`);
  }
  if (p.mulligans) lines.push(`${p.mulligans} mulligan${p.mulligans === 1 ? "" : "s"}`);
  const pool = p.mana_pool.flatMap((n, i) => Array(n).fill(MANA[i])).join(" ");
  if (pool) lines.push(`Mana: ${pool}`);
  const piles = div("piles",
    pile("Library", [], actions, String(p.library)),
    pile("Graveyard", p.graveyard, actions),
    pile("Exile", p.exile, actions));
  const command = p.command_zone ? div("command", ...cards(p.command_zone, actions), text("label", "Command zone")) : "";
  return div("info" + (p.active ? " active" : ""), life, ...lines.map(l => text("line", l)), piles, command);
}

// One player's side; the opponent's mirrors yours, with its lands farthest from the middle.
function area(box, p, key, actions, revealedHand) {
  const lands = p.battlefield.filter(c => c.land && c.power == null);
  const others = p.battlefield.filter(c => !(c.land && c.power == null));
  const hand = div("hand" + (revealedHand ? " revealed" : ""), ...cards(revealedHand || p.hand, actions));
  const field = key === "you"
    ? div("field", div("row", ...cards(others, actions)), div("row lands", ...cards(lands, actions)), hand)
    : div("field", hand, div("row lands", ...cards(lands, actions)), div("row", ...cards(others, actions)));
  box.className = "area " + key;
  box.replaceChildren(info(p, key, actions), field);
}

const words = step => step.replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase();

export function renderTable(board, actions, agentHand) {
  area(document.getElementById("opponent"), board.opponent, "opponent", actions, agentHand);
  area(document.getElementById("you"), board.you, "you", actions);
  const turn = board.mulligans ? "Mulligans" : `Turn ${board.turn}: ${board.active === "you" ? "your" : "the opponent's"} turn`;
  document.getElementById("phase").replaceChildren(text("turn", turn), text("step", words(board.step)));
  const stack = board.stack.map(item => {
    const el = item.ability ? abilityElement(item) : cardElement(item);
    el.classList.add(item.controller);
    actions.decorate(el, item.key);
    return el;
  });
  document.getElementById("stack").replaceChildren(...(stack.length ? [text("label", "Stack (top first)"), ...stack] : []));
}
