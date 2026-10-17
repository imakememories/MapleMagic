// The table: each player's side (life, library, graveyard, exile, command
// zone, battlefield, hand) and between them the turn's steps, the stack and
// floating mana. Identical permanents are stacked.
import { abilityElement, badge, cardElement } from "./card.js";
import { openZone } from "./popup.js";
import { STEPS, stepEntries } from "./timeline.js";

export function div(cls, ...children) {
  const d = document.createElement("div");
  d.className = cls;
  d.append(...children);
  return d;
}

export function text(cls, t) {
  const d = div(cls);
  d.textContent = t;
  return d;
}

// `deco`: { decorate(el, card), has(key), mark(key) }; mark is the card's look in a combat selection.
const cards = (list, deco) => list.map(c => {
  const el = cardElement(c);
  if (c.key) deco.decorate(el, c);
  return el;
});

// Permanents in groups that look and act alike, each group stacked with its count.
function stacked(list, deco) {
  const groups = new Map();
  for (const c of list) {
    const sig = JSON.stringify([c.name, c.tapped, c.power, c.toughness, c.counters, c.damage, c.attacking, c.blocking,
      c.token, deco.has(c.key), deco.mark(c.key)]);
    groups.set(sig, [...(groups.get(sig) || []), c]);
  }
  return [...groups.values()].map(g => {
    if (g.length === 1) return cards(g, deco)[0];
    const els = cards(g, deco);
    els.forEach((el, i) => el.style.setProperty("--i", i));
    const box = div("group", ...els, text("count-badge", "×" + g.length));
    box.style.setProperty("--n", g.length);
    return box;
  });
}

// A face-down library, or a graveyard or exile showing its top card; a click lays the pile out.
function pile(label, list, deco, faceDown) {
  const empty = faceDown ? faceDown === "0" : !list.length;
  const top = faceDown || empty ? div("card back" + (empty ? " empty" : "")) : cardElement(list[0]);
  const p = div("pile", top, text("count", `${label} ${faceDown ? faceDown : list.length}`));
  if (!faceDown && list.length) {
    p.classList.add("open");
    if (list.some(c => c.key && deco.has(c.key))) top.classList.add("can");
    p.onclick = e => {
      e.stopPropagation();
      openZone(label, cards(list, deco));
    };
  }
  return p;
}

export const MANA = ["W", "U", "B", "R", "G", "C"];

// Mana symbols for a pool [W, U, B, R, G, C].
export function manaSymbols(pool) {
  return pool.flatMap((n, i) => Array(n).fill(MANA[i])).map(m => text("mana " + m, m));
}

function info(p, key, deco) {
  const life = div("life", text("number", p.life), text("label", key === "you" ? "You" : "Opponent"));
  deco.decorate(life, { key });
  const lines = [];
  if (p.commander) {
    const c = p.commander;
    lines.push(text("line", `${c.name}: cast ${c.casts} time${c.casts === 1 ? "" : "s"}` + (c.tax ? `, tax {${c.tax}}` : "")));
    if (c.damage_taken) lines.push(text("line", `${c.damage_taken} commander damage taken`));
  }
  if (p.mulligans) lines.push(text("line", `${p.mulligans} mulligan${p.mulligans === 1 ? "" : "s"}`));
  if (p.mana_pool.some(n => n)) lines.push(div("line pool", text("", "Floating:"), ...manaSymbols(p.mana_pool)));
  const piles = div("piles",
    pile("Library", [], deco, String(p.library)),
    pile("Graveyard", p.graveyard, deco),
    pile("Exile", p.exile, deco));
  let command = "";
  if (p.command_zone) {
    const els = cards(p.command_zone, deco);
    // The commander tax on the card itself (903.8).
    if (p.commander && p.commander.tax) els.forEach(el => el.append(badge("tax", `+{${p.commander.tax}}`)));
    command = div("command", ...els, text("label", "Command zone"));
  }
  return div("info" + (p.active ? " active" : ""), life, ...lines, piles, command);
}

// One player's side; the opponent's mirrors yours, with its lands farthest from the middle.
function area(box, p, key, deco, revealedHand) {
  const lands = p.battlefield.filter(c => c.land && c.power == null);
  const others = p.battlefield.filter(c => !(c.land && c.power == null));
  const hand = div("hand" + (revealedHand ? " revealed" : ""), ...cards(revealedHand || p.hand, deco));
  const field = key === "you"
    ? div("field", div("row", ...stacked(others, deco)), div("row lands", ...stacked(lands, deco)), hand)
    : div("field", hand, div("row lands", ...stacked(lands, deco)), div("row", ...stacked(others, deco)));
  box.className = "area " + key;
  box.replaceChildren(info(p, key, deco), field);
}

// This turn's steps: where the game is, what happened in each step so far, and where you stop.
function steps(board, s, stops) {
  const at = STEPS.findIndex(([k]) => k === board.step);
  const done = stepEntries(s, board.turn, board.active);
  return div("steps", ...STEPS.map(([k, label], i) => {
    const pill = text("pill", label);
    if (i === at) pill.classList.add("now");
    else if (at >= 0 && i < at || board.step === "Cleanup") pill.classList.add("past");
    const entries = done.get(k) || [];
    if (entries.length) {
      pill.classList.add("happened");
      pill.title = entries.map(e => e.text).join("\n");
    } else if (i < at) pill.title = "Nothing happened.";
    if (stops && stops[board.active].includes(k)) {
      pill.classList.add("stop");
      pill.title = (pill.title ? pill.title + "\n" : "") + "You stop here (see Auto-pass stops).";
    }
    return pill;
  }));
}

const words = step => step.replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase();

export function renderTable(s, deco, stops) {
  const board = s.board;
  area(document.getElementById("opponent-area"), board.opponent, "opponent", deco, s.debug && s.debug.agent_hand);
  area(document.getElementById("you-area"), board.you, "you", deco);
  const turn = board.mulligans ? "Mulligans" : `Turn ${board.turn}: ${board.active === "you" ? "your" : "the opponent's"} turn`;
  const phase = [text("turn", turn)];
  if (!board.mulligans) phase.push(text("step", words(board.step)), steps(board, s, stops));
  document.getElementById("phase").replaceChildren(...phase);
  const stack = board.stack.map(item => {
    const el = item.ability ? abilityElement(item) : cardElement(item);
    el.classList.add(item.controller);
    deco.decorate(el, item);
    return el;
  });
  const pool = board.you.mana_pool.some(n => n)
    ? [div("floating", text("label", "Your floating mana (empties at the end of the step)"), div("symbols", ...manaSymbols(board.you.mana_pool)))]
    : [];
  document.getElementById("stack").replaceChildren(...pool,
    ...(stack.length ? [text("label", "Stack (top first)"), ...stack] : []));
}
