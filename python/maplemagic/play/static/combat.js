// Declaring attackers and blockers by clicking: creatures to attack with are
// toggled, then sent as one declaration; a blocker is clicked, then the
// attacker it blocks, with a line between them.
import { button } from "./actions.js";
import { badge } from "./card.js";

// The selection of the declaration on screen, kept across redraws.
let sel = { id: "", picked: new Set(), assign: new Map(), blocker: null };

function reset(id) {
  if (sel.id !== id) sel = { id, picked: new Set(), assign: new Map(), blocker: null };
}

function byKey(board) {
  return new Map([...board.you.battlefield, ...board.opponent.battlefield].map(c => [c.key, c]));
}

// actions: the usual choices, for everything the selection doesn't cover; send(path, body, label, el); redraw().
export function combatMode(s, actions, send, redraw) {
  if (!s.board || !s.choices || !s.choices.length) return null;
  if (s.board.attack) return attackMode(s, actions, send, redraw);
  if (s.board.block) return blockMode(s, actions, send, redraw);
  return null;
}

function attackMode(s, actions, send, redraw) {
  const plan = s.board.attack, cards = byKey(s.board);
  reset("attack " + plan.asking.join());
  const asking = new Set(plan.asking), locked = new Set(plan.chosen);
  const name = k => cards.get(k)?.name || k;
  const power = [...sel.picked, ...locked].reduce((t, k) => t + (cards.get(k)?.power || 0), 0);
  const n = sel.picked.size + locked.size;
  const confirm = button(n ? `Attack with ${n} (power ${power})` : "No attack", el => send("/api/attack",
    { attackers: [...sel.picked] }, n ? "Attack with " + [...locked, ...sel.picked].map(name).join(", ") : "No attack", el), "go");
  const all = button("Select all", () => { plan.asking.forEach(k => sel.picked.add(k)); redraw(); }, "secondary");
  const none = button("Clear", () => { sel.picked.clear(); redraw(); }, "secondary");
  return {
    prompt: "Declare attackers",
    hint: "Click your creatures to choose which attack, then confirm." + (locked.size ? " Highlighted in red with a lock must attack." : ""),
    buttons: [confirm, all, none],
    has: k => asking.has(k) || actions.has(k),
    mark: k => sel.picked.has(k) || locked.has(k) ? "attack" : "",
    decorate(el, item) {
      if (locked.has(item.key)) {
        el.classList.add("picked-attack");
        el.append(badge("attack", "must attack"));
        return;
      }
      if (!asking.has(item.key)) return actions.decorate(el, item);
      el.classList.add("can");
      el.title = sel.picked.has(item.key) ? "Click: don't attack" : "Click: attack";
      if (sel.picked.has(item.key)) {
        el.classList.add("picked-attack");
        el.append(badge("attack", "attacking"));
      }
      el.onclick = e => {
        e.stopPropagation();
        if (sel.picked.has(item.key)) sel.picked.delete(item.key);
        else sel.picked.add(item.key);
        redraw();
      };
    },
    lines: [],
  };
}

function blockMode(s, actions, send, redraw) {
  const plan = s.board.block, cards = byKey(s.board);
  reset("block " + plan.asking.map(b => b.key).join());
  const can = new Map(plan.asking.map(b => [b.key, new Set(b.attackers)]));
  const name = k => cards.get(k)?.name || k;
  const pairs = [...plan.pairs, ...sel.assign];
  const attackers = s.board.opponent.battlefield.filter(c => c.attacking);
  const blocked = new Map();
  for (const [b, a] of pairs) blocked.set(a, (blocked.get(a) || 0) + 1);
  const unblocked = attackers.filter(c => !blocked.has(c.key)).reduce((t, c) => t + (c.power || 0), 0);
  const n = sel.assign.size;
  const confirm = button(n ? `Confirm ${n} block${n === 1 ? "" : "s"}` : "No blocks", el => send("/api/block",
    { blocks: [...sel.assign] }, n ? [...sel.assign].map(([b, a]) => `Block ${name(a)} with ${name(b)}`).join("; ") : "No blocks", el), "go");
  const clear = button("Clear", () => { sel.assign.clear(); sel.blocker = null; redraw(); }, "secondary");
  const picked = sel.blocker;
  return {
    prompt: "Declare blockers",
    hint: (picked ? `Now click the attacker ${name(picked)} should block, or click it again to cancel.`
      : "Click one of your creatures, then the attacker it should block. Click a blocker again to change it.") +
      ` Unblocked, the attackers would deal ${unblocked} damage (before abilities like trample).`,
    buttons: [confirm, clear],
    has: k => can.has(k) || blocked.has(k) || actions.has(k) || attackers.some(c => c.key === k),
    mark: k => (k === picked ? "picked " : "") + (sel.assign.get(k) || "") + (picked && can.get(picked).has(k) ? " target" : ""),
    decorate(el, item) {
      const k = item.key;
      if (can.has(k)) {
        el.classList.add("can");
        if (k === picked) el.classList.add("picked-blocker");
        if (sel.assign.has(k)) el.append(badge("blocking", "blocks " + name(sel.assign.get(k))));
        el.onclick = e => {
          e.stopPropagation();
          if (sel.blocker === k) sel.blocker = null;
          else {
            sel.assign.delete(k);
            sel.blocker = k;
          }
          redraw();
        };
        return;
      }
      if (attackers.some(c => c.key === k)) {
        if (blocked.get(k)) el.append(badge("blocked", `blocked by ${blocked.get(k)}`));
        if (picked && can.get(picked).has(k)) {
          el.classList.add("block-target");
          el.title = `Block ${name(k)} with ${name(picked)}`;
          el.onclick = e => {
            e.stopPropagation();
            sel.assign.set(picked, k);
            sel.blocker = null;
            redraw();
          };
        }
        return;
      }
      actions.decorate(el, item);
    },
    lines: pairs,
  };
}

// Lines from each blocker to the attacker it blocks, over the page.
export function drawLines(pairs) {
  const svg = document.getElementById("lines");
  const centre = k => {
    const el = document.querySelector(`#table .card[data-key="${k}"]`);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return [r.left + r.width / 2, r.top + r.height / 2];
  };
  svg.replaceChildren(...pairs.map(([b, a]) => [centre(b), centre(a)]).filter(([p, q]) => p && q).map(([p, q]) => {
    const l = document.createElementNS("http://www.w3.org/2000/svg", "line");
    l.setAttribute("x1", p[0]);
    l.setAttribute("y1", p[1]);
    l.setAttribute("x2", q[0]);
    l.setAttribute("y2", q[1]);
    return l;
  }));
}
