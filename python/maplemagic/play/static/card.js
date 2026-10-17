// One card as it sits on the table: its art (its name until the art loads, or
// if there is none), and its state: tapped, attacking, P/T, counters, damage.
import { artUrl, preview } from "./art.js";

export function badge(cls, text) {
  const b = document.createElement("span");
  b.className = "badge " + cls;
  b.textContent = text;
  return b;
}

export function cardElement(c) {
  const el = document.createElement("div");
  el.className = "card";
  if (c.hidden) {
    el.classList.add("back");
    return el;
  }
  el.dataset.key = c.key;
  const name = document.createElement("span");
  name.className = "name";
  name.textContent = c.name;
  const img = document.createElement("img");
  img.alt = c.name;
  img.onerror = () => img.remove();
  img.src = artUrl(c.name);
  el.append(name, img);
  for (const flag of ["tapped", "attacking", "token"]) if (c[flag]) el.classList.add(flag);
  if (c.blocking) el.append(badge("blocking", "blocking"));
  if (c.attacking) el.append(badge("attack", "attacking"));
  if (c.power != null) el.append(badge("pt", `${c.power}/${c.toughness}`));
  if (c.damage) el.append(badge("damage", `${c.damage} damage`));
  if (c.counters) el.append(badge("counters", c.counters.map(([label, n]) => `${n} ${label}`).join(", ")));
  el.addEventListener("mouseenter", () => preview(c.name));
  return el;
}

// An ability on the stack: a small card with its source's art.
export function abilityElement(item) {
  const el = cardElement({ key: item.key, name: item.name });
  el.classList.add("ability");
  el.append(badge("label", "ability"));
  return el;
}
