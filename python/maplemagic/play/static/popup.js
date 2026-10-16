// Popups over the table: a menu of one card's choices, and the cards of a
// pile (graveyard, exile) laid out to look at. One at a time; a click outside closes it.

const box = () => document.getElementById("popup");

export function closePopup() {
  const b = box();
  b.hidden = true;
  b.replaceChildren();
}

// At (x, y), kept on screen; x null centers it.
function place(b, x, y) {
  b.hidden = false;
  const w = b.offsetWidth, h = b.offsetHeight;
  if (x == null) x = (innerWidth - w) / 2;
  b.style.left = Math.max(4, Math.min(x, innerWidth - w - 4)) + "px";
  b.style.top = Math.max(4, Math.min(y, innerHeight - h - 4)) + "px";
}

// items: [{ text, run }]
export function openMenu(x, y, items) {
  const b = box();
  b.className = "menu";
  b.replaceChildren(...items.map(it => {
    const btn = document.createElement("button");
    btn.textContent = it.text;
    btn.onclick = e => {
      e.stopPropagation();
      closePopup();
      it.run();
    };
    return btn;
  }));
  place(b, x, y);
}

export function openZone(title, cards) {
  const b = box();
  b.className = "zone";
  const head = document.createElement("h4");
  head.textContent = title;
  const row = document.createElement("div");
  row.className = "row";
  row.append(...cards);
  b.replaceChildren(head, row);
  place(b, null, 80);
}

document.addEventListener("click", e => {
  if (!box().hidden && !box().contains(e.target)) closePopup();
});
