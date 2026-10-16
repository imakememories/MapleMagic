// Card art: where a card's image is, and the big copy of the card under the pointer.

export const artUrl = name => "/art/" + encodeURIComponent(name);

export function preview(name) {
  const img = document.getElementById("preview");
  img.onerror = () => { img.hidden = true; };
  img.src = artUrl(name);
  img.alt = name;
  img.hidden = false;
}
