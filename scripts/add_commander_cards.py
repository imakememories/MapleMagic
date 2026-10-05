"""Append to data/cards_v1.json every card in configs/commander/*.txt it lacks,
with characteristics from Scryfall and no behaviour (engine_capability
"no_effect"), so card ids stay fixed while behaviour is added card by card.

usage: python scripts/add_commander_cards.py [xmage checkout]
"""
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "data", "cards_v1.json")
KEYWORDS = {"Flying", "Reach", "Haste", "Vigilance", "Trample", "First strike", "Double strike", "Deathtouch",
            "Menace", "Defender", "Lifelink", "Hexproof", "Indestructible", "Flash"}
SUPERTYPES = {"Legendary", "Basic", "Snow", "World"}
MULTI_FACE = {"modal_dfc", "transform", "adventure", "split", "flip"}


def deck_cards():
    """{card name: [deck labels]} over every Commander list, commanders included."""
    out = {}
    folder = os.path.join(ROOT, "configs", "commander")
    for fname in sorted(os.listdir(folder)):
        label = "Commander - " + fname[:-4].capitalize()
        for line in open(os.path.join(folder, fname), encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            name = line.split(":", 1)[1].strip() if line.startswith("commander:") else line.split(" ", 1)[1]
            out.setdefault(name, []).append(label)
    return out


def fetch(names):
    cards = {}
    for i in range(0, len(names), 75):
        body = json.dumps({"identifiers": [{"name": n.split(" // ")[0]} for n in names[i:i + 75]]}).encode()
        req = urllib.request.Request("https://api.scryfall.com/cards/collection", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Accept": "application/json",
                                              "User-Agent": "MapleMagic/0.1"})
        data = json.load(urllib.request.urlopen(req))
        if data.get("not_found"):
            sys.exit(f"not on Scryfall: {data['not_found']}")
        for c in data["data"]:
            cards[c["name"].split(" // ")[0]] = c
        time.sleep(0.15)
    return cards


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def java_files(xmage):
    root = os.path.join(xmage, "Mage.Sets", "src", "mage", "cards")
    out = {}
    if os.path.isdir(root):
        for d in os.listdir(root):
            p = os.path.join(root, d)
            if os.path.isdir(p):
                for f in os.listdir(p):
                    if f.endswith(".java"):
                        out[norm(f[:-5])] = f"Mage.Sets/src/mage/cards/{d}/{f}"
    return out


def number(v):
    return int(v) if v is not None and re.fullmatch(r"-?\d+", v) else None


def entry(card, decks, java):
    face = card["card_faces"][0] if card["layout"] in MULTI_FACE else card
    left, _, right = face["type_line"].partition("\u2014")
    words = left.split()
    supertypes = [w for w in words if w in SUPERTYPES]
    types = [w for w in words if w not in SUPERTYPES]
    oracle = card.get("oracle_text") or " // ".join(f.get("oracle_text", "") for f in card.get("card_faces", []))
    return {
        "name": face["name"],
        "engine_capability": "no_effect",
        "mana_cost": face.get("mana_cost", ""),
        "mana_value": int(card.get("cmc", 0)),
        "colors": face.get("colors", card.get("colors", [])),
        "types": types,
        "subtypes": right.split(),
        "supertypes": supertypes,
        "power": number(face.get("power")),
        "toughness": number(face.get("toughness")),
        "is_land": "Land" in types,
        "produces_mana": [],
        "decks": decks,
        "sideboard_only": False,
        "complexity": "commander",
        "mechanics": [],
        "keywords": sorted(k.lower() for k in card.get("keywords", []) if k in KEYWORDS),
        "oracle": oracle,
        "layout": card["layout"],
        "java_file": java.get(norm(face["name"]), ""),
    }


def main():
    xmage = sys.argv[1] if len(sys.argv) > 1 else r"C:\git\mageFork"
    reg = json.load(open(REGISTRY, encoding="utf-8"))
    have = {c["name"] for c in reg["cards"]}
    wanted = deck_cards()
    missing = sorted(n for n in wanted if n not in have and n.split(" // ")[0] not in have)
    if not missing:
        print("nothing to add")
        return
    cards = fetch(missing)
    java = java_files(xmage)
    added = [entry(cards[n.split(" // ")[0]], wanted[n], java) for n in missing]
    reg["cards"].extend(added)
    with open(REGISTRY, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(reg, indent=2) + "\n")
    subtypes = sorted({s for c in added for s in c["subtypes"]})
    types = sorted({t for c in added for t in c["types"]})
    print(f"added {len(added)} cards; types {types}")
    print("subtypes:", subtypes)
    print("no XMage file:", [c["name"] for c in added if not c["java_file"]])


if __name__ == "__main__":
    main()
