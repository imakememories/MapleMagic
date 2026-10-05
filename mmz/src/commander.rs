//! Commander decks: the four lists in `configs/commander`, resolved against
//! the engine's card pool.

use mtg_kernel::card_def::{card_id_by_name, CARD_DEFS};

const LISTS: [(&str, &str); 4] = [
    ("Ognis", include_str!("../../configs/commander/ognis.txt")),
    ("Thranduil", include_str!("../../configs/commander/thranduil.txt")),
    ("Radha", include_str!("../../configs/commander/radha.txt")),
    ("Krenko", include_str!("../../configs/commander/krenko.txt")),
];

/// A commander and the other 99 cards, as card definitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommanderDeck {
    pub name: String,
    pub commander: u16,
    pub cards: Vec<u16>,
}

pub fn list_names() -> Vec<&'static str> {
    LISTS.iter().map(|(n, _)| *n).collect()
}

/// (commander, [(count, card)]) from a list: a `commander: <name>` line, then `<count> <name>` lines.
pub fn parse_list(text: &str) -> Result<(String, Vec<(u32, String)>), String> {
    let mut commander = None;
    let mut cards = Vec::new();
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if let Some(name) = line.strip_prefix("commander:") {
            commander = Some(name.trim().to_string());
            continue;
        }
        let (count, name) = line.split_once(' ').ok_or_else(|| format!("bad line {line:?}"))?;
        let count: u32 = count.parse().map_err(|_| format!("bad count in {line:?}"))?;
        cards.push((count, name.trim().to_string()));
    }
    Ok((commander.ok_or("no commander line")?, cards))
}

/// The pool's id for `name`, trying the front face of a double-faced card.
fn resolve(name: &str) -> Option<u16> {
    card_id_by_name(name).or_else(|| card_id_by_name(name.split(" // ").next()?.trim()))
}

fn supported(id: u16) -> bool {
    CARD_DEFS[id as usize].capability.is_fully_supported()
}

impl CommanderDeck {
    /// The named list, or the cards the engine can't play yet.
    pub fn named(name: &str) -> Result<CommanderDeck, Vec<String>> {
        Self::named_with_stand_in(name, None)
    }

    /// [`Self::named`], swapping each unsupported card for `stand_in` (for engine tests).
    pub fn named_with_stand_in(name: &str, stand_in: Option<&str>) -> Result<CommanderDeck, Vec<String>> {
        let text = LISTS.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).ok_or_else(|| vec![format!("unknown deck {name}")])?.1;
        let (commander, list) = parse_list(text).map_err(|e| vec![e])?;
        let stand_in = stand_in.map(|s| resolve(s).filter(|&id| supported(id)).expect("supported stand-in"));
        let mut missing = Vec::new();
        let mut pick = |n: &str| match resolve(n).filter(|&id| supported(id)) {
            Some(id) => Some(id),
            None => {
                missing.push(n.to_string());
                stand_in
            }
        };
        let cmd = pick(&commander);
        let mut cards = Vec::new();
        for (count, n) in &list {
            if let Some(id) = pick(n) {
                cards.extend(std::iter::repeat_n(id, *count as usize));
            }
        }
        if !missing.is_empty() && stand_in.is_none() {
            missing.sort();
            missing.dedup();
            return Err(missing);
        }
        Ok(CommanderDeck { name: name.to_string(), commander: cmd.expect("commander"), cards })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_list_is_a_hundred_cards() {
        for (name, text) in LISTS {
            let (commander, cards) = parse_list(text).unwrap();
            let total: u32 = cards.iter().map(|c| c.0).sum();
            assert_eq!(total, 99, "{name}");
            assert!(!cards.iter().any(|c| c.1 == commander), "{name}: commander listed twice");
        }
    }

    #[test]
    fn unsupported_cards_are_reported_or_stood_in() {
        let err = CommanderDeck::named("Ognis").err().unwrap_or_default();
        let deck = CommanderDeck::named_with_stand_in("Ognis", Some("Mountain")).unwrap();
        assert_eq!(deck.cards.len(), 99);
        if !err.is_empty() {
            let mountain = card_id_by_name("Mountain").unwrap();
            assert!(deck.cards.iter().filter(|&&c| c == mountain).count() >= 27);
        }
        assert!(CommanderDeck::named("Nobody").is_err());
    }
}
