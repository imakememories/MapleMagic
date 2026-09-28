//! Build-time validated runtime deck catalog.
//!
//! `build.rs` reads the checked-in `data/runtime_decks_v1.json`, resolves
//! every materialized card copy against the stable `cards_v1.json` array,
//! verifies the frozen deck hash, and emits the static definitions included
//! below. Ordinary release execution uses only those generated definitions;
//! the explicitly feature-gated Store V2 production capture guard reopens the
//! raw catalog no-follow solely to prove its exact digest against them.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeDeckDefinition {
    pub canonical_pool_order: u32,
    pub id: &'static str,
    pub source_path: &'static str,
    pub source_sha256: &'static str,
    pub mainboard_count: usize,
    pub runtime_deck_hash: u64,
    pub card_ids: &'static [u16],
}

impl RuntimeDeckDefinition {
    pub fn is_fully_materialized(self) -> bool {
        self.mainboard_count == self.card_ids.len()
    }
}

include!(concat!(env!("OUT_DIR"), "/runtime_decks.rs"));

pub fn runtime_deck_by_id(id: &str) -> Option<&'static RuntimeDeckDefinition> {
    RUNTIME_DECKS.iter().find(|deck| deck.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_def::preflight_fully_supported_deck;

    #[test]
    fn generated_runtime_catalog_is_exact_and_fully_supported() {
        assert_eq!(RUNTIME_DECK_CATALOG_SCHEMA, "kernel_runtime_decks/v1");
        assert_eq!(RUNTIME_DECK_PROTOCOL, "canonical-mainboard-bo1/v1");
        assert_eq!(
            RUNTIME_DECK_CATALOG_FILE_SHA256,
            "68e7602f3a4df6217119406973954630800c358a10fca9f28e6cf9f20fd3b851"
        );
        assert_eq!(
            RUNTIME_DECK_MATERIALIZATION_PROTOCOL,
            "xmage_xml_row_then_copy_ordinal/v1"
        );
        assert_eq!(
            RUNTIME_DECK_HASH_ALGORITHM,
            "fnv1a64-serde-json-u16-array/v1"
        );
        assert_eq!(
            RUNTIME_DECKS.iter().map(|deck| deck.id).collect::<Vec<_>>(),
            [
                "Wildfire", "Rally", "Affinity", "Elves", "Spy", "Burn", "Terror", "CawGates",
                "Faeries",
            ]
        );

        for deck in RUNTIME_DECKS {
            assert_eq!(deck.mainboard_count, 60, "{}", deck.id);
            assert!(deck.is_fully_materialized(), "{}", deck.id);
            assert_eq!(deck.source_sha256.len(), 64, "{}", deck.id);
            assert_ne!(deck.runtime_deck_hash, 0, "{}", deck.id);
            preflight_fully_supported_deck(deck.card_ids).unwrap();
        }
    }

    #[test]
    fn lookup_is_exact_case_and_hashes_are_frozen() {
        for (deck_id, expected_hash) in [
            ("Wildfire", 0x552a_cb5f_c963_1d3b),
            ("Rally", 0x0c9f_01c2_5444_12bf),
            ("Affinity", 0xff4b_f00d_eadf_8821),
            ("Elves", 0x6a18_7257_d6d3_7346),
            ("Spy", 0xcd2a_fdfe_e357_3675),
            ("Burn", 0x5fdb_7b92_986b_6fc1),
            ("Terror", 0xfd6f_1a4a_ceaa_157b),
            ("CawGates", 0x25c1_916a_4d20_c08e),
            ("Faeries", 0xd7a4_7ab2_fa78_dbaa),
        ] {
            assert_eq!(
                runtime_deck_by_id(deck_id).unwrap().runtime_deck_hash,
                expected_hash,
                "{deck_id}"
            );
        }
        for unsupported in [
            "wildfire", "WILDFIRE", "burn", "BURN", "rally", "RALLY", "terror", "TERROR", "",
        ] {
            assert!(runtime_deck_by_id(unsupported).is_none(), "{unsupported:?}");
        }
    }
}
