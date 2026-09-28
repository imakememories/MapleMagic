//! Environment randomization protocol v2: SHA-256 domain-separated
//! per-owner, per-purpose shuffle substreams.
//!
//! Frozen contract identity `mtg-kernel-environment-randomization-sha256-v2`
//! (collab, 2026-07-29). The pair root is the exact unsigned 64-bit
//! `pair_environment_seed`; each substream seed is the first eight SHA-256
//! digest bytes big-endian with no u63 mask, over length-framed atoms
//! (`u32be(tag_len) || tag || u64be(payload_len) || payload`) in the exact
//! order version, namespace, pair_environment_seed, physical_owner, purpose,
//! ordinal. A fresh local `SplitMix64` consumes the derived seed with the
//! existing descending Fisher-Yates modulo algorithm. The legacy single
//! raw-seeded stream is untouched by this module; the trainer schedule and
//! its pair-environment-seed derivation are byte-for-byte unchanged.
//!
//! Vocabulary is closed: owners `p0`/`p1`, purposes
//! `initial-library-shuffle` (ordinal zero only) and
//! `in-game-library-shuffle`. Ordinal `u64::MAX` is rejected before
//! derivation because the committed ordinal could not advance afterward;
//! roots may equal `u64::MAX`.
//!
//! Cross-language goldens: `data/environment_randomization_v2/goldens_v1.json`
//! is generated and independently verified by the stdlib-only reference
//! `python/tools/environment_randomization_v2_reference.py`; the tests below
//! assert the same vectors from this implementation.

use sha2::{Digest, Sha256};

use crate::state::SplitMix64;

pub const ENVIRONMENT_RANDOMIZATION_IDENTITY_V2: &str =
    "mtg-kernel-environment-randomization-sha256-v2";
pub const ENVIRONMENT_RANDOMIZATION_NAMESPACE_V2: &str = "environment-randomization-substream";
/// Sealed golden authority: schema identity and exact raw-byte SHA-256 of the
/// embedded cross-language golden artifact. Run-record validation binds these
/// exact constants; the artifact is compiled in via include_str! and its byte
/// hash is verified before any parse.
pub const ENVIRONMENT_RANDOMIZATION_GOLDENS_SCHEMA_V1: &str =
    "mtg-kernel-environment-randomization-v2-goldens/v1";
pub const ENVIRONMENT_RANDOMIZATION_GOLDENS_SHA256_V1: &str =
    "bc2b0d66f8e3eb608b6035321f23a214bbf5141aaf7305f50f606f6c85b4a3bc";
pub const ENVIRONMENT_RANDOMIZATION_GOLDENS_V1: &str =
    include_str!("../../data/environment_randomization_v2/goldens_v1.json");

/// Sealed pre-change legacy bytes (environment-v2 step 2, gate 1): the exact
/// legacy GameState JSON, diagnostic envelope bytes, and frozen hash captured
/// before any representation work.
pub const LEGACY_STATE_BYTES_SHA256_V1: &str =
    "3e0f044d0cdbfa356b8ccc3ef2c7a198078be0e6a119e8e428664a3a0347a349";
pub const LEGACY_STATE_BYTES_V1: &str =
    include_str!("../../data/environment_randomization_v2/legacy_state_bytes_v1.json");

/// Sealed portable reset physical-trajectory golden authority. The portable
/// reset boundary is the state immediately after the fourteen alternating
/// opening draws and before session policy advance; Python owns only the
/// physical card-definition projection of that boundary. The committed V6
/// tests already freeze the post-advance Decision, and native full-episode
/// trajectory folding remains exactly
/// `mtg-kernel-native-full-episode-trajectory-sha256-v1`.
///
/// The artifact is generated and independently verified by the stdlib-only
/// `python/tools/generate_environment_randomization_v2_reset_physical_trajectory_goldens_v1.py`.
/// Both the raw-file SHA-256 and the separately pinned portable
/// vector-stream SHA-256 are minted from that generator's final canonical
/// bytes; the test-only module reconstructs the stream from typed artifact
/// values and reads no Python-precomputed semantic stream.
pub const ENVIRONMENT_RANDOMIZATION_RESET_TRAJECTORY_GOLDENS_SCHEMA_V1: &str =
    "mtg-kernel-environment-randomization-v2-reset-physical-trajectory-goldens/v1";
pub const ENVIRONMENT_RANDOMIZATION_RESET_TRAJECTORY_GENERATOR_IDENTITY_V1: &str =
    "mtg-kernel-environment-randomization-v2-reset-physical-trajectory-goldens-stdlib-python-v1";
pub const ENVIRONMENT_RANDOMIZATION_RESET_TRAJECTORY_PHYSICAL_PROJECTION_IDENTITY_V1: &str =
    "mtg-kernel-environment-randomization-v2-physical-card-definition-projection/v1";
pub const ENVIRONMENT_RANDOMIZATION_RESET_TRAJECTORY_PORTABLE_VECTOR_STREAM_IDENTITY_V1: &str =
    "mtg-kernel-environment-randomization-v2-reset-physical-trajectory-portable-vector-stream-sha256-v1";
pub const ENVIRONMENT_RANDOMIZATION_RESET_TRAJECTORY_GOLDENS_SHA256_V1: &str =
    "18ec6cd138a76bce1bf06c6b794fe169fbe8d83c0a9265d0ff99119a4c4a16bc";
pub const ENVIRONMENT_RANDOMIZATION_RESET_TRAJECTORY_PORTABLE_VECTOR_STREAM_SHA256_V1: &str =
    "97f8eeff002ec15f3e30f58fd1f1e477a8abf1db3a38e25aaeb810f87da2a085";
pub const ENVIRONMENT_RANDOMIZATION_RESET_TRAJECTORY_GOLDENS_V1: &str = include_str!(
    "../../data/environment_randomization_v2/reset_physical_trajectory_goldens_v1.json"
);

/// Production owner constants for the frozen v2 derivation descriptor.
///
/// These were previously literals inside this module's golden test only. They
/// are promoted here because the run-record manifest section
/// (`native_training_store_run_v2::EnvironmentRandomizationContractV2`) is
/// validated against *projections of these constants*, never against a second
/// restatement of the same strings. The descriptor test below now also projects
/// from them, so the golden artifact, the manifest validator, and the
/// implementation vocabulary cannot drift apart pairwise.
pub const ENVIRONMENT_RANDOMIZATION_ATOM_FRAMING_V2: &str =
    "u32be(tag_byte_len) || utf8(tag) || u64be(payload_byte_len) || payload";
pub const ENVIRONMENT_RANDOMIZATION_EXTRACTION_V2: &str =
    "first-8-sha256-digest-bytes-big-endian-no-mask";
/// The closed owner vocabulary, in the frozen order. `PhysicalOwnerV2`
/// projects from this array rather than restating the strings.
pub const ENVIRONMENT_RANDOMIZATION_OWNERS_V2: [&str; 2] = ["p0", "p1"];
/// The closed purpose vocabulary, in the frozen order. `ShufflePurposeV2`
/// projects from this array.
pub const ENVIRONMENT_RANDOMIZATION_PURPOSES_V2: [&str; 2] =
    ["initial-library-shuffle", "in-game-library-shuffle"];
pub const ENVIRONMENT_RANDOMIZATION_INITIAL_ORDINAL_RULE_V2: &str =
    "initial-library-shuffle-requires-ordinal-zero";
pub const ENVIRONMENT_RANDOMIZATION_OVERFLOW_RULE_V2: &str =
    "ordinal-u64-max-rejected-before-derivation-roots-may-equal-u64-max";
pub const ENVIRONMENT_RANDOMIZATION_SHUFFLE_ALGORITHM_V2: &str =
    "splitmix64-descending-fisher-yates-modulo";
/// The six ordered atoms of the derivation preimage, in the exact frozen
/// order. Row arities are 2, 2, 4, 4, 4, 4; the first two rows carry the
/// identity and namespace owner constants themselves.
pub const ENVIRONMENT_RANDOMIZATION_ORDERED_ATOMS_V2: [&[&str]; 6] = [
    &["version", ENVIRONMENT_RANDOMIZATION_IDENTITY_V2],
    &["namespace", ENVIRONMENT_RANDOMIZATION_NAMESPACE_V2],
    &[
        "field-name",
        "pair_environment_seed",
        "u64",
        "8-byte-big-endian",
    ],
    &["field-name", "physical_owner", "str", "utf-8"],
    &["field-name", "purpose", "str", "utf-8"],
    &["field-name", "ordinal", "u64", "8-byte-big-endian"],
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalOwnerV2 {
    P0,
    P1,
}

impl PhysicalOwnerV2 {
    pub const fn as_str(self) -> &'static str {
        match self {
            PhysicalOwnerV2::P0 => ENVIRONMENT_RANDOMIZATION_OWNERS_V2[0],
            PhysicalOwnerV2::P1 => ENVIRONMENT_RANDOMIZATION_OWNERS_V2[1],
        }
    }

    pub fn parse(value: &str) -> Result<Self, EnvironmentRandomizationErrorV2> {
        if value == ENVIRONMENT_RANDOMIZATION_OWNERS_V2[0] {
            Ok(PhysicalOwnerV2::P0)
        } else if value == ENVIRONMENT_RANDOMIZATION_OWNERS_V2[1] {
            Ok(PhysicalOwnerV2::P1)
        } else {
            Err(EnvironmentRandomizationErrorV2::InvalidOwner)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShufflePurposeV2 {
    InitialLibraryShuffle,
    InGameLibraryShuffle,
}

impl ShufflePurposeV2 {
    pub const fn as_str(self) -> &'static str {
        match self {
            ShufflePurposeV2::InitialLibraryShuffle => ENVIRONMENT_RANDOMIZATION_PURPOSES_V2[0],
            ShufflePurposeV2::InGameLibraryShuffle => ENVIRONMENT_RANDOMIZATION_PURPOSES_V2[1],
        }
    }

    pub fn parse(value: &str) -> Result<Self, EnvironmentRandomizationErrorV2> {
        if value == ENVIRONMENT_RANDOMIZATION_PURPOSES_V2[0] {
            Ok(ShufflePurposeV2::InitialLibraryShuffle)
        } else if value == ENVIRONMENT_RANDOMIZATION_PURPOSES_V2[1] {
            Ok(ShufflePurposeV2::InGameLibraryShuffle)
        } else {
            Err(EnvironmentRandomizationErrorV2::InvalidPurpose)
        }
    }
}

/// The sealed KDF error vocabulary. Game-state library-shuffle transaction
/// failures live in `state::LibraryShuffleError`, which wraps (never
/// extends) this enum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnvironmentRandomizationErrorV2 {
    InvalidOwner,
    InvalidPurpose,
    NonZeroInitialOrdinal,
    OrdinalOverflow,
}

fn append_atom(hasher: &mut Sha256, tag: &str, payload: &[u8]) {
    let tag_len = u32::try_from(tag.len()).expect("atom tag length fits u32");
    let payload_len = u64::try_from(payload.len()).expect("atom payload length fits u64");
    hasher.update(tag_len.to_be_bytes());
    hasher.update(tag.as_bytes());
    hasher.update(payload_len.to_be_bytes());
    hasher.update(payload);
}

/// Derive the substream seed for one (owner, purpose, ordinal) under the
/// pair root. Full-width u64: no u63 mask.
pub fn derive_environment_randomization_seed_v2(
    pair_environment_seed: u64,
    physical_owner: PhysicalOwnerV2,
    purpose: ShufflePurposeV2,
    ordinal: u64,
) -> Result<u64, EnvironmentRandomizationErrorV2> {
    if ordinal == u64::MAX {
        return Err(EnvironmentRandomizationErrorV2::OrdinalOverflow);
    }
    if purpose == ShufflePurposeV2::InitialLibraryShuffle && ordinal != 0 {
        return Err(EnvironmentRandomizationErrorV2::NonZeroInitialOrdinal);
    }
    let mut hasher = Sha256::new();
    append_atom(
        &mut hasher,
        "version",
        ENVIRONMENT_RANDOMIZATION_IDENTITY_V2.as_bytes(),
    );
    append_atom(
        &mut hasher,
        "namespace",
        ENVIRONMENT_RANDOMIZATION_NAMESPACE_V2.as_bytes(),
    );
    append_atom(&mut hasher, "field-name", b"pair_environment_seed");
    append_atom(&mut hasher, "u64", &pair_environment_seed.to_be_bytes());
    append_atom(&mut hasher, "field-name", b"physical_owner");
    append_atom(&mut hasher, "str", physical_owner.as_str().as_bytes());
    append_atom(&mut hasher, "field-name", b"purpose");
    append_atom(&mut hasher, "str", purpose.as_str().as_bytes());
    append_atom(&mut hasher, "field-name", b"ordinal");
    append_atom(&mut hasher, "u64", &ordinal.to_be_bytes());
    let digest = hasher.finalize();
    let mut prefix = [0_u8; 8];
    prefix.copy_from_slice(&digest[..8]);
    Ok(u64::from_be_bytes(prefix))
}

/// Internal generic slice shuffle: fresh local SplitMix64 over the derived
/// seed, existing descending Fisher-Yates modulo algorithm (identical to
/// `rl::shuffled`). Shared by the golden wrapper and the game-state live
/// shuffle transaction so `ObjectId` libraries never narrow through u16.
pub(crate) fn shuffle_slice_in_place_v2<T>(derived_seed: u64, items: &mut [T]) {
    let mut rng = SplitMix64::seed(derived_seed);
    for i in (1..items.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

/// The v2 permutation golden wrapper over the generic slice shuffle.
pub fn permutation_v2(derived_seed: u64, ids: &[u16]) -> Vec<u16> {
    let mut v = ids.to_vec();
    shuffle_slice_in_place_v2(derived_seed, &mut v);
    v
}

/// The v2 game-randomness state: the exact u64 pair root plus per-owner
/// live-shuffle ordinals. No generic RNG exists on the v2 path. Fields are
/// private; mutation happens only through the game-state live-shuffle
/// transaction.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameEnvironmentRandomizationV2 {
    pair_environment_seed: u64,
    /// Frozen compact wire shape: `[p0, p1]`.
    next_live_shuffle_ordinal: [u64; 2],
}

impl GameEnvironmentRandomizationV2 {
    pub fn new(pair_environment_seed: u64) -> Self {
        GameEnvironmentRandomizationV2 {
            pair_environment_seed,
            next_live_shuffle_ordinal: [0, 0],
        }
    }

    pub fn pair_environment_seed(&self) -> u64 {
        self.pair_environment_seed
    }

    pub fn next_live_shuffle_ordinal(&self, owner: PhysicalOwnerV2) -> u64 {
        match owner {
            PhysicalOwnerV2::P0 => self.next_live_shuffle_ordinal[0],
            PhysicalOwnerV2::P1 => self.next_live_shuffle_ordinal[1],
        }
    }

    /// Advances one owner's committed live-shuffle ordinal exactly once.
    /// Callers must have preflighted: the pre-advance value is never
    /// `u64::MAX` because derivation rejects it first.
    /// Writes one owner's committed ordinal to the preflight-checked
    /// successor value. No arithmetic happens here; the successor was
    /// computed and overflow-checked at preflight before derivation.
    pub(crate) fn set_live_shuffle_ordinal(
        &mut self,
        owner: PhysicalOwnerV2,
        checked_successor: u64,
    ) {
        let slot = match owner {
            PhysicalOwnerV2::P0 => &mut self.next_live_shuffle_ordinal[0],
            PhysicalOwnerV2::P1 => &mut self.next_live_shuffle_ordinal[1],
        };
        *slot = checked_successor;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn goldens() -> serde_json::Value {
        let bytes = ENVIRONMENT_RANDOMIZATION_GOLDENS_V1.as_bytes();
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let observed = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            observed, ENVIRONMENT_RANDOMIZATION_GOLDENS_SHA256_V1,
            "embedded golden artifact bytes do not match the sealed SHA-256"
        );
        let doc: serde_json::Value =
            serde_json::from_slice(bytes).expect("decode environment randomization goldens");
        assert_eq!(
            doc["schema"].as_str().expect("golden schema"),
            ENVIRONMENT_RANDOMIZATION_GOLDENS_SCHEMA_V1,
            "golden artifact schema identity mismatch"
        );
        doc
    }

    #[test]
    fn derived_seed_goldens_match_python_reference() {
        let doc = goldens();
        assert_eq!(
            doc["identity"].as_str().expect("identity"),
            ENVIRONMENT_RANDOMIZATION_IDENTITY_V2
        );
        assert_eq!(
            doc["namespace"].as_str().expect("namespace"),
            ENVIRONMENT_RANDOMIZATION_NAMESPACE_V2
        );
        let rows = doc["derived_seeds"].as_array().expect("derived seeds");
        assert_eq!(rows.len(), 40, "golden derived-seed cardinality");
        for row in rows {
            let root = row["pair_environment_seed"].as_u64().expect("root");
            let owner = PhysicalOwnerV2::parse(row["physical_owner"].as_str().expect("owner"))
                .expect("golden owner vocabulary");
            let purpose = ShufflePurposeV2::parse(row["purpose"].as_str().expect("purpose"))
                .expect("golden purpose vocabulary");
            let ordinal = row["ordinal"].as_u64().expect("ordinal");
            let expected = row["derived_seed"].as_u64().expect("derived seed");
            let derived = derive_environment_randomization_seed_v2(root, owner, purpose, ordinal)
                .expect("golden row derives");
            assert_eq!(derived, expected, "derived seed mismatch for {row}");
        }
    }

    #[test]
    fn permutation_goldens_match_python_reference() {
        let doc = goldens();
        let rows = doc["permutations"].as_array().expect("permutations");
        assert_eq!(rows.len(), 200, "golden permutation cardinality");
        for row in rows {
            let seed = row["derived_seed"].as_u64().expect("seed");
            let ids: Vec<u16> = row["input_ids"]
                .as_array()
                .expect("ids")
                .iter()
                .map(|v| u16::try_from(v.as_u64().expect("id")).expect("id fits u16"))
                .collect();
            let expected: Vec<u16> = row["permutation"]
                .as_array()
                .expect("permutation")
                .iter()
                .map(|v| u16::try_from(v.as_u64().expect("id")).expect("id fits u16"))
                .collect();
            assert_eq!(permutation_v2(seed, &ids), expected, "permutation mismatch");
        }
    }

    #[test]
    fn contract_descriptor_is_literal() {
        let doc = goldens();
        let contract = &doc["contract"];
        assert_eq!(
            contract["atom"].as_str().expect("atom"),
            ENVIRONMENT_RANDOMIZATION_ATOM_FRAMING_V2
        );
        assert_eq!(
            contract["extraction"].as_str().expect("extraction"),
            ENVIRONMENT_RANDOMIZATION_EXTRACTION_V2
        );
        assert_eq!(
            contract["shuffle_algorithm"].as_str().expect("shuffle"),
            ENVIRONMENT_RANDOMIZATION_SHUFFLE_ALGORITHM_V2
        );
        assert_eq!(
            contract["initial_ordinal_rule"]
                .as_str()
                .expect("initial rule"),
            ENVIRONMENT_RANDOMIZATION_INITIAL_ORDINAL_RULE_V2
        );
        assert_eq!(
            contract["overflow_rule"].as_str().expect("overflow rule"),
            ENVIRONMENT_RANDOMIZATION_OVERFLOW_RULE_V2
        );
        let owners: Vec<&str> = contract["owners"]
            .as_array()
            .expect("owners")
            .iter()
            .map(|v| v.as_str().expect("owner"))
            .collect();
        assert_eq!(owners, ENVIRONMENT_RANDOMIZATION_OWNERS_V2);
        let purposes: Vec<&str> = contract["purposes"]
            .as_array()
            .expect("purposes")
            .iter()
            .map(|v| v.as_str().expect("purpose"))
            .collect();
        assert_eq!(purposes, ENVIRONMENT_RANDOMIZATION_PURPOSES_V2);
        let ordered = contract["ordered_atoms"].as_array().expect("ordered atoms");
        let expected_atoms = serde_json::Value::Array(
            ENVIRONMENT_RANDOMIZATION_ORDERED_ATOMS_V2
                .iter()
                .map(|row| {
                    serde_json::Value::Array(
                        row.iter()
                            .map(|part| serde_json::Value::String((*part).to_owned()))
                            .collect(),
                    )
                })
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            serde_json::Value::Array(ordered.clone()),
            expected_atoms,
            "ordered atom descriptor must match the implementation literally"
        );
    }

    /// The promoted owner constants are the single source of the closed
    /// vocabulary: the enums project from them in both directions, and the
    /// ordered-atom rows keep their frozen arities.
    #[test]
    fn owner_constants_are_the_single_vocabulary_source() {
        assert_eq!(PhysicalOwnerV2::P0.as_str(), "p0");
        assert_eq!(PhysicalOwnerV2::P1.as_str(), "p1");
        assert_eq!(
            ShufflePurposeV2::InitialLibraryShuffle.as_str(),
            "initial-library-shuffle"
        );
        assert_eq!(
            ShufflePurposeV2::InGameLibraryShuffle.as_str(),
            "in-game-library-shuffle"
        );
        for (index, owner) in [PhysicalOwnerV2::P0, PhysicalOwnerV2::P1]
            .into_iter()
            .enumerate()
        {
            assert_eq!(owner.as_str(), ENVIRONMENT_RANDOMIZATION_OWNERS_V2[index]);
            assert_eq!(PhysicalOwnerV2::parse(owner.as_str()), Ok(owner));
        }
        for (index, purpose) in [
            ShufflePurposeV2::InitialLibraryShuffle,
            ShufflePurposeV2::InGameLibraryShuffle,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                purpose.as_str(),
                ENVIRONMENT_RANDOMIZATION_PURPOSES_V2[index]
            );
            assert_eq!(ShufflePurposeV2::parse(purpose.as_str()), Ok(purpose));
        }
        assert_eq!(
            PhysicalOwnerV2::parse("p2"),
            Err(EnvironmentRandomizationErrorV2::InvalidOwner)
        );
        assert_eq!(
            ShufflePurposeV2::parse("other-shuffle"),
            Err(EnvironmentRandomizationErrorV2::InvalidPurpose)
        );
        let arities: Vec<usize> = ENVIRONMENT_RANDOMIZATION_ORDERED_ATOMS_V2
            .iter()
            .map(|row| row.len())
            .collect();
        assert_eq!(arities, vec![2, 2, 4, 4, 4, 4]);
        assert_eq!(
            ENVIRONMENT_RANDOMIZATION_ORDERED_ATOMS_V2[0][1],
            ENVIRONMENT_RANDOMIZATION_IDENTITY_V2
        );
        assert_eq!(
            ENVIRONMENT_RANDOMIZATION_ORDERED_ATOMS_V2[1][1],
            ENVIRONMENT_RANDOMIZATION_NAMESPACE_V2
        );
    }

    #[test]
    fn reject_vectors_execute_with_expected_codes() {
        let doc = goldens();
        let rows = doc["reject_cases"].as_array().expect("reject cases");
        assert_eq!(rows.len(), 4, "golden reject cardinality");
        for row in rows {
            let input = &row["input"];
            let expected_code = row["expected_code"].as_str().expect("expected code");
            let root = input["pair_environment_seed"].as_u64().expect("root");
            let ordinal = input["ordinal"].as_u64().expect("ordinal");
            let observed_code =
                match PhysicalOwnerV2::parse(input["physical_owner"].as_str().expect("owner")) {
                    Err(EnvironmentRandomizationErrorV2::InvalidOwner) => "invalid-owner",
                    Err(_) => panic!("owner parse produced a non-owner error"),
                    Ok(owner) => {
                        match ShufflePurposeV2::parse(input["purpose"].as_str().expect("purpose")) {
                            Err(EnvironmentRandomizationErrorV2::InvalidPurpose) => {
                                "invalid-purpose"
                            }
                            Err(_) => panic!("purpose parse produced a non-purpose error"),
                            Ok(purpose) => {
                                match derive_environment_randomization_seed_v2(
                                    root, owner, purpose, ordinal,
                                ) {
                                    Err(EnvironmentRandomizationErrorV2::NonZeroInitialOrdinal) => {
                                        "nonzero-initial-ordinal"
                                    }
                                    Err(EnvironmentRandomizationErrorV2::OrdinalOverflow) => {
                                        "ordinal-overflow"
                                    }
                                    Err(_) => panic!("derive produced an unexpected error"),
                                    Ok(_) => panic!("reject vector unexpectedly derived"),
                                }
                            }
                        }
                    }
                };
            assert_eq!(
                observed_code, expected_code,
                "reject code mismatch for {}",
                row["name"]
            );
        }
    }

    #[test]
    fn rejects_invalid_vocabulary_and_ordinals() {
        assert_eq!(
            PhysicalOwnerV2::parse("P0"),
            Err(EnvironmentRandomizationErrorV2::InvalidOwner)
        );
        assert_eq!(
            ShufflePurposeV2::parse("live-shuffle"),
            Err(EnvironmentRandomizationErrorV2::InvalidPurpose)
        );
        assert_eq!(
            derive_environment_randomization_seed_v2(
                1,
                PhysicalOwnerV2::P1,
                ShufflePurposeV2::InitialLibraryShuffle,
                1,
            ),
            Err(EnvironmentRandomizationErrorV2::NonZeroInitialOrdinal)
        );
        assert_eq!(
            derive_environment_randomization_seed_v2(
                1,
                PhysicalOwnerV2::P0,
                ShufflePurposeV2::InGameLibraryShuffle,
                u64::MAX,
            ),
            Err(EnvironmentRandomizationErrorV2::OrdinalOverflow)
        );
        derive_environment_randomization_seed_v2(
            u64::MAX,
            PhysicalOwnerV2::P0,
            ShufflePurposeV2::InGameLibraryShuffle,
            0,
        )
        .expect("root may equal u64::MAX");
    }

    #[test]
    fn substreams_are_pairwise_distinct_and_root_sensitive() {
        let mut seen = std::collections::BTreeSet::new();
        for root in [0_u64, 1, u64::MAX] {
            for owner in [PhysicalOwnerV2::P0, PhysicalOwnerV2::P1] {
                for (purpose, ordinals) in [
                    (ShufflePurposeV2::InitialLibraryShuffle, &[0_u64][..]),
                    (ShufflePurposeV2::InGameLibraryShuffle, &[0, 1, 2][..]),
                ] {
                    for &ordinal in ordinals {
                        let seed =
                            derive_environment_randomization_seed_v2(root, owner, purpose, ordinal)
                                .expect("valid derivation");
                        assert!(
                            seen.insert(seed),
                            "collision across substreams at root {root} owner \
                             {owner:?} purpose {purpose:?} ordinal {ordinal}"
                        );
                    }
                }
            }
        }
    }
}

