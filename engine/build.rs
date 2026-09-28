#![allow(dead_code, unused_imports)]
//! Codegen: `data/cards_v1.json` -> `$OUT_DIR/card_defs.rs`, included
//! verbatim by `src/card_def.rs`.
//!
//! Fails the build on:
//! - schema version mismatch (`cards_v1.json`'s `"version"` != what this
//!   codegen understands)
//! - duplicate card names
//! - a card with empty deck coverage (`"decks": []`)
//!
//! `u16` ids are assigned in JSON array order (stable: the file is a
//! checked-in fixed pool, not regenerated per build). Executability and
//! full-support status come from each record's fail-closed
//! `engine_capability`; ordinary supported permanents and intrinsic basic-
//! land mana are generated from metadata, while exceptional rules text is
//! still composed explicitly below.
//!
//! The build also embeds a Git commit-tree integrity proof: build HEAD, its
//! exact Git tree object, clean/dirty status, and a deterministic SHA-256 over
//! every tracked path, mode, type, and Git blob (or gitlink id). This is not a sealed-builder or
//! complete rustc-input attestation; it does not claim to capture toolchain,
//! environment, proc-macro, generated, or every dependency byte consumed by
//! compilation.


use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const EXPECTED_SCHEMA_VERSION: u32 = 2;
const TRACKED_TREE_HASH_CONTRACT: &str =
    "git-ls-tree-r-z-path-mode-type-framed-blob-content-or-gitlink-oid-sha256/v1";

#[derive(Debug, Deserialize)]
struct CardJson {
    name: String,
    /// Absent means `no_effect`: adding a registry record can never make a
    /// card playable accidentally. `partial` is executable but is rejected
    /// by the full-deck preflight; `full` is both executable and accepted.
    #[serde(default)]
    engine_capability: EngineCapabilityJson,
    mana_cost: String,
    mana_value: u16,
    #[serde(default)]
    types: Vec<String>,
    #[serde(default)]
    subtypes: Vec<String>,
    #[serde(default)]
    supertypes: Vec<String>,
    power: Option<i32>,
    toughness: Option<i32>,
    is_land: bool,
    #[serde(default)]
    produces_mana: Vec<String>,
    #[serde(default)]
    colors: Vec<String>,
    #[serde(default)]
    mechanics: Vec<String>,
    decks: Vec<String>,
    /// A permanent token (e.g. Blood), not itself a deck card: exempt from
    /// the empty-deck-coverage check (see `main`'s validation loop) and
    /// never castable (its `Special` falls through to `Special::None`,
    /// giving it `no_effect`/`no_effect`/`TargetSpec::None` -- correct,
    /// since tokens are never cast, only created by another card's effect).
    #[serde(default)]
    is_token: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum EngineCapabilityJson {
    #[default]
    NoEffect,
    Partial,
    Full,
}

#[derive(Debug, Deserialize)]
struct CardsFile {
    version: u32,
    cards: Vec<CardJson>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeDeckCatalogJson {
    schema: String,
    protocol: String,
    source_hash_normalization: String,
    materialization: RuntimeDeckMaterializationJson,
    card_ids: RuntimeDeckCardIdsJson,
    decks: Vec<RuntimeDeckJson>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeDeckMaterializationJson {
    order: String,
    source_row_ordinal_base: u32,
    copy_ordinal_base: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeDeckCardIdsJson {
    assignment: String,
    deck_hash_algorithm: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeDeckJson {
    canonical_pool_order: u32,
    id: String,
    source_path: String,
    source_sha256: String,
    mainboard_copy_count: usize,
    unique_mainboard_cards: usize,
    runtime_deck_hash: String,
    materialized_mainboard: Vec<RuntimeDeckCopyJson>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeDeckCopyJson {
    source_row_ordinal: u32,
    copy_ordinal: u32,
    name: String,
    card_id: u16,
}

#[derive(Debug)]
struct GitTreeEntry {
    mode: Vec<u8>,
    kind: Vec<u8>,
    object_id: String,
    path: Vec<u8>,
}

fn git_output(repo_root: &Path, args: &[&str], operation: &str) -> Vec<u8> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .output()
        .unwrap_or_else(|error| panic!("failed to execute git for {operation}: {error}"));
    if !output.status.success() {
        panic!("git failed while resolving {operation}");
    }
    output.stdout
}

fn git_text(repo_root: &Path, args: &[&str], operation: &str) -> String {
    let bytes = git_output(repo_root, args, operation);
    std::str::from_utf8(&bytes)
        .unwrap_or_else(|_| panic!("git returned non-UTF-8 data for {operation}"))
        .trim()
        .to_string()
}

fn parse_tree_entries(bytes: &[u8]) -> Vec<GitTreeEntry> {
    let mut entries = Vec::new();
    for record in bytes
        .split(|byte| *byte == 0)
        .filter(|record| !record.is_empty())
    {
        let tab = record
            .iter()
            .position(|byte| *byte == b'\t')
            .expect("git ls-tree record contains a path separator");
        let mut metadata = record[..tab].split(|byte| *byte == b' ');
        let mode = metadata.next().expect("git ls-tree record contains a mode");
        let kind = metadata.next().expect("git ls-tree record contains a type");
        let object_id = metadata
            .next()
            .expect("git ls-tree record contains an object id");
        if metadata.next().is_some()
            || mode.is_empty()
            || kind.is_empty()
            || object_id.is_empty()
            || record[tab + 1..].is_empty()
        {
            panic!("git ls-tree returned malformed tracked-tree metadata");
        }
        let object_id = std::str::from_utf8(object_id)
            .expect("git object id is ASCII")
            .to_string();
        if !object_id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            panic!("git ls-tree returned a malformed object id");
        }
        entries.push(GitTreeEntry {
            mode: mode.to_vec(),
            kind: kind.to_vec(),
            object_id,
            path: record[tab + 1..].to_vec(),
        });
    }
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    if entries.windows(2).any(|pair| pair[0].path == pair[1].path) {
        panic!("git ls-tree returned duplicate tracked paths");
    }
    entries
}

fn git_blob_contents(repo_root: &Path, entries: &[GitTreeEntry]) -> Vec<Option<Vec<u8>>> {
    let mut child = Command::new("git")
        .args(["cat-file", "--batch"])
        .current_dir(repo_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("git cat-file starts for tracked-tree binding");
    {
        let stdin = child.stdin.as_mut().expect("git cat-file stdin is piped");
        for entry in entries.iter().filter(|entry| entry.kind == b"blob") {
            writeln!(stdin, "{}", entry.object_id).expect("git cat-file accepts tracked blob ids");
        }
    }
    let output = child
        .wait_with_output()
        .expect("git cat-file completes for tracked-tree binding");
    if !output.status.success() {
        panic!("git cat-file failed for tracked-tree binding");
    }

    let mut cursor = 0usize;
    let mut contents = Vec::with_capacity(entries.len());
    for entry in entries {
        if entry.kind == b"commit" {
            if entry.mode != b"160000" {
                panic!("tracked commit entry is not a gitlink");
            }
            contents.push(None);
            continue;
        }
        if entry.kind != b"blob" || entry.mode == b"160000" {
            panic!("tracked tree contains an unsupported entry type");
        }
        let relative_newline = output.stdout[cursor..]
            .iter()
            .position(|byte| *byte == b'\n')
            .expect("git cat-file response contains a header terminator");
        let header_end = cursor + relative_newline;
        let header = std::str::from_utf8(&output.stdout[cursor..header_end])
            .expect("git cat-file header is ASCII");
        let mut fields = header.split(' ');
        let returned_id = fields.next().expect("git cat-file header has object id");
        let returned_kind = fields.next().expect("git cat-file header has type");
        let size = fields
            .next()
            .expect("git cat-file header has size")
            .parse::<usize>()
            .expect("git cat-file blob size fits usize");
        if fields.next().is_some() || returned_id != entry.object_id || returned_kind != "blob" {
            panic!("git cat-file returned unexpected tracked blob metadata");
        }
        let content_start = header_end + 1;
        let content_end = content_start
            .checked_add(size)
            .expect("tracked blob bounds fit usize");
        if content_end >= output.stdout.len() || output.stdout[content_end] != b'\n' {
            panic!("git cat-file returned a truncated tracked blob");
        }
        contents.push(Some(output.stdout[content_start..content_end].to_vec()));
        cursor = content_end + 1;
    }
    if cursor != output.stdout.len() {
        panic!("git cat-file returned unconsumed tracked blob bytes");
    }
    contents
}

fn hash_frame(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn tracked_tree_sha256(repo_root: &Path, commit: &str) -> String {
    let listing = git_output(
        repo_root,
        &["ls-tree", "-r", "-z", "--full-tree", commit],
        "tracked tree listing",
    );
    let entries = parse_tree_entries(&listing);
    let contents = git_blob_contents(repo_root, &entries);
    let mut hasher = Sha256::new();
    hasher.update(TRACKED_TREE_HASH_CONTRACT.as_bytes());
    hasher.update([0]);
    hasher.update((entries.len() as u64).to_be_bytes());
    for (entry, content) in entries.iter().zip(contents.iter()) {
        hash_frame(&mut hasher, &entry.path);
        hash_frame(&mut hasher, &entry.mode);
        hash_frame(&mut hasher, &entry.kind);
        match content {
            Some(bytes) => hash_frame(&mut hasher, bytes),
            None => hash_frame(&mut hasher, entry.object_id.as_bytes()),
        }
    }
    format!("{:x}", hasher.finalize())
}

fn resolve_git_path(repo_root: &Path, name: &str) -> PathBuf {
    let resolved = git_text(
        repo_root,
        &["rev-parse", "--git-path", name],
        "git metadata path",
    );
    let path = PathBuf::from(resolved);
    if path.is_absolute() {
        path
    } else {
        repo_root.join(path)
    }
}

fn emit_commit_tree_rerun_inputs(repo_root: &Path) {
    for name in ["HEAD", "index", "packed-refs"] {
        println!(
            "cargo:rerun-if-changed={}",
            resolve_git_path(repo_root, name).display()
        );
    }
    let symbolic_ref = Command::new("git")
        .args(["symbolic-ref", "-q", "HEAD"])
        .current_dir(repo_root)
        .output()
        .expect("git symbolic-ref executes for commit-tree binding");
    if symbolic_ref.status.success() {
        let reference = std::str::from_utf8(&symbolic_ref.stdout)
            .expect("git symbolic ref is UTF-8")
            .trim();
        println!(
            "cargo:rerun-if-changed={}",
            resolve_git_path(repo_root, reference).display()
        );
    }
    let tracked = git_output(
        repo_root,
        &["ls-files", "-z"],
        "tracked worktree rerun inputs",
    );
    for path in tracked
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
    {
        let path =
            std::str::from_utf8(path).expect("tracked paths must be UTF-8 for Cargo rerun binding");
        println!("cargo:rerun-if-changed={}", repo_root.join(path).display());
    }
    for name in ["GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
}

fn configure_commit_tree_binding(repo_root: &Path) {
    emit_commit_tree_rerun_inputs(repo_root);
    let head = git_text(repo_root, &["rev-parse", "HEAD"], "build HEAD binding");
    if head.len() != 40
        || !head
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        panic!("build HEAD binding is not a lowercase SHA-1 commit id");
    }
    let treeish = format!("{head}^{{tree}}");
    let tree = git_text(
        repo_root,
        &["rev-parse", "--verify", &treeish],
        "build Git tree binding",
    );
    if tree.len() != 40
        || !tree
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        panic!("build Git tree binding is not a lowercase SHA-1 tree id");
    }
    let status = git_output(
        repo_root,
        &["status", "--porcelain=v1", "--untracked-files=all"],
        "build source status",
    );
    let clean = status.is_empty();
    let tree_sha256 = tracked_tree_sha256(repo_root, &head);
    println!("cargo:rustc-env=MTG_KERNEL_BUILD_GIT_HEAD={head}");
    println!("cargo:rustc-env=MTG_KERNEL_BUILD_GIT_TREE={tree}");
    println!("cargo:rustc-env=MTG_KERNEL_BUILD_GIT_CLEAN={clean}");
    println!("cargo:rustc-env=MTG_KERNEL_BUILD_TRACKED_TREE_SHA256={tree_sha256}");
    println!("cargo:rustc-env=MTG_KERNEL_BUILD_TRACKED_TREE_CONTRACT={TRACKED_TREE_HASH_CONTRACT}");
}

/// The build-override variables re-exported to the crate so
/// `model_guided_search_contract_digests_v1` can certify (or refuse to
/// certify) the build it is compiled into.
///
/// `option_env!` inside the crate is NOT sufficient on its own, which is
/// the whole reason this function exists. Flags configured in a
/// `.cargo/config.toml` `[build] rustflags` key or a
/// `[target.<triple>] rustflags` table are applied by Cargo to the rustc
/// invocation WITHOUT ever appearing in an environment variable the
/// compiled crate can observe, so a configured
/// `-C llvm-args=-fp-contract=fast` used to pass certification untouched.
///
/// A build script sees the resolved picture: Cargo sets
/// `CARGO_ENCODED_RUSTFLAGS` for build scripts, and that value already
/// includes config-derived flags. Re-exporting it with
/// `cargo:rustc-env` puts the effective flag set where `option_env!` can
/// read it at crate-compile time.
///
/// The pair `(source variable, re-exported name)`. Plain `RUSTC` is
/// deliberately absent: Cargo always sets it for build scripts (to the
/// rustc it is driving), so its mere presence is not evidence of an
/// override, and its value is an absolute path that would make any digest
/// binding it host-specific. The crate keeps its own compile-time
/// `option_env!("RUSTC")` check for that variable instead.
const EFFECTIVE_BUILD_FLAG_VARIABLES: &[(&str, &str)] = &[
    (
        "CARGO_ENCODED_RUSTFLAGS",
        "MTG_KERNEL_EFFECTIVE_ENCODED_RUSTFLAGS",
    ),
    ("RUSTFLAGS", "MTG_KERNEL_EFFECTIVE_RUSTFLAGS"),
    (
        "CARGO_BUILD_RUSTFLAGS",
        "MTG_KERNEL_EFFECTIVE_BUILD_RUSTFLAGS",
    ),
    ("RUSTC_WRAPPER", "MTG_KERNEL_EFFECTIVE_RUSTC_WRAPPER"),
    (
        "RUSTC_WORKSPACE_WRAPPER",
        "MTG_KERNEL_EFFECTIVE_RUSTC_WORKSPACE_WRAPPER",
    ),
    ("CARGO_BUILD_TARGET", "MTG_KERNEL_EFFECTIVE_BUILD_TARGET"),
];

/// Flattens a build-flag value into something safe to carry through
/// `cargo:rustc-env`, WITHOUT destroying the substrings the crate scans
/// for.
///
/// `CARGO_ENCODED_RUSTFLAGS` separates flags with `\x1f`, and any control
/// character (a newline above all) would corrupt the one-line
/// `cargo:` directive protocol. Every control character therefore becomes
/// a space: flags stay individually readable and every forbidden fragment
/// (`fp-contract`, `+fma`, ...) survives intact, because none of them
/// contains a control character. Hex- or base64-encoding the value would
/// have been safe too, but would have made the crate's substring scan
/// impossible without a decoder.
fn flatten_build_flag_value(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect()
}

/// Re-exports the effective build-flag environment to the crate.
///
/// Every variable is emitted unconditionally, empty when unset, so the
/// crate can distinguish "the build script reported no override" from
/// "the build script never ran". The latter is treated as a violation on
/// the crate side, because a certification that silently degrades to
/// "no evidence" is not a certification.
fn configure_effective_build_flags() {
    for (source, exported) in EFFECTIVE_BUILD_FLAG_VARIABLES {
        println!("cargo:rerun-if-env-changed={source}");
        let value = env::var(source).unwrap_or_default();
        println!(
            "cargo:rustc-env={exported}={}",
            flatten_build_flag_value(&value)
        );
    }
    // The target-specific table, which only exists once the triple is
    // known. `TARGET` is always set for build scripts; the variable name
    // Cargo honours is the upper-cased triple with `-` mapped to `_`.
    let target = env::var("TARGET").unwrap_or_default();
    let target_variable = format!(
        "CARGO_TARGET_{}_RUSTFLAGS",
        target.to_uppercase().replace('-', "_")
    );
    println!("cargo:rerun-if-env-changed={target_variable}");
    let target_value = env::var(&target_variable).unwrap_or_default();
    println!(
        "cargo:rustc-env=MTG_KERNEL_EFFECTIVE_TARGET_RUSTFLAGS={}",
        flatten_build_flag_value(&target_value)
    );
    // Recorded so a violation report can name the table it came from.
    println!("cargo:rustc-env=MTG_KERNEL_EFFECTIVE_TARGET_RUSTFLAGS_VARIABLE={target_variable}");
}

fn canonical_json(value: &Value, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&value.to_string()),
        Value::String(value) => output
            .push_str(&serde_json::to_string(value).expect("JSON strings serialize canonically")),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                canonical_json(value, output);
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort_unstable();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(
                    &serde_json::to_string(key).expect("JSON object keys serialize canonically"),
                );
                output.push(':');
                canonical_json(&values[key], output);
            }
            output.push('}');
        }
    }
}

fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn canonical_json_sha256(value: &Value) -> [u8; 32] {
    let mut canonical = String::new();
    canonical_json(value, &mut canonical);
    sha256_bytes(canonical.as_bytes())
}

fn sha256_hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn json_string<'a>(value: &'a Value, field: &str, source: &Path) -> &'a str {
    value
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{}: {field} must be a string", source.display()))
}

fn json_u64(value: &Value, field: &str, source: &Path) -> u64 {
    value
        .get(field)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("{}: {field} must be an unsigned integer", source.display()))
}

fn require_exact_object_keys(value: &Value, expected: &[&str], label: &str, source: &Path) {
    let object = value
        .as_object()
        .unwrap_or_else(|| panic!("{}: {label} must be an object", source.display()));
    let mut actual = object.keys().map(String::as_str).collect::<Vec<_>>();
    let mut expected = expected.to_vec();
    actual.sort_unstable();
    expected.sort_unstable();
    if actual != expected {
        panic!(
            "{}: {label} keys drift: expected {expected:?}, found {actual:?}",
            source.display()
        );
    }
}

fn require_exact_string_pairs(
    value: &Value,
    expected: &[(&str, &str)],
    label: &str,
    source: &Path,
) {
    let rows = value
        .as_array()
        .unwrap_or_else(|| panic!("{}: {label} must be an array", source.display()));
    if rows.len() != expected.len() {
        panic!(
            "{}: {label} width drift: expected {}, found {}",
            source.display(),
            expected.len(),
            rows.len()
        );
    }
    for (index, ((expected_name, expected_encoding), row)) in expected.iter().zip(rows).enumerate()
    {
        let pair = row.as_array().unwrap_or_else(|| {
            panic!(
                "{}: {label}[{index}] must be a two-string array",
                source.display()
            )
        });
        if pair.len() != 2
            || pair[0].as_str() != Some(expected_name)
            || pair[1].as_str() != Some(expected_encoding)
        {
            panic!(
                "{}: {label}[{index}] drift: expected [{expected_name:?}, {expected_encoding:?}], found {row}",
                source.display()
            );
        }
    }
}

fn parse_sha256(value: &str, label: &str) -> [u8; 32] {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        panic!("{label} must be a 64-character lowercase SHA-256");
    }
    let mut digest = [0_u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .expect("validated SHA-256 pairs parse as hexadecimal");
    }
    digest
}

fn require_digest(actual: [u8; 32], expected: [u8; 32], label: &str) {
    if actual != expected {
        panic!(
            "flat-policy-v1 {label} mismatch: generated {}, recomputed {}",
            sha256_hex(expected),
            sha256_hex(actual)
        );
    }
}

fn require_digest_v2(actual: [u8; 32], expected: [u8; 32], label: &str) {
    if actual != expected {
        panic!(
            "flat-policy-v2 {label} mismatch: generated {}, recomputed {}",
            sha256_hex(expected),
            sha256_hex(actual)
        );
    }
}

fn rust_byte_array(bytes: [u8; 32]) -> String {
    let mut output = String::from("[");
    for (index, byte) in bytes.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        write!(output, "0x{byte:02x}").expect("writing generated Rust to a String cannot fail");
    }
    output.push(']');
    output
}

fn flat_policy_contract_codegen(repo_root: &Path) -> String {
    const INVENTORY_SCHEMA: &str = "flat-policy-feature-inventory-v1";
    const GOLDENS_SCHEMA: &str = "flat-policy-v1-independent-goldens-v1";
    const CROSSWALK_SCHEMA: &str = "flat-policy-action-ref-role-crosswalk-v1";
    const CROSSWALK_VERSION: u64 = 1;
    const INTERNAL_ROLES: [(&str, u64, u64); 8] = [
        ("source", 0, 0),
        ("candidate", 1, 1),
        ("card", 2, 2),
        ("attacker", 3, 3),
        ("blocker", 4, 4),
        ("target_object", 5, 5),
        ("cards", 6, 6),
        ("pending_sources", 7, 9),
    ];
    const PROJECTION_ROLES: [(&str, u64); 10] = [
        ("source", 0),
        ("candidate", 1),
        ("card", 2),
        ("attacker", 3),
        ("blocker", 4),
        ("target_object", 5),
        ("cards", 6),
        ("attackers", 7),
        ("blockers", 8),
        ("pending_sources", 9),
    ];
    const PROJECTION_ONLY: [(&str, u64); 2] = [("attackers", 7), ("blockers", 8)];

    let inventory_path = repo_root.join("data/flat_policy_v1/feature_inventory_v1.json");
    let goldens_path = repo_root.join("data/flat_policy_v1/goldens_v1.json");
    let typed_layout_path = repo_root.join("mtg-kernel/src/flat_policy_v1.rs");
    let features_path = repo_root.join("python/mtg_kernel_rl/features.py");
    let cards_path = repo_root.join("data/cards_v1.json");
    for path in [
        &inventory_path,
        &goldens_path,
        &typed_layout_path,
        &features_path,
        &cards_path,
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let inventory_bytes = fs::read(&inventory_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", inventory_path.display()));
    let inventory: Value = serde_json::from_slice(&inventory_bytes)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", inventory_path.display()));
    let goldens_bytes = fs::read(&goldens_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", goldens_path.display()));
    let goldens: Value = serde_json::from_slice(&goldens_bytes)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", goldens_path.display()));

    if json_string(&inventory, "schema", &inventory_path) != INVENTORY_SCHEMA {
        panic!("{}: unsupported schema", inventory_path.display());
    }
    if json_string(&goldens, "schema", &goldens_path) != GOLDENS_SCHEMA {
        panic!("{}: unsupported schema", goldens_path.display());
    }

    let inventory_digest = canonical_json_sha256(&inventory);
    require_digest(
        inventory_digest,
        parse_sha256(
            json_string(&goldens, "inventory_sha256", &goldens_path),
            "goldens inventory_sha256",
        ),
        "canonical inventory digest",
    );

    let typed_layout_digest = parse_sha256(
        json_string(&inventory, "rust_typed_layout_sha256", &inventory_path),
        "inventory rust_typed_layout_sha256",
    );
    require_digest(
        sha256_bytes(&fs::read(&typed_layout_path).unwrap_or_else(|error| {
            panic!("failed to read {}: {error}", typed_layout_path.display())
        })),
        typed_layout_digest,
        "Rust typed-layout source digest",
    );
    require_digest(
        sha256_bytes(
            &fs::read(&features_path).unwrap_or_else(|error| {
                panic!("failed to read {}: {error}", features_path.display())
            }),
        ),
        parse_sha256(
            json_string(&inventory, "authoritative_features_sha256", &inventory_path),
            "inventory authoritative_features_sha256",
        ),
        "authoritative Python features source digest",
    );
    require_digest(
        sha256_bytes(
            &fs::read(&cards_path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", cards_path.display())),
        ),
        parse_sha256(
            json_string(&goldens, "card_catalog_sha256", &goldens_path),
            "goldens card_catalog_sha256",
        ),
        "card catalog source digest",
    );
    for field in ["feature_contract_digest", "encoding_contract_digest"] {
        let inventory_value = parse_sha256(
            json_string(&inventory, field, &inventory_path),
            &format!("inventory {field}"),
        );
        let golden_value = parse_sha256(
            json_string(&goldens, field, &goldens_path),
            &format!("goldens {field}"),
        );
        require_digest(inventory_value, golden_value, field);
    }

    let enum_maps = goldens
        .get("enum_maps")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("{}: enum_maps must be an object", goldens_path.display()));
    let projection_map = enum_maps
        .get("action_ref_role")
        .and_then(Value::as_object)
        .unwrap_or_else(|| {
            panic!(
                "{}: enum_maps.action_ref_role must be an object",
                goldens_path.display()
            )
        });
    if projection_map.len() != PROJECTION_ROLES.len() {
        panic!("flat-policy-v1 projection action-ref role width changed without a V1 update");
    }
    for (role, expected_id) in PROJECTION_ROLES {
        if projection_map.get(role).and_then(Value::as_u64) != Some(expected_id) {
            panic!("flat-policy-v1 projection role {role:?} changed without a V1 update");
        }
    }

    let crosswalk = goldens.get("action_ref_role_crosswalk").unwrap_or_else(|| {
        panic!(
            "{}: missing action_ref_role_crosswalk",
            goldens_path.display()
        )
    });
    let crosswalk_object = crosswalk.as_object().unwrap_or_else(|| {
        panic!(
            "{}: action_ref_role_crosswalk must be an object",
            goldens_path.display()
        )
    });
    if crosswalk_object.len() != 6
        || json_string(crosswalk, "schema", &goldens_path) != CROSSWALK_SCHEMA
        || json_u64(crosswalk, "mapping_version", &goldens_path) != CROSSWALK_VERSION
        || json_u64(crosswalk, "rust_internal_width", &goldens_path) != INTERNAL_ROLES.len() as u64
        || json_u64(crosswalk, "python_projection_width", &goldens_path)
            != PROJECTION_ROLES.len() as u64
    {
        panic!("flat-policy-v1 action-ref role crosswalk envelope changed without a V1 update");
    }
    let entries = crosswalk
        .get("entries")
        .and_then(Value::as_array)
        .unwrap_or_else(|| {
            panic!(
                "{}: crosswalk entries must be an array",
                goldens_path.display()
            )
        });
    if entries.len() != INTERNAL_ROLES.len() {
        panic!("flat-policy-v1 internal action-ref role width changed without a V1 update");
    }
    for (entry, (role, internal_id, projection_id)) in entries.iter().zip(INTERNAL_ROLES) {
        if entry.as_object().is_none_or(|entry| entry.len() != 3)
            || json_string(entry, "role", &goldens_path) != role
            || json_u64(entry, "rust_internal_id", &goldens_path) != internal_id
            || json_u64(entry, "python_projection_id", &goldens_path) != projection_id
        {
            panic!("flat-policy-v1 internal role {role:?} crosswalk changed without a V1 update");
        }
    }
    let projection_only = crosswalk
        .get("projection_only")
        .and_then(Value::as_array)
        .unwrap_or_else(|| {
            panic!(
                "{}: crosswalk projection_only must be an array",
                goldens_path.display()
            )
        });
    if projection_only.len() != PROJECTION_ONLY.len() {
        panic!("flat-policy-v1 projection-only action-ref roles changed without a V1 update");
    }
    for (entry, (role, projection_id)) in projection_only.iter().zip(PROJECTION_ONLY) {
        if entry.as_object().is_none_or(|entry| entry.len() != 2)
            || json_string(entry, "role", &goldens_path) != role
            || json_u64(entry, "python_projection_id", &goldens_path) != projection_id
        {
            panic!("flat-policy-v1 projection-only role {role:?} changed without a V1 update");
        }
    }

    let mapping_contract = serde_json::json!({
        "action_ref_role_crosswalk": crosswalk,
        "enum_maps": Value::Object(enum_maps.clone()),
    });
    let mapping_digest = canonical_json_sha256(&mapping_contract);
    require_digest(
        mapping_digest,
        parse_sha256(
            json_string(&goldens, "mapping_sha256", &goldens_path),
            "goldens mapping_sha256",
        ),
        "mapping digest",
    );

    let payload_digest = parse_sha256(
        json_string(&goldens, "payload_sha256", &goldens_path),
        "goldens payload_sha256",
    );
    let mut payload_without_digest = goldens.clone();
    payload_without_digest
        .as_object_mut()
        .expect("goldens root was validated as an object")
        .remove("payload_sha256");
    require_digest(
        canonical_json_sha256(&payload_without_digest),
        payload_digest,
        "goldens payload digest",
    );

    let internal_to_projection = INTERNAL_ROLES
        .iter()
        .map(|(_, _, projection_id)| projection_id.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "// @generated by mtg-kernel/build.rs from data/flat_policy_v1; do not edit.\n\
pub const FLAT_POLICY_MAPPING_SHA256_V1: [u8; 32] = {};\n\
pub const FLAT_POLICY_FEATURE_INVENTORY_SHA256_V1: [u8; 32] = {};\n\
pub const FLAT_POLICY_TYPED_LAYOUT_SHA256_V1: [u8; 32] = {};\n\
pub const FLAT_ACTION_REF_PROJECTION_ROLE_MAPPING_VERSION_V1: u32 = {CROSSWALK_VERSION};\n\
pub const FLAT_ACTION_REF_INTERNAL_ROLE_WIDTH_V1: u8 = {};\n\
pub const FLAT_ACTION_REF_PROJECTION_ROLE_WIDTH_V1: u8 = {};\n\
pub const FLAT_ACTION_REF_INTERNAL_TO_PROJECTION_V1: [u8; {}] = [{}];\n",
        rust_byte_array(mapping_digest),
        rust_byte_array(inventory_digest),
        rust_byte_array(typed_layout_digest),
        INTERNAL_ROLES.len(),
        PROJECTION_ROLES.len(),
        INTERNAL_ROLES.len(),
        internal_to_projection,
    )
}

fn flat_policy_contract_v2_codegen(repo_root: &Path) -> (String, String) {
    const INVENTORY_SCHEMA: &str = "flat-policy-feature-inventory-v2";
    const GOLDENS_SCHEMA: &str = "flat-policy-v2-independent-goldens-v1";
    const ACTION_CONTRACT_SCHEMA: &str = "flat-action-contract-v2";
    const BASE_INVENTORY_SCHEMA: &str = "flat-policy-feature-inventory-v1";
    const CROSSWALK_SCHEMA: &str = "flat-policy-action-ref-role-crosswalk-v1";
    const TYPED_LAYOUT_DOMAIN: &[u8] = b"mtg-kernel-flat-policy-typed-layout-v2\0";
    const VISIBLE_MANIFEST: &str = "globals,objects,relations,object_subtypes,ability_uses,goads,completed_dungeons,effect_subtype_changes,context_path_elements,actions,action_refs";
    const INTERNAL_ROLES: [(&str, u64, u64); 8] = [
        ("source", 0, 0),
        ("candidate", 1, 1),
        ("card", 2, 2),
        ("attacker", 3, 3),
        ("blocker", 4, 4),
        ("target_object", 5, 5),
        ("cards", 6, 6),
        ("pending_sources", 7, 9),
    ];
    const PROJECTION_ONLY: [(&str, u64); 2] = [("attackers", 7), ("blockers", 8)];

    let inventory_path = repo_root.join("data/flat_policy_v2/feature_inventory_v2.json");
    let goldens_path = repo_root.join("data/flat_policy_v2/goldens_v2.json");
    let action_contract_path = repo_root.join("data/flat_policy_v2/action_contract_v2.json");
    let base_inventory_path = repo_root.join("data/flat_policy_v1/feature_inventory_v1.json");
    let base_goldens_path = repo_root.join("data/flat_policy_v1/goldens_v1.json");
    let base_layout_path = repo_root.join("mtg-kernel/src/flat_policy_v1.rs");
    let overlay_layout_path = repo_root.join("mtg-kernel/src/flat_policy_v2.rs");
    let features_path = repo_root.join("python/mtg_kernel_rl/features.py");
    let topology_audit_path = repo_root.join("data/flat_policy_v2/ordered_topology_audit_v2.md");
    for path in [
        &inventory_path,
        &goldens_path,
        &action_contract_path,
        &base_inventory_path,
        &base_goldens_path,
        &base_layout_path,
        &overlay_layout_path,
        &features_path,
        &topology_audit_path,
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let read_json = |path: &Path| {
        let bytes = fs::read(path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let value = serde_json::from_slice(&bytes)
            .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()));
        (bytes, value)
    };
    let (_inventory_bytes, inventory): (Vec<u8>, Value) = read_json(&inventory_path);
    let (_, goldens): (Vec<u8>, Value) = read_json(&goldens_path);
    let (action_contract_bytes, action_contract): (Vec<u8>, Value) =
        read_json(&action_contract_path);
    let (base_inventory_bytes, base_inventory): (Vec<u8>, Value) = read_json(&base_inventory_path);
    let (_, base_goldens): (Vec<u8>, Value) = read_json(&base_goldens_path);

    if json_string(&inventory, "schema", &inventory_path) != INVENTORY_SCHEMA {
        panic!("{}: unsupported schema", inventory_path.display());
    }
    if json_string(&goldens, "schema", &goldens_path) != GOLDENS_SCHEMA {
        panic!("{}: unsupported schema", goldens_path.display());
    }
    if json_string(&action_contract, "schema", &action_contract_path) != ACTION_CONTRACT_SCHEMA {
        panic!("{}: unsupported schema", action_contract_path.display());
    }
    if json_string(&base_inventory, "schema", &base_inventory_path) != BASE_INVENTORY_SCHEMA {
        panic!("{}: unsupported base schema", base_inventory_path.display());
    }

    require_exact_object_keys(
        &action_contract,
        &[
            "schema",
            "source_digest_semantics",
            "semantic_digest_semantics",
            "type_identities",
            "typed_layouts",
            "versions",
            "reference_role_mapping",
            "candidate_commitment",
            "card_token_contract",
        ],
        "action contract root",
        &action_contract_path,
    );
    if json_string(
        &action_contract,
        "source_digest_semantics",
        &action_contract_path,
    ) != "raw_utf8_file_bytes_sha256"
        || json_string(
            &action_contract,
            "semantic_digest_semantics",
            &action_contract_path,
        ) != "canonical_json_utf8_sha256"
    {
        panic!(
            "{}: action-contract digest semantics drift",
            action_contract_path.display()
        );
    }
    let type_identities = action_contract.get("type_identities").unwrap_or_else(|| {
        panic!(
            "{}: missing type_identities",
            action_contract_path.display()
        )
    });
    require_exact_object_keys(
        type_identities,
        &[
            "action_row",
            "object_row",
            "reference_row",
            "binding",
            "slice",
            "buffers",
        ],
        "action contract type_identities",
        &action_contract_path,
    );
    for (field, expected) in [
        ("action_row", "FlatActionCoreV1_shared_by_v2"),
        ("object_row", "FlatActionObjectV2"),
        ("reference_row", "FlatActionRefV2"),
        ("binding", "FlatActionDecisionBindingV2"),
        ("slice", "FlatActionDecisionSliceV2"),
        ("buffers", "FlatActionDecisionSliceBuffersV2"),
    ] {
        if json_string(type_identities, field, &action_contract_path) != expected {
            panic!(
                "{}: action-contract type identity {field} drift",
                action_contract_path.display()
            );
        }
    }
    let action_typed_layouts = action_contract
        .get("typed_layouts")
        .unwrap_or_else(|| panic!("{}: missing typed_layouts", action_contract_path.display()));
    require_exact_object_keys(
        action_typed_layouts,
        &[
            "FlatActionDecisionBindingV2",
            "FlatActionDecisionSliceV2",
            "FlatActionDecisionSliceBuffersV2",
        ],
        "action contract typed_layouts",
        &action_contract_path,
    );
    require_exact_string_pairs(
        action_typed_layouts
            .get("FlatActionDecisionBindingV2")
            .unwrap_or_else(|| {
                panic!(
                    "{}: missing FlatActionDecisionBindingV2 layout",
                    action_contract_path.display()
                )
            }),
        &[
            ("slice_version", "u32"),
            ("ref_role_mapping_version", "u32"),
            ("card_token_mapping_version", "u32"),
            ("candidate_commitment_version", "u32"),
            ("card_db_hash", "u64"),
            ("episode_id", "u64"),
            ("environment_revision", "u64"),
            ("bound_policy_step_count", "u64"),
            ("physical_decision_id", "u64"),
            ("bound_physical_decision_count", "u64"),
            ("substep_index", "u32"),
            ("substep_count", "u32"),
            ("acting_player", "u8"),
            ("decision_kind", "u8"),
            ("legal_action_count", "u32"),
            ("candidate_order_commitment", "[u8;16]"),
        ],
        "FlatActionDecisionBindingV2 layout",
        &action_contract_path,
    );
    require_exact_string_pairs(
        action_typed_layouts
            .get("FlatActionDecisionSliceV2")
            .unwrap_or_else(|| {
                panic!(
                    "{}: missing FlatActionDecisionSliceV2 layout",
                    action_contract_path.display()
                )
            }),
        &[
            ("binding", "FlatActionDecisionBindingV2"),
            ("active_action_count", "u32"),
            ("active_ref_count", "u32"),
            ("active_object_count", "u16"),
        ],
        "FlatActionDecisionSliceV2 layout",
        &action_contract_path,
    );
    require_exact_string_pairs(
        action_typed_layouts
            .get("FlatActionDecisionSliceBuffersV2")
            .unwrap_or_else(|| {
                panic!(
                    "{}: missing FlatActionDecisionSliceBuffersV2 layout",
                    action_contract_path.display()
                )
            }),
        &[
            ("actions", "&mut[FlatActionCoreV1]"),
            ("refs", "&mut[FlatActionRefV2]"),
            ("objects", "&mut[FlatActionObjectV2]"),
        ],
        "FlatActionDecisionSliceBuffersV2 layout",
        &action_contract_path,
    );
    let action_versions = action_contract
        .get("versions")
        .unwrap_or_else(|| panic!("{}: missing versions", action_contract_path.display()));
    require_exact_object_keys(
        action_versions,
        &[
            "slice",
            "reference_role_mapping",
            "card_token_mapping",
            "candidate_commitment",
        ],
        "action contract versions",
        &action_contract_path,
    );
    for (field, expected) in [
        ("slice", 2),
        ("reference_role_mapping", 1),
        ("card_token_mapping", 2),
        ("candidate_commitment", 2),
    ] {
        if json_u64(action_versions, field, &action_contract_path) != expected {
            panic!(
                "{}: action-contract version {field} drift",
                action_contract_path.display()
            );
        }
    }

    let reference_role_mapping = action_contract
        .get("reference_role_mapping")
        .unwrap_or_else(|| {
            panic!(
                "{}: missing reference_role_mapping",
                action_contract_path.display()
            )
        });
    require_exact_object_keys(
        reference_role_mapping,
        &[
            "authority_path",
            "authority_schema",
            "mapping_version",
            "canonical_sha256",
            "compatibility",
        ],
        "action contract reference_role_mapping",
        &action_contract_path,
    );
    if json_string(
        reference_role_mapping,
        "authority_path",
        &action_contract_path,
    ) != "data/flat_policy_v1/goldens_v1.json#action_ref_role_crosswalk"
        || json_string(
            reference_role_mapping,
            "authority_schema",
            &action_contract_path,
        ) != CROSSWALK_SCHEMA
        || json_string(
            reference_role_mapping,
            "compatibility",
            &action_contract_path,
        ) != "v2_exactly_reuses_all_v1_internal_and_projection_role_ids"
    {
        panic!(
            "{}: reference-role authority identity drift",
            action_contract_path.display()
        );
    }

    let base_crosswalk = base_goldens
        .get("action_ref_role_crosswalk")
        .unwrap_or_else(|| {
            panic!(
                "{}: missing action_ref_role_crosswalk",
                base_goldens_path.display()
            )
        });
    require_exact_object_keys(
        base_crosswalk,
        &[
            "entries",
            "mapping_version",
            "projection_only",
            "python_projection_width",
            "rust_internal_width",
            "schema",
        ],
        "V1 action-ref role crosswalk authority",
        &base_goldens_path,
    );
    let crosswalk_version = json_u64(base_crosswalk, "mapping_version", &base_goldens_path);
    let internal_role_width = json_u64(base_crosswalk, "rust_internal_width", &base_goldens_path);
    let projection_role_width = json_u64(
        base_crosswalk,
        "python_projection_width",
        &base_goldens_path,
    );
    if json_string(base_crosswalk, "schema", &base_goldens_path) != CROSSWALK_SCHEMA
        || crosswalk_version != 1
        || internal_role_width != INTERNAL_ROLES.len() as u64
        || projection_role_width != 10
        || json_u64(
            reference_role_mapping,
            "mapping_version",
            &action_contract_path,
        ) != crosswalk_version
        || json_u64(
            action_versions,
            "reference_role_mapping",
            &action_contract_path,
        ) != crosswalk_version
    {
        panic!("V2 action-ref role mapping identity diverged from its V1 authority");
    }
    let entries = base_crosswalk
        .get("entries")
        .and_then(Value::as_array)
        .unwrap_or_else(|| {
            panic!(
                "{}: crosswalk entries must be an array",
                base_goldens_path.display()
            )
        });
    if entries.len() != INTERNAL_ROLES.len() {
        panic!("V2 action-ref internal role width diverged from its V1 authority");
    }
    let mut internal_to_projection = Vec::with_capacity(INTERNAL_ROLES.len());
    for (entry, (role, internal_id, projection_id)) in entries.iter().zip(INTERNAL_ROLES) {
        if entry.as_object().is_none_or(|entry| entry.len() != 3)
            || json_string(entry, "role", &base_goldens_path) != role
            || json_u64(entry, "rust_internal_id", &base_goldens_path) != internal_id
            || json_u64(entry, "python_projection_id", &base_goldens_path) != projection_id
        {
            panic!(
                "V2 action-ref role {role:?} diverged from the validated V1 crosswalk authority"
            );
        }
        internal_to_projection.push(
            u8::try_from(projection_id)
                .expect("validated V1 action-ref projection ids must fit in u8"),
        );
    }
    let projection_only = base_crosswalk
        .get("projection_only")
        .and_then(Value::as_array)
        .unwrap_or_else(|| {
            panic!(
                "{}: crosswalk projection_only must be an array",
                base_goldens_path.display()
            )
        });
    if projection_only.len() != PROJECTION_ONLY.len() {
        panic!("V2 projection-only role width diverged from its V1 authority");
    }
    for (entry, (role, projection_id)) in projection_only.iter().zip(PROJECTION_ONLY) {
        if entry.as_object().is_none_or(|entry| entry.len() != 2)
            || json_string(entry, "role", &base_goldens_path) != role
            || json_u64(entry, "python_projection_id", &base_goldens_path) != projection_id
        {
            panic!(
                "V2 projection-only role {role:?} diverged from the validated V1 crosswalk authority"
            );
        }
    }
    let action_ref_role_crosswalk_digest = canonical_json_sha256(base_crosswalk);
    require_digest_v2(
        action_ref_role_crosswalk_digest,
        parse_sha256(
            json_string(
                reference_role_mapping,
                "canonical_sha256",
                &action_contract_path,
            ),
            "V2 reference-role authority canonical_sha256",
        ),
        "V1 action-reference crosswalk authority digest",
    );

    let commitment_contract = action_contract
        .get("candidate_commitment")
        .unwrap_or_else(|| {
            panic!(
                "{}: missing candidate_commitment",
                action_contract_path.display()
            )
        });
    require_exact_object_keys(
        commitment_contract,
        &[
            "domain_utf8",
            "stream_order",
            "digest",
            "published_bytes",
            "header_fields",
            "object_row",
            "action_row",
            "reference_row",
        ],
        "action contract candidate_commitment",
        &action_contract_path,
    );
    let action_commitment_domain =
        json_string(commitment_contract, "domain_utf8", &action_contract_path);
    if action_commitment_domain != "mtg-kernel-flat-action-candidate-order-v2\0"
        || json_string(commitment_contract, "stream_order", &action_contract_path)
            != "header_then_objects_by_object_index_then_each_actions_refs_in_slice_order_then_action"
        || json_string(commitment_contract, "digest", &action_contract_path) != "sha256"
        || json_string(
            commitment_contract,
            "published_bytes",
            &action_contract_path,
        ) != "first_16"
    {
        panic!(
            "{}: candidate commitment identity drift",
            action_contract_path.display()
        );
    }
    require_exact_string_pairs(
        commitment_contract
            .get("header_fields")
            .unwrap_or_else(|| panic!("{}: missing header_fields", action_contract_path.display())),
        &[
            ("slice_version", "u32_le"),
            ("reference_role_mapping_version", "u32_le"),
            ("card_token_mapping_version", "u32_le"),
            ("candidate_commitment_version", "u32_le"),
            ("card_db_hash", "u64_le"),
            ("actor_seat", "u8_p0_0_p1_1"),
            ("action_count", "u32_le"),
            ("reference_count", "u32_le"),
            ("object_count", "u16_le"),
        ],
        "action contract header_fields",
        &action_contract_path,
    );
    let object_row = commitment_contract
        .get("object_row")
        .unwrap_or_else(|| panic!("{}: missing object_row", action_contract_path.display()));
    require_exact_object_keys(
        object_row,
        &["marker_ascii", "fields"],
        "action contract object_row",
        &action_contract_path,
    );
    if json_string(object_row, "marker_ascii", &action_contract_path) != "O" {
        panic!("{}: object marker drift", action_contract_path.display());
    }
    require_exact_string_pairs(
        object_row
            .get("fields")
            .unwrap_or_else(|| panic!("{}: missing object fields", action_contract_path.display())),
        &[
            ("object_index", "u16_le"),
            ("card_token", "u32_le"),
            ("group", "u8_discriminant"),
            ("actor_visible_ordinal", "u16_le"),
            ("owner_relative", "u8"),
            ("controller_relative", "u8"),
            ("zone", "u8"),
            ("zone_change_count", "u32_le"),
        ],
        "action contract object fields",
        &action_contract_path,
    );
    let action_row = commitment_contract
        .get("action_row")
        .unwrap_or_else(|| panic!("{}: missing action_row", action_contract_path.display()));
    require_exact_object_keys(
        action_row,
        &["marker_ascii", "fields"],
        "action contract action_row",
        &action_contract_path,
    );
    if json_string(action_row, "marker_ascii", &action_contract_path) != "A" {
        panic!("{}: action marker drift", action_contract_path.display());
    }
    require_exact_string_pairs(
        action_row
            .get("fields")
            .unwrap_or_else(|| panic!("{}: missing action fields", action_contract_path.display())),
        &[
            ("action_index", "u32_le"),
            ("kind", "u8_discriminant"),
            ("flags", "u16_le"),
            ("ability_index", "u8"),
            ("remaining", "u8"),
            ("mode_index", "u8"),
            ("mode_count", "u8"),
            ("option_index", "u16_le"),
            ("option_count", "u16_le"),
            ("selected_count", "u16_le"),
            ("min_targets", "u16_le"),
            ("max_targets", "u16_le"),
            ("number", "i32_le_twos_complement"),
            ("minimum", "i32_le_twos_complement"),
            ("maximum", "i32_le_twos_complement"),
            ("mana_choice", "u8"),
            ("color", "u8"),
            ("cast_mode", "u8"),
            ("cost_kind", "u8"),
            ("optional_cost_choice", "u8"),
            ("target_kind", "u8"),
            ("target_player", "u8"),
            ("ref_start", "u32_le"),
            ("ref_len", "u16_le"),
        ],
        "action contract action fields",
        &action_contract_path,
    );
    let reference_row = commitment_contract
        .get("reference_row")
        .unwrap_or_else(|| panic!("{}: missing reference_row", action_contract_path.display()));
    require_exact_object_keys(
        reference_row,
        &["marker_ascii", "fields"],
        "action contract reference_row",
        &action_contract_path,
    );
    if json_string(reference_row, "marker_ascii", &action_contract_path) != "R" {
        panic!("{}: reference marker drift", action_contract_path.display());
    }
    require_exact_string_pairs(
        reference_row.get("fields").unwrap_or_else(|| {
            panic!(
                "{}: missing reference fields",
                action_contract_path.display()
            )
        }),
        &[
            ("action_index", "u32_le"),
            ("role", "u8_discriminant"),
            ("order_index", "u16_le"),
            ("associated_order", "u16_le"),
            ("card_token", "u32_le"),
            ("object_index", "u16_le"),
            ("bound_object.card_token", "u32_le"),
            ("bound_object.group", "u8_discriminant"),
            ("bound_object.actor_visible_ordinal", "u16_le"),
            ("bound_object.owner_relative", "u8"),
            ("bound_object.controller_relative", "u8"),
            ("bound_object.zone", "u8"),
            ("bound_object.zone_change_count", "u32_le"),
        ],
        "action contract reference fields",
        &action_contract_path,
    );
    let action_token_contract = action_contract
        .get("card_token_contract")
        .unwrap_or_else(|| {
            panic!(
                "{}: missing card_token_contract",
                action_contract_path.display()
            )
        });
    require_exact_object_keys(
        action_token_contract,
        &[
            "minimum",
            "maximum",
            "encoding",
            "wire_encoding",
            "zero_reserved_for_padding",
        ],
        "action contract card_token_contract",
        &action_contract_path,
    );
    if json_u64(action_token_contract, "minimum", &action_contract_path) != 1
        || json_u64(action_token_contract, "maximum", &action_contract_path) != 65_536
        || json_string(action_token_contract, "encoding", &action_contract_path)
            != "card_db_id_plus_one_u32"
        || json_string(
            action_token_contract,
            "wire_encoding",
            &action_contract_path,
        ) != "u32_le"
        || action_token_contract
            .get("zero_reserved_for_padding")
            .and_then(Value::as_bool)
            != Some(true)
    {
        panic!(
            "{}: card-token contract drift",
            action_contract_path.display()
        );
    }

    let action_contract_source_digest = sha256_bytes(&action_contract_bytes);
    let action_contract_semantic_digest = canonical_json_sha256(&action_contract);
    let action_inventory = inventory.get("action_contract").unwrap_or_else(|| {
        panic!(
            "{}: missing action_contract binding",
            inventory_path.display()
        )
    });
    require_exact_object_keys(
        action_inventory,
        &["path", "schema", "source_sha256", "canonical_sha256"],
        "inventory action_contract",
        &inventory_path,
    );
    if json_string(action_inventory, "path", &inventory_path)
        != "data/flat_policy_v2/action_contract_v2.json"
        || json_string(action_inventory, "schema", &inventory_path) != ACTION_CONTRACT_SCHEMA
    {
        panic!(
            "{}: action-contract inventory identity drift",
            inventory_path.display()
        );
    }
    require_digest_v2(
        action_contract_source_digest,
        parse_sha256(
            json_string(action_inventory, "source_sha256", &inventory_path),
            "V2 action-contract source_sha256",
        ),
        "action-contract raw source digest",
    );
    require_digest_v2(
        action_contract_semantic_digest,
        parse_sha256(
            json_string(action_inventory, "canonical_sha256", &inventory_path),
            "V2 action-contract canonical_sha256",
        ),
        "action-contract canonical semantic digest",
    );

    let inventory_digest = canonical_json_sha256(&inventory);
    require_digest_v2(
        inventory_digest,
        parse_sha256(
            json_string(&goldens, "inventory_sha256", &goldens_path),
            "V2 goldens inventory_sha256",
        ),
        "canonical inventory digest",
    );
    let payload_digest = parse_sha256(
        json_string(&goldens, "payload_sha256", &goldens_path),
        "V2 goldens payload_sha256",
    );
    let mut payload_without_digest = goldens.clone();
    payload_without_digest
        .as_object_mut()
        .expect("V2 goldens root was validated as an object")
        .remove("payload_sha256");
    require_digest_v2(
        canonical_json_sha256(&payload_without_digest),
        payload_digest,
        "goldens payload digest",
    );

    let base = inventory
        .get("base_inventory")
        .unwrap_or_else(|| panic!("{}: missing base_inventory", inventory_path.display()));
    if json_string(base, "schema", &inventory_path) != BASE_INVENTORY_SCHEMA {
        panic!("{}: base inventory schema drift", inventory_path.display());
    }
    require_digest_v2(
        sha256_bytes(&base_inventory_bytes),
        parse_sha256(
            json_string(base, "source_sha256", &inventory_path),
            "V2 base inventory source_sha256",
        ),
        "base inventory source digest",
    );
    require_digest_v2(
        canonical_json_sha256(&base_inventory),
        parse_sha256(
            json_string(base, "canonical_sha256", &inventory_path),
            "V2 base inventory canonical_sha256",
        ),
        "base inventory canonical digest",
    );

    let typed_layout = inventory
        .get("typed_layout")
        .unwrap_or_else(|| panic!("{}: missing typed_layout", inventory_path.display()));
    if json_string(typed_layout, "domain", &inventory_path).as_bytes() != TYPED_LAYOUT_DOMAIN {
        panic!("{}: typed-layout domain drift", inventory_path.display());
    }
    let base_typed_layout_digest = parse_sha256(
        json_string(typed_layout, "base_sha256", &inventory_path),
        "V2 typed-layout base_sha256",
    );
    require_digest_v2(
        base_typed_layout_digest,
        parse_sha256(
            json_string(
                &base_inventory,
                "rust_typed_layout_sha256",
                &base_inventory_path,
            ),
            "V1 inventory rust_typed_layout_sha256",
        ),
        "base typed-layout inventory binding",
    );
    require_digest_v2(
        sha256_bytes(&fs::read(&base_layout_path).unwrap_or_else(|error| {
            panic!("failed to read {}: {error}", base_layout_path.display())
        })),
        base_typed_layout_digest,
        "base typed-layout source digest",
    );
    let overlay_typed_layout_digest = parse_sha256(
        json_string(typed_layout, "overlay_sha256", &inventory_path),
        "V2 typed-layout overlay_sha256",
    );
    require_digest_v2(
        sha256_bytes(&fs::read(&overlay_layout_path).unwrap_or_else(|error| {
            panic!("failed to read {}: {error}", overlay_layout_path.display())
        })),
        overlay_typed_layout_digest,
        "overlay typed-layout source digest",
    );
    let mut composite_hasher = Sha256::new();
    composite_hasher.update(TYPED_LAYOUT_DOMAIN);
    composite_hasher.update(base_typed_layout_digest);
    composite_hasher.update(overlay_typed_layout_digest);
    let composite_typed_layout_digest: [u8; 32] = composite_hasher.finalize().into();
    require_digest_v2(
        composite_typed_layout_digest,
        parse_sha256(
            json_string(typed_layout, "composite_sha256", &inventory_path),
            "V2 typed-layout composite_sha256",
        ),
        "composite typed-layout digest",
    );

    let unchanged = inventory
        .get("unchanged_contracts")
        .unwrap_or_else(|| panic!("{}: missing unchanged_contracts", inventory_path.display()));
    require_exact_object_keys(
        unchanged,
        &[
            "mapping_sha256",
            "action_ref_role_crosswalk_sha256",
            "authoritative_features_sha256",
            "feature_contract_digest",
            "encoding_contract_digest",
        ],
        "V2 unchanged_contracts",
        &inventory_path,
    );
    let mapping_digest = parse_sha256(
        json_string(unchanged, "mapping_sha256", &inventory_path),
        "V2 unchanged mapping_sha256",
    );
    require_digest_v2(
        mapping_digest,
        parse_sha256(
            json_string(&base_goldens, "mapping_sha256", &base_goldens_path),
            "V1 goldens mapping_sha256",
        ),
        "unchanged enum and projection mapping digest",
    );
    require_digest_v2(
        parse_sha256(
            json_string(
                unchanged,
                "action_ref_role_crosswalk_sha256",
                &inventory_path,
            ),
            "V2 unchanged action_ref_role_crosswalk_sha256",
        ),
        action_ref_role_crosswalk_digest,
        "unchanged action-reference role crosswalk digest",
    );
    for field in [
        "authoritative_features_sha256",
        "feature_contract_digest",
        "encoding_contract_digest",
    ] {
        let base_value = parse_sha256(
            json_string(&base_inventory, field, &base_inventory_path),
            &format!("V1 inventory {field}"),
        );
        require_digest_v2(
            parse_sha256(
                json_string(unchanged, field, &inventory_path),
                &format!("V2 unchanged {field}"),
            ),
            base_value,
            &format!("unchanged {field}"),
        );
    }
    require_digest_v2(
        sha256_bytes(
            &fs::read(&features_path).unwrap_or_else(|error| {
                panic!("failed to read {}: {error}", features_path.display())
            }),
        ),
        parse_sha256(
            json_string(unchanged, "authoritative_features_sha256", &inventory_path),
            "V2 unchanged authoritative_features_sha256",
        ),
        "authoritative Python features source digest",
    );

    let versions = inventory
        .get("versions")
        .unwrap_or_else(|| panic!("{}: missing versions", inventory_path.display()));
    for (field, expected) in [
        ("typed_layout", 2),
        ("feature_inventory", 2),
        ("enum_mapping", 1),
        ("object_group_mapping", 1),
        ("relation_role_mapping", 1),
        ("context_subrole_mapping", 1),
        ("scorer_packet", 2),
        ("scorer_action_ref", 2),
        ("scorer_visible_manifest", 2),
        ("action_decision_slice", 2),
        ("action_card_token_mapping", 2),
        ("action_candidate_commitment", 2),
    ] {
        if json_u64(versions, field, &inventory_path) != expected {
            panic!("{}: {field} version drift", inventory_path.display());
        }
    }
    if json_string(&inventory, "visible_manifest", &inventory_path) != VISIBLE_MANIFEST {
        panic!("{}: visible manifest drift", inventory_path.display());
    }
    let blocked_order = inventory.get("blocked_order_contract").unwrap_or_else(|| {
        panic!(
            "{}: missing blocked_order_contract",
            inventory_path.display()
        )
    });
    require_exact_object_keys(
        blocked_order,
        &[
            "type",
            "none",
            "some",
            "invariant",
            "v1_redundant_field_removed",
        ],
        "V2 blocked_order_contract",
        &inventory_path,
    );
    if json_string(blocked_order, "type", &inventory_path) != "Option<u32>"
        || json_string(blocked_order, "none", &inventory_path)
            != "attacker absent from attacker_to_ordered_blockers"
        || json_string(blocked_order, "some", &inventory_path)
            != "exact zero-based mapping index including present-empty"
        || json_string(blocked_order, "invariant", &inventory_path)
            != "present indices unique and contiguous"
        || json_string(blocked_order, "v1_redundant_field_removed", &inventory_path)
            != "was_blocked"
    {
        panic!("{}: blocked_order contract drift", inventory_path.display());
    }

    let topology_audit = inventory
        .get("topology_audit")
        .unwrap_or_else(|| panic!("{}: missing topology_audit", inventory_path.display()));
    if json_string(topology_audit, "path", &inventory_path)
        != "data/flat_policy_v2/ordered_topology_audit_v2.md"
    {
        panic!("{}: topology audit path drift", inventory_path.display());
    }
    require_digest_v2(
        sha256_bytes(&fs::read(&topology_audit_path).unwrap_or_else(|error| {
            panic!("failed to read {}: {error}", topology_audit_path.display())
        })),
        parse_sha256(
            json_string(topology_audit, "source_sha256", &inventory_path),
            "V2 topology audit source_sha256",
        ),
        "ordered topology audit source digest",
    );

    let internal_to_projection = internal_to_projection
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let flat_policy_contract = format!(
        "// @generated by mtg-kernel/build.rs from data/flat_policy_v2; do not edit.\n\
pub const FLAT_POLICY_MAPPING_SHA256_V2: [u8; 32] = {};\n\
pub const FLAT_POLICY_FEATURE_INVENTORY_SHA256_V2: [u8; 32] = {};\n\
pub const FLAT_POLICY_BASE_TYPED_LAYOUT_SHA256_V2: [u8; 32] = {};\n\
pub const FLAT_POLICY_OVERLAY_TYPED_LAYOUT_SHA256_V2: [u8; 32] = {};\n\
pub const FLAT_POLICY_TYPED_LAYOUT_SHA256_V2: [u8; 32] = {};\n\
pub const FLAT_ACTION_REF_PROJECTION_ROLE_MAPPING_VERSION_V2: u32 = {crosswalk_version};\n\
pub const FLAT_ACTION_REF_INTERNAL_ROLE_WIDTH_V2: u8 = {internal_role_width};\n\
pub const FLAT_ACTION_REF_PROJECTION_ROLE_WIDTH_V2: u8 = {projection_role_width};\n\
pub const FLAT_ACTION_REF_INTERNAL_TO_PROJECTION_V2: [u8; {internal_role_width}] = [{internal_to_projection}];\n",
        rust_byte_array(mapping_digest),
        rust_byte_array(inventory_digest),
        rust_byte_array(base_typed_layout_digest),
        rust_byte_array(overlay_typed_layout_digest),
        rust_byte_array(composite_typed_layout_digest),
    );
    let action_domain_bytes = action_commitment_domain.as_bytes();
    let action_domain = action_domain_bytes
        .iter()
        .map(|byte| format!("0x{byte:02x}"))
        .collect::<Vec<_>>()
        .join(", ");
    let flat_action_contract = format!(
        "// @generated by mtg-kernel/build.rs from data/flat_policy_v2/action_contract_v2.json; do not edit.\n\
pub const FLAT_ACTION_CONTRACT_SOURCE_SHA256_V2: [u8; 32] = {};\n\
pub const FLAT_ACTION_CONTRACT_SEMANTIC_SHA256_V2: [u8; 32] = {};\n\
pub const FLAT_ACTION_CANDIDATE_COMMITMENT_DOMAIN_V2: &[u8; {}] = &[{}];\n",
        rust_byte_array(action_contract_source_digest),
        rust_byte_array(action_contract_semantic_digest),
        action_domain_bytes.len(),
        action_domain,
    );
    (flat_policy_contract, flat_action_contract)
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo");
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let data_dir = Path::new(&manifest_dir).join("..").join("data");
    let json_path = data_dir.join("cards_v1.json");
    let runtime_decks_path = data_dir.join("runtime_decks_v1.json");
    println!("cargo:rerun-if-changed={}", json_path.display());
    println!("cargo:rerun-if-changed={}", runtime_decks_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let text = fs::read_to_string(&json_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", json_path.display()));
    let data: CardsFile = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", json_path.display()));

    if data.version != EXPECTED_SCHEMA_VERSION {
        panic!(
            "cards_v1.json schema version mismatch: codegen expects version {EXPECTED_SCHEMA_VERSION},              file has version {}. Update build.rs's CardJson/codegen for the new schema before building.",
            data.version
        );
    }

    let mut seen_names = HashSet::new();
    for c in &data.cards {
        if !seen_names.insert(c.name.clone()) {
            panic!("cards_v1.json: duplicate card name {:?}", c.name);
        }
        if c.decks.is_empty() && !c.is_token {
            panic!("cards_v1.json: card {:?} has empty deck coverage", c.name);
        }
        if c.is_token && c.engine_capability != EngineCapabilityJson::Full {
            panic!(
                "cards_v1.json: token {:?} must be explicitly full so CreateToken cannot materialize unsupported behavior",
                c.name
            );
        }
    }

    let runtime_decks_text = fs::read_to_string(&runtime_decks_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", runtime_decks_path.display()));
    let runtime_decks: RuntimeDeckCatalogJson = serde_json::from_str(&runtime_decks_text)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", runtime_decks_path.display()));
    let runtime_decks_file_sha256 = sha256_hex(sha256_bytes(runtime_decks_text.as_bytes()));

    let out = codegen(&data.cards);
    let runtime_decks_out =
        runtime_decks_codegen(&runtime_decks, &data.cards, &runtime_decks_file_sha256);

    let dest = Path::new(&out_dir).join("card_defs.rs");
    fs::write(&dest, out).unwrap_or_else(|e| panic!("failed to write {}: {e}", dest.display()));
    let runtime_decks_dest = Path::new(&out_dir).join("runtime_decks.rs");
    fs::write(&runtime_decks_dest, runtime_decks_out)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", runtime_decks_dest.display()));
}

fn runtime_decks_codegen(
    catalog: &RuntimeDeckCatalogJson,
    cards: &[CardJson],
    catalog_file_sha256: &str,
) -> String {
    const EXPECTED_SCHEMA: &str = "kernel_runtime_decks/v1";
    const EXPECTED_PROTOCOL: &str = "canonical-mainboard-bo1/v1";
    const EXPECTED_SOURCE_HASH_NORMALIZATION: &str = "utf8_text_crlf_v1";
    const EXPECTED_MATERIALIZATION: &str = "xmage_xml_row_then_copy_ordinal/v1";
    const EXPECTED_CARD_ID_ASSIGNMENT: &str = "zero_based_data_cards_v1_json_cards_array_index/v1";
    const EXPECTED_DECK_HASH_ALGORITHM: &str = "fnv1a64-serde-json-u16-array/v1";
    const EXPECTED_DECKS: [(&str, u32, &str, &str, usize, u64); 9] = [
        (
            "Wildfire",
            1,
            "oracle/xmage/decks/Pauper/Deck - Jund Wildfire.dek",
            "cff35798ff724888a9e5a4520dd55e70b0c628a55908697aa116089d8fd980a5",
            22,
            0x552acb5fc9631d3b,
        ),
        (
            "Rally",
            2,
            "oracle/xmage/decks/Pauper/Deck - Mono Red Rally.dek",
            "4b5019bd08f9387aeabebdca0d90aaa10dfd75fc75ed3a87c95a2fabf4dba834",
            14,
            0x0c9f01c2544412bf,
        ),
        (
            "Affinity",
            3,
            "oracle/xmage/decks/Pauper/Deck - Grixis Affinity.dek",
            "4a41135ac6d14960e75ddce8e9980c0505c0b71a9c08a2e10578a10d2fcf8801",
            22,
            0xff4bf00deadf8821,
        ),
        (
            "Elves",
            4,
            "oracle/xmage/decks/Pauper/Deck - Elves.dek",
            "6b040933c9b3506536e7dc71c94dcaf5f16c7ade43a3d0f7f9b240be6deb0d87",
            14,
            0x6a187257d6d37346,
        ),
        (
            "Spy",
            5,
            "oracle/xmage/decks/Pauper/Deck - Spy Combo.dek",
            "f08177d5ed133b18312f59649d1155e15b5074ababeaabcdf3f31ded650308ba",
            21,
            0xcd2afdfee3573675,
        ),
        (
            "Burn",
            6,
            "oracle/xmage/decks/Pauper/Deck - Mono-Red Burn.dek",
            "4ebba6b42bb27a0ea55001cee133aada81f0dffd8661b46b012fc5026675aa32",
            12,
            0x5fdb7b92986b6fc1,
        ),
        (
            "Terror",
            7,
            "oracle/xmage/decks/Pauper/Deck - Mono-Blue Terror.dek",
            "8ba22b67b843bc49a421e1c2814c4dd24a04ab2b45131ec7876a8312115a9fda",
            14,
            0xfd6f1a4aceaa157b,
        ),
        (
            "CawGates",
            8,
            "oracle/xmage/decks/Pauper/Deck - Caw-Gates.dek",
            "72c2bbf76a7fd219349a0ad81c44dc6166b4a797a1f66fe9b5a5de79aa6cdc14",
            20,
            0x25c1916a4d20c08e,
        ),
        (
            "Faeries",
            9,
            "oracle/xmage/decks/Pauper/Deck - Mono-Blue Faeries.dek",
            "8cb962c4ccee6a5f8c0c70fc27c17d13323d13606c82b9b12b8985aa87e0f344",
            14,
            0xd7a47ab2fa78dbaa,
        ),
    ];

    if catalog.schema != EXPECTED_SCHEMA {
        panic!(
            "runtime_decks_v1.json: schema mismatch: expected {EXPECTED_SCHEMA:?}, got {:?}",
            catalog.schema
        );
    }
    if catalog.protocol != EXPECTED_PROTOCOL {
        panic!(
            "runtime_decks_v1.json: protocol mismatch: expected {EXPECTED_PROTOCOL:?}, got {:?}",
            catalog.protocol
        );
    }
    if catalog.source_hash_normalization != EXPECTED_SOURCE_HASH_NORMALIZATION {
        panic!(
            "runtime_decks_v1.json: source hash normalization mismatch: expected {EXPECTED_SOURCE_HASH_NORMALIZATION:?}, got {:?}",
            catalog.source_hash_normalization
        );
    }
    if catalog.materialization.order != EXPECTED_MATERIALIZATION
        || catalog.materialization.source_row_ordinal_base != 1
        || catalog.materialization.copy_ordinal_base != 1
    {
        panic!(
            "runtime_decks_v1.json: unsupported materialization contract {:?} with row/copy bases {}/{}",
            catalog.materialization.order,
            catalog.materialization.source_row_ordinal_base,
            catalog.materialization.copy_ordinal_base
        );
    }
    if catalog.card_ids.assignment != EXPECTED_CARD_ID_ASSIGNMENT
        || catalog.card_ids.deck_hash_algorithm != EXPECTED_DECK_HASH_ALGORITHM
    {
        panic!(
            "runtime_decks_v1.json: unsupported card-id/hash contract {:?} / {:?}",
            catalog.card_ids.assignment, catalog.card_ids.deck_hash_algorithm
        );
    }
    if catalog.decks.len() != EXPECTED_DECKS.len() {
        panic!(
            "runtime_decks_v1.json: expected exactly {} runnable decks, got {}",
            EXPECTED_DECKS.len(),
            catalog.decks.len()
        );
    }

    let mut generated_decks = Vec::with_capacity(catalog.decks.len());
    let mut seen_ids = HashSet::new();
    for (
        deck,
        &(
            expected_id,
            expected_pool_order,
            expected_path,
            expected_sha,
            expected_unique_count,
            expected_hash,
        ),
    ) in catalog.decks.iter().zip(EXPECTED_DECKS.iter())
    {
        if !seen_ids.insert(deck.id.as_str()) {
            panic!("runtime_decks_v1.json: duplicate deck id {:?}", deck.id);
        }
        if deck.id != expected_id
            || deck.canonical_pool_order != expected_pool_order
            || deck.source_path != expected_path
            || deck.source_sha256 != expected_sha
        {
            panic!(
                "runtime_decks_v1.json: deck slot for {expected_id:?} does not match its frozen id/order/source provenance"
            );
        }
        if deck.source_sha256.len() != 64
            || !deck
                .source_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            panic!(
                "runtime_decks_v1.json: deck {:?} source_sha256 is not 64 lowercase hexadecimal characters",
                deck.id
            );
        }
        if deck.mainboard_copy_count != 60
            || deck.materialized_mainboard.len() != deck.mainboard_copy_count
        {
            panic!(
                "runtime_decks_v1.json: deck {:?} must materialize exactly 60 mainboard copies, declared {}, found {}",
                deck.id,
                deck.mainboard_copy_count,
                deck.materialized_mainboard.len()
            );
        }

        let mut last_row = 0u32;
        let mut expected_copy_ordinal = 0u32;
        let mut current_row_identity: Option<(&str, u16)> = None;
        let mut unique_names = HashSet::new();
        let mut card_ids = Vec::with_capacity(deck.mainboard_copy_count);
        for (materialized_index, copy) in deck.materialized_mainboard.iter().enumerate() {
            if copy.source_row_ordinal == 0 || copy.copy_ordinal == 0 {
                panic!(
                    "runtime_decks_v1.json: deck {:?} materialized copy {materialized_index} uses a zero ordinal",
                    deck.id
                );
            }
            if copy.source_row_ordinal == last_row {
                expected_copy_ordinal += 1;
                let (row_name, row_card_id) = current_row_identity
                    .expect("a repeated materialized row has a first copy identity");
                if copy.name != row_name || copy.card_id != row_card_id {
                    panic!(
                        "runtime_decks_v1.json: deck {:?} row {} changes card identity from {:?}/{} to {:?}/{}",
                        deck.id,
                        copy.source_row_ordinal,
                        row_name,
                        row_card_id,
                        copy.name,
                        copy.card_id
                    );
                }
            } else {
                if copy.source_row_ordinal <= last_row {
                    panic!(
                        "runtime_decks_v1.json: deck {:?} materialized rows are not strictly ordered at copy {materialized_index}",
                        deck.id
                    );
                }
                last_row = copy.source_row_ordinal;
                expected_copy_ordinal = 1;
                current_row_identity = Some((copy.name.as_str(), copy.card_id));
            }
            if copy.copy_ordinal != expected_copy_ordinal {
                panic!(
                    "runtime_decks_v1.json: deck {:?} row {} copy ordinal {}, expected {}",
                    deck.id, copy.source_row_ordinal, copy.copy_ordinal, expected_copy_ordinal
                );
            }

            let card_index = usize::from(copy.card_id);
            let Some(card) = cards.get(card_index) else {
                panic!(
                    "runtime_decks_v1.json: deck {:?} materialized copy {materialized_index} references unknown card id {}",
                    deck.id, copy.card_id
                );
            };
            if card.name != copy.name {
                panic!(
                    "runtime_decks_v1.json: deck {:?} materialized copy {materialized_index} card id {} resolves to {:?}, not {:?}",
                    deck.id, copy.card_id, card.name, copy.name
                );
            }
            if card.is_token || card.engine_capability != EngineCapabilityJson::Full {
                panic!(
                    "runtime_decks_v1.json: deck {:?} materialized copy {materialized_index} ({:?}) is not a full, non-token card definition",
                    deck.id, copy.name
                );
            }
            unique_names.insert(copy.name.as_str());
            card_ids.push(copy.card_id);
        }
        if deck.unique_mainboard_cards != expected_unique_count
            || unique_names.len() != expected_unique_count
        {
            panic!(
                "runtime_decks_v1.json: deck {:?} must have exactly {} unique mainboard cards, declared {}, materialization has {}",
                deck.id,
                expected_unique_count,
                deck.unique_mainboard_cards,
                unique_names.len()
            );
        }

        let declared_hash = parse_runtime_deck_hash(&deck.id, &deck.runtime_deck_hash);
        let serialized_ids = serde_json::to_vec(&card_ids)
            .expect("runtime deck card-id sequence serializes as JSON");
        let computed_hash = fnv1a64(&serialized_ids);
        if declared_hash != expected_hash || computed_hash != expected_hash {
            panic!(
                "runtime_decks_v1.json: deck {:?} hash mismatch: declared {declared_hash:#018x}, computed {computed_hash:#018x}, expected {expected_hash:#018x}",
                deck.id
            );
        }
        generated_decks.push((deck, card_ids, declared_hash));
    }

    let mut out = String::new();
    writeln!(
        out,
        "// GENERATED by build.rs from data/runtime_decks_v1.json. Do not edit by hand."
    )
    .unwrap();
    writeln!(out).unwrap();
    for (name, value) in [
        ("RUNTIME_DECK_CATALOG_SCHEMA", catalog.schema.as_str()),
        ("RUNTIME_DECK_PROTOCOL", catalog.protocol.as_str()),
        (
            "RUNTIME_DECK_SOURCE_HASH_NORMALIZATION",
            catalog.source_hash_normalization.as_str(),
        ),
        (
            "RUNTIME_DECK_MATERIALIZATION_PROTOCOL",
            catalog.materialization.order.as_str(),
        ),
        (
            "RUNTIME_CARD_ID_ASSIGNMENT",
            catalog.card_ids.assignment.as_str(),
        ),
        (
            "RUNTIME_DECK_HASH_ALGORITHM",
            catalog.card_ids.deck_hash_algorithm.as_str(),
        ),
    ] {
        writeln!(out, "pub const {name}: &str = {value:?};").unwrap();
    }
    writeln!(
        out,
        "pub const RUNTIME_DECK_CATALOG_FILE_SHA256: &str = {catalog_file_sha256:?};"
    )
    .unwrap();
    writeln!(
        out,
        "pub const RUNTIME_DECK_SOURCE_ROW_ORDINAL_BASE: u32 = {};",
        catalog.materialization.source_row_ordinal_base
    )
    .unwrap();
    writeln!(
        out,
        "pub const RUNTIME_DECK_COPY_ORDINAL_BASE: u32 = {};",
        catalog.materialization.copy_ordinal_base
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "pub static RUNTIME_DECKS: &[RuntimeDeckDefinition] = &["
    )
    .unwrap();
    for (deck, card_ids, runtime_deck_hash) in generated_decks {
        writeln!(out, "    RuntimeDeckDefinition {{").unwrap();
        writeln!(
            out,
            "        canonical_pool_order: {},",
            deck.canonical_pool_order
        )
        .unwrap();
        writeln!(out, "        id: {:?},", deck.id).unwrap();
        writeln!(out, "        source_path: {:?},", deck.source_path).unwrap();
        writeln!(out, "        source_sha256: {:?},", deck.source_sha256).unwrap();
        writeln!(
            out,
            "        mainboard_count: {},",
            deck.mainboard_copy_count
        )
        .unwrap();
        writeln!(out, "        runtime_deck_hash: {runtime_deck_hash:#018x},").unwrap();
        write!(out, "        card_ids: &[").unwrap();
        for (index, card_id) in card_ids.iter().enumerate() {
            if index != 0 {
                write!(out, ", ").unwrap();
            }
            write!(out, "{card_id}u16").unwrap();
        }
        writeln!(out, "],").unwrap();
        writeln!(out, "    }},").unwrap();
    }
    writeln!(out, "];").unwrap();
    out
}

fn parse_runtime_deck_hash(deck_id: &str, value: &str) -> u64 {
    let Some(digits) = value.strip_prefix("0x") else {
        panic!("runtime_decks_v1.json: deck {deck_id:?} hash must start with 0x");
    };
    if digits.len() != 16
        || !digits
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        panic!(
            "runtime_decks_v1.json: deck {deck_id:?} hash must contain exactly 16 lowercase hexadecimal digits"
        );
    }
    u64::from_str_radix(digits, 16)
        .unwrap_or_else(|_| panic!("runtime_decks_v1.json: deck {deck_id:?} hash is invalid"))
}

/// Generated special-case recipes shared by the fully supported nine-deck
/// Pauper pool. Cards without a special recipe are still admitted only through
/// their explicit full-capability registry metadata and generated definitions.
#[derive(Clone, Copy)]
enum Special {
    None,
    /// Great Furnace's explicit `{T}: Add {R}` program. Basic-land mana is
    /// not a name special: it is derived from Basic + Land + one
    /// `produces_mana` color in `codegen`.
    GreatFurnace,
    /// A plain "draw N cards" spell. Lorien Revealed is the first consumer;
    /// keeping the generated recipe parameterized avoids a runtime card-name
    /// special while leaving room for later draw spells to share it.
    DrawCards(u8),
    /// Deals `amount` damage to any target (Lightning Bolt, Fiery Temper,
    /// Fireblast, Lava Dart). Fireblast's/Lava Dart's real alt cost /
    /// flashback, and Fiery Temper's Madness, are modeled via the separate
    /// `alt_cost_for`/`flashback_for`/`madness_cost_for` tables below
    /// (independent of `Special`, since a card's targeting/damage shape and
    /// its cost shape are orthogonal).
    BurnAnyTarget(i32),
    /// "Deals 3 damage to any target. Then that player or that permanent's
    /// controller may pay {R}{R}. If the player does, they may copy this
    /// spell and may choose a new target for that copy." (Chain Lightning,
    /// Rally-only). The mandatory damage is byte-for-byte
    /// `BurnAnyTarget(3)`'s shape; the optional-copy continuation suspends
    /// resolution in the engine's dedicated payment/copy/retarget state
    /// machine -- see `effect::EffectOp::OfferAffectedPlayerSpellCopy`.
    ChainLightning,
    /// Target player draws `draw` cards. Deep Analysis is the first consumer;
    /// `PlayerRef::Target(0)` keeps self- and opponent-targeting on the same
    /// generic effect path.
    TargetPlayerDraw {
        draw: u8,
    },
    /// "Draw `draw` cards, then discard `discard` cards" (Faithless
    /// Looting: 2 and 2). The discard is a resolution effect (not a cost),
    /// so it's `EffectOp::DiscardCards`, staged via
    /// `engine::EngineState::pending_discard` same as cleanup.
    DrawThenDiscard {
        draw: i32,
        discard: i32,
    },
    /// Each opponent sacrifices a creature, restricted to greatest power
    /// when this exact cast collected evidence.
    ExtractAConfession,
    /// Grab the Prize: draw two cards, then (if the card discarded to pay
    /// the mandatory additional cost -- see `additional_cost_for` --
    /// wasn't a land) deal 2 damage to the opponent.
    GrabThePrize,
    /// "You may discard a card or sacrifice a land. If you do, draw two
    /// cards." (Highway Robbery's resolution effect; its Plot ability is
    /// modeled separately via `plot_cost_for`, since Plot is a cast-time
    /// alternative, not a resolution effect).
    HighwayRobbery,
    /// 1 damage to target player/planeswalker's controller and 1 damage to
    /// a creature that player controls; 3/3 instead with landfall this
    /// turn (Searing Blaze).
    SearingBlaze,
    /// Counter target spell, with the target pool selected independently
    /// from the shared generated counter effect.
    CounterTarget(StackSpellFilter),
    /// Counter a target spell unless its controller pays a fixed generic
    /// amount. Target eligibility remains definition data rather than a
    /// runtime card-name branch.
    CounterUnlessPaysGeneric {
        filter: StackSpellFilter,
        generic: u8,
    },
    /// Choose one: counter target artifact spell, or return target artifact
    /// permanent to its owner's hand.
    SteelSabotage,
    /// Choose one: grant target creature islandwalk; give target creature
    /// +2/-1; or have target player discard a card.
    PiracyCharm,
    /// Choose one: deal one damage to each of up to two target creatures, or
    /// exile target artifact.
    CastIntoTheFire,
    /// Exile exactly two target artifact permanents.
    DustToDust,
    /// Choose one: dynamic creature-count damage, destroy an enchantment, or
    /// exile any number of target players' graveyards.
    ThrabenCharm,
    /// Symmetric Elemental Blast recipe. `checked_color` is the color the
    /// target must have; `filter_timing` distinguishes the Elemental Blasts'
    /// targeting restriction from Pyroblast/Hydroblast's resolution-time
    /// "if it's [color]" check without introducing card-named runtime logic.
    ColorBlast {
        checked_color: BlastColor,
        filter_timing: BlastFilterTiming,
    },
    /// "Deals 1 damage to each opponent and each creature and planeswalker
    /// they control." (End the Festivities, Rally-only). No planeswalker
    /// card exists in this 132-card pool, so the planeswalker half of the
    /// text is vacuously satisfied by construction, not modeled separately
    /// -- see `EffectOp::DamageOpponentAndTheirCreatures`.
    EndTheFestivities,
    /// "Deals 2 damage to any target. Metalcraft -- 4 instead if you
    /// control three or more artifacts." (Galvanic Blast, Rally-only).
    GalvanicBlast,
    /// "Create two 1/1 white Human Soldier creature tokens. Humans you
    /// control gain haste until end of turn." (Rally at the Hornburg,
    /// Rally-only -- the card the deck is named for).
    RallyAtTheHornburg,
    /// "Exile the top two cards of your library. Until the end of your next
    /// turn, you may play those cards." (Reckless Impulse, Rally-only).
    RecklessImpulse,
    /// "Choose creature or land. Reveal the top four cards of your
    /// library. Put all cards of the chosen type revealed this way into
    /// your hand and the rest into your graveyard." The choice happens
    /// during resolution, so it is the first real consumer of the generic
    /// resumable `EffectOp::Choice` interpreter.
    WindingWay,
    /// Privately look at the top `look` cards, choose any number of cards of
    /// `card_type` for hand, and put the rest on the bottom in any order.
    LookTopSelectByTypeToHandBottomRest {
        look: u8,
        card_type: &'static str,
    },
    /// Mill `mill` cards from `player`'s library, then draw `draw` cards.
    /// Mental Note mills its controller without targeting; Thought Scour
    /// mills its chosen player in target slot zero. Both use the same
    /// generated recipe so the runtime behavior is not card-name-specific.
    MillThenDraw {
        player: MillPlayer,
        mill: u8,
        draw: u8,
    },
    /// Privately look at and reorder the top `look` cards of the controller's
    /// library, optionally shuffle that library, then draw `draw` cards.
    /// Ponder is the first consumer; keeping this as a parameterized recipe
    /// prevents card-name behavior from leaking into the runtime engine.
    LookReorderMayShuffleThenDraw {
        look: u8,
        draw: u8,
    },
    /// Draw `draw` cards, then put up to `put` cards from the controller's
    /// hand on top of their library through sequential private choices.
    /// Brainstorm is the first consumer; the engine owns the generic hand and
    /// library semantics rather than branching on the printed card name.
    DrawThenPutHandOnLibraryTop {
        draw: u8,
        put: u8,
    },
    /// Privately scry `scry`, then draw `draw` cards. Preordain is the first
    /// consumer; the engine owns subset selection, both ordering directions,
    /// hidden information, and the atomic final library transition.
    ScryThenDraw {
        scry: u8,
        draw: u8,
    },
    /// Target nonland permanent; its owner chooses whether the exact bound
    /// incarnation goes second from top or on the bottom of their library.
    DeemInferior,
    /// Tap target creature and mark that exact battlefield incarnation to
    /// skip its current controller's next untap. Sleep of the Dead is the
    /// first consumer.
    TapAndSkipNextUntap,
    /// Destroy target nonlegendary creature. Cast Down is the first
    /// consumer of the append-only target filter and shared destroy leaf.
    DestroyNonlegendaryCreature,
    /// Return target creature or land card from a graveyard to its owner's
    /// hand, then the controller gains a fixed amount of life. Pulse of
    /// Murasa is the first consumer.
    ReturnCreatureOrLandFromGraveyardAndGainLife {
        amount: u8,
    },
    /// Deal one simultaneous damage batch to every creature other than the
    /// excluded subtype. Breath Weapon excludes Dragons and deals two.
    DamageEachCreatureWithoutSubtype {
        amount: i32,
        excluded_subtype: &'static str,
    },
    /// Draw cards, then create one named token.
    DrawThenCreateToken {
        draw: u8,
        token: &'static str,
    },
    /// Gain life equal to the frozen mana value of the paid sacrifice, then
    /// draw cards.
    GainPaidCostManaValueThenDraw {
        draw: u8,
    },
    /// Return target creature card from the controller's graveyard to the
    /// battlefield. Dread Return's flashback cost is modeled independently.
    ReturnOwnGraveyardCreatureToBattlefield,
    /// Search the controller's library for a Forest card, reveal it, put it
    /// into hand, then shuffle. Land Grant's conditional hand-reveal
    /// alternative cost is modeled independently.
    SearchForestToHand,
    /// Put one +1/+1 counter and one lifelink keyword counter on target
    /// creature. Unexpected Fangs is the first consumer.
    AddPlusOnePlusOneAndLifelinkCounters,
    /// Aura permanent spell that enters attached to its creature target.
    BindTheMonster,
    /// Return target creature, then choose up to two controlled lands to
    /// untap during the same resolution.
    Snap,
    /// Damage cannot be prevented this turn. Flaring Pain's flashback cost
    /// is modeled independently by `flashback_for`.
    DamageCannotBePreventedThisTurn,
    /// Choose a color, then prevent all damage from sources of that color
    /// this turn. Its flashback tap cost is modeled independently.
    PrismaticStrands,
    /// Destroy target land, offer its resolution-time controller an optional
    /// basic-land search to the battlefield tapped, then draw a card.
    CleansingWildfire,
    /// Reveal target opponent's hand and choose a noncreature, nonland card
    /// from it for that player to discard.
    Duress,
    /// Target creature gains deathtouch and lifelink until end of turn, then
    /// the controller investigates.
    ToxinAnalysis,
    /// Gain three life. The card's CastSelf Storm trigger is defined in the
    /// shared trigger table.
    WeatherTheStorm,
    /// Deal damage to target creature equal to the resolution-time power,
    /// or last known battlefield power, of the creature chosen or revealed
    /// for the spell's mandatory additional cost.
    MonstrousEmergence,
    /// An X creature that enters with X counters. Bestow is modeled by the
    /// independently generated `CardDef::bestow` characteristics.
    NyxbornHydra,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MillPlayer {
    Controller,
    Target0,
}

#[derive(Clone, Copy)]
enum StackSpellFilter {
    Any,
    Instant,
    ArtifactOrEnchantment,
    Sorcery,
    Noncreature,
    Artifact,
}

impl MillPlayer {
    fn canonical_token(self) -> &'static str {
        match self {
            MillPlayer::Controller => "controller",
            MillPlayer::Target0 => "target0",
        }
    }
}

impl StackSpellFilter {
    fn canonical_token(self) -> &'static str {
        match self {
            StackSpellFilter::Any => "any",
            StackSpellFilter::Instant => "instant",
            StackSpellFilter::ArtifactOrEnchantment => "artifact_or_enchantment",
            StackSpellFilter::Sorcery => "sorcery",
            StackSpellFilter::Noncreature => "noncreature",
            StackSpellFilter::Artifact => "artifact",
        }
    }

    fn target_spec(self) -> &'static str {
        match self {
            StackSpellFilter::Any => "TargetSpec::AnySpellOnStack",
            StackSpellFilter::Instant => "TargetSpec::InstantSpellOnStack",
            StackSpellFilter::ArtifactOrEnchantment => {
                "TargetSpec::ArtifactOrEnchantmentSpellOnStack"
            }
            StackSpellFilter::Sorcery => "TargetSpec::SorcerySpellOnStack",
            StackSpellFilter::Noncreature => "TargetSpec::NoncreatureSpellOnStack",
            StackSpellFilter::Artifact => "TargetSpec::ArtifactSpellOnStack",
        }
    }
}

#[derive(Clone, Copy)]
enum BlastColor {
    Blue,
    Red,
}

impl BlastColor {
    fn mana_variant(self) -> &'static str {
        match self {
            BlastColor::Blue => "ManaColor::U",
            BlastColor::Red => "ManaColor::R",
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            BlastColor::Blue => "blue",
            BlastColor::Red => "red",
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            BlastColor::Blue => "U",
            BlastColor::Red => "R",
        }
    }

    fn filtered_spell_target_spec(self) -> &'static str {
        match self {
            BlastColor::Blue => "TargetSpec::BlueSpellOnStack",
            BlastColor::Red => "TargetSpec::RedSpellOnStack",
        }
    }

    fn filtered_permanent_target_spec(self) -> &'static str {
        match self {
            BlastColor::Blue => "TargetSpec::BluePermanent",
            BlastColor::Red => "TargetSpec::RedPermanent",
        }
    }
}

#[derive(Clone, Copy)]
enum BlastFilterTiming {
    Targeting,
    Resolution,
}

impl BlastFilterTiming {
    fn canonical_token(self) -> &'static str {
        match self {
            BlastFilterTiming::Targeting => "targeting",
            BlastFilterTiming::Resolution => "resolution",
        }
    }
}

impl Special {
    /// Stable semantic token derived from the same recipe enum that drives
    /// code generation. This deliberately avoids hashing emitted Rust text:
    /// formatting-only codegen edits must not churn the card-database hash.
    fn canonical_token(self) -> String {
        match self {
            Special::None => "none".to_string(),
            Special::GreatFurnace => "great_furnace:add_r".to_string(),
            Special::DrawCards(count) => format!("draw_cards:{count}"),
            Special::BurnAnyTarget(amount) => format!("burn_any_target:{amount}"),
            Special::ChainLightning => "chain_lightning".to_string(),
            Special::TargetPlayerDraw { draw } => format!("target_player_draw:{draw}"),
            Special::DrawThenDiscard { draw, discard } => {
                format!("draw_then_discard:{draw}:{discard}")
            }
            Special::ExtractAConfession => "extract_a_confession".to_string(),
            Special::GrabThePrize => "grab_the_prize".to_string(),
            Special::HighwayRobbery => "highway_robbery".to_string(),
            Special::SearingBlaze => "searing_blaze".to_string(),
            Special::CounterTarget(filter) => {
                format!("counter_target:{}", filter.canonical_token())
            }
            Special::CounterUnlessPaysGeneric { filter, generic } => format!(
                "counter_unless_pays_generic:{}:{generic}",
                filter.canonical_token()
            ),
            Special::SteelSabotage => "steel_sabotage:counter_or_bounce_artifact".to_string(),
            Special::PiracyCharm => "piracy_charm:islandwalk_or_pump_or_discard".to_string(),
            Special::CastIntoTheFire => {
                "cast_into_the_fire:damage_up_to_two_creatures_or_exile_artifact".to_string()
            }
            Special::DustToDust => "dust_to_dust:exile_exactly_two_artifacts".to_string(),
            Special::ThrabenCharm => {
                "thraben_charm:creature_count_damage_or_destroy_enchantment_or_exile_target_graveyards".to_string()
            }
            Special::ColorBlast {
                checked_color,
                filter_timing,
            } => format!(
                "color_blast:{}:{}",
                checked_color.suffix(),
                filter_timing.canonical_token()
            ),
            Special::EndTheFestivities => "end_the_festivities".to_string(),
            Special::GalvanicBlast => "galvanic_blast".to_string(),
            Special::RallyAtTheHornburg => "rally_at_the_hornburg".to_string(),
            Special::RecklessImpulse => "reckless_impulse".to_string(),
            Special::WindingWay => "winding_way".to_string(),
            Special::LookTopSelectByTypeToHandBottomRest { look, card_type } => {
                format!("look_top_select_by_type_to_hand_bottom_rest:{look}:{card_type}")
            }
            Special::MillThenDraw { player, mill, draw } => {
                format!("mill_then_draw:{}:{mill}:{draw}", player.canonical_token())
            }
            Special::LookReorderMayShuffleThenDraw { look, draw } => {
                format!("look_reorder_may_shuffle_then_draw:{look}:{draw}")
            }
            Special::DrawThenPutHandOnLibraryTop { draw, put } => {
                format!("draw_then_put_hand_on_library_top:{draw}:{put}")
            }
            Special::ScryThenDraw { scry, draw } => {
                format!("scry_then_draw:{scry}:{draw}")
            }
            Special::DeemInferior => "deem_inferior".to_string(),
            Special::TapAndSkipNextUntap => "tap_and_skip_next_untap".to_string(),
            Special::DestroyNonlegendaryCreature => "destroy_nonlegendary_creature".to_string(),
            Special::ReturnCreatureOrLandFromGraveyardAndGainLife { amount } => {
                format!("return_creature_or_land_from_graveyard_and_gain_life:{amount}")
            }
            Special::DamageEachCreatureWithoutSubtype {
                amount,
                excluded_subtype,
            } => format!("damage_each_creature_without_subtype:{amount}:{excluded_subtype}"),
            Special::DrawThenCreateToken { draw, token } => {
                format!("draw_then_create_token:{draw}:{token}")
            }
            Special::GainPaidCostManaValueThenDraw { draw } => {
                format!("gain_paid_cost_mana_value_then_draw:{draw}")
            }
            Special::ReturnOwnGraveyardCreatureToBattlefield => {
                "return_own_graveyard_creature_to_battlefield".to_string()
            }
            Special::SearchForestToHand => "search_forest_to_hand".to_string(),
            Special::AddPlusOnePlusOneAndLifelinkCounters => {
                "add_plus_one_plus_one_and_lifelink_counters".to_string()
            }
            Special::BindTheMonster => "bind_the_monster:aura_creature".to_string(),
            Special::Snap => "snap:bounce_then_untap_up_to_two_lands".to_string(),
            Special::DamageCannotBePreventedThisTurn => {
                "damage_cannot_be_prevented_this_turn".to_string()
            }
            Special::PrismaticStrands => {
                "prismatic_strands:choose_color_prevent_source_damage_this_turn".to_string()
            }
            Special::CleansingWildfire => {
                "cleansing_wildfire:destroy_land_optional_basic_tapped_draw".to_string()
            }
            Special::Duress => "duress:reveal_opponent_hand_choose_noncreature_nonland_discard"
                .to_string(),
            Special::ToxinAnalysis => {
                "toxin_analysis:deathtouch_lifelink_eot_investigate".to_string()
            }
            Special::WeatherTheStorm => "weather_the_storm:gain_three:storm".to_string(),
            Special::MonstrousEmergence => {
                "monstrous_emergence:chosen_creature_power_damage".to_string()
            }
            Special::NyxbornHydra => "nyxborn_hydra:x_counters_bestow".to_string(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AbilityCostRecipe {
    Mana {
        colored: Option<&'static str>,
        generic: u8,
    },
    VariableMana {
        generic: u8,
        x_count: u8,
    },
    Tap,
    DiscardCards(u8),
    DiscardSelf,
    SacrificeSelf,
    ReturnControlledLandWithSubtype(&'static str),
    ExileSelf,
    TapOtherUntappedControlledPermanentWithSubtype(&'static str),
    SacrificeControlled {
        count: u8,
        filter: PermanentFilterRecipe,
    },
    ReturnControlledUnblockedAttacker,
    /// An arbitrary printed mana cost parsed by the same canonical cost
    /// grammar as spell costs. Twisted Landscape's Cycling is the first
    /// multicolor consumer.
    ManaCost(&'static str),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AbilityEffectRecipe {
    DrawCards(u8),
    GainLife(u8),
    CreateToken(&'static str),
    DamageTarget(u8),
    MoveAllTargetsToHand,
    ExploreTarget,
    DealDamageAnyTarget(i32),
    DamageAllCreatures {
        amount: i32,
        filter: CreatureEffectFilterRecipe,
    },
    ExileTargetPlayersGraveyard,
    ExileOneFromTargetPlayersGraveyard,
    ExileAllGraveyardsThenDraw(u8),
    /// The interpreter currently supports exactly this typecycling search
    /// contract. Keeping all semantic knobs in the recipe makes codegen fail
    /// closed if a future caller asks for a different cardinality/reveal/
    /// shuffle shape without first extending `EffectOp`.
    SearchLibraryToHand {
        filter: LibrarySearchFilterRecipe,
        min_targets: u8,
        max_targets: u8,
        reveal_selected: bool,
        shuffle: bool,
    },
    UntapTarget,
    GainLifeBattlefieldSubtypeCount(&'static str),
    PumpTargetBattlefieldSubtypeCount(&'static str),
    PumpTargetByControlledSubtypeCount(&'static str),
    AttachSourceToTarget,
    AddStunCounterToOptionalTarget,
    DestroyTarget,
    DrawThenDiscard {
        draw: u8,
        discard: u8,
    },
    PutSourceOntoBattlefieldTappedAndAttacking,
    MoveAllTargetsToExile,
    AddMinusOneMinusOneCounter,
    SearchLibraryToBattlefieldTapped {
        filter: LibrarySearchFilterRecipe,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PermanentFilterRecipe {
    Artifact,
    ArtifactOrCreature,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CreatureEffectFilterRecipe {
    WithoutKeyword(&'static str),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LibrarySearchFilterRecipe {
    LandWithSubtype(&'static str),
    BasicLand,
    BasicLandWithAnySubtype([&'static str; 3]),
}

#[derive(Clone, Copy)]
struct ActivatedAbilityRecipe {
    cost: &'static [AbilityCostRecipe],
    effect: AbilityEffectRecipe,
    activation_zone: &'static str,
    sorcery_speed_only: bool,
    target_spec: &'static str,
    activation_target_filter: &'static str,
    max_activations_per_turn: Option<u8>,
}

fn special_for(name: &str) -> Special {
    match name {
        // Great Furnace is intentionally explicit: unlike a basic land, its
        // mana ability is rules text, not intrinsic to a basic land type.
        "Great Furnace" => Special::GreatFurnace,
        "Lorien Revealed" => Special::DrawCards(3),
        "Thoughtcast" => Special::DrawCards(2),
        "Of One Mind" => Special::DrawCards(2),
        "Eviscerator's Insight" => Special::DrawCards(2),
        "Fanatical Offering" => Special::DrawThenCreateToken {
            draw: 2,
            token: "Map Token",
        },
        "Reckoner's Bargain" => Special::GainPaidCostManaValueThenDraw { draw: 2 },
        "Lightning Bolt" => Special::BurnAnyTarget(3),
        "Fiery Temper" => Special::BurnAnyTarget(3),
        "Fireblast" => Special::BurnAnyTarget(4),
        "Lava Dart" => Special::BurnAnyTarget(1),
        "Gut Shot" => Special::BurnAnyTarget(1),
        "Chain Lightning" => Special::ChainLightning,
        "Deep Analysis" => Special::TargetPlayerDraw { draw: 2 },
        "Faithless Looting" => Special::DrawThenDiscard {
            draw: 2,
            discard: 2,
        },
        "Extract a Confession" => Special::ExtractAConfession,
        "Grab the Prize" => Special::GrabThePrize,
        "Highway Robbery" => Special::HighwayRobbery,
        "Searing Blaze" => Special::SearingBlaze,
        "Counterspell" => Special::CounterTarget(StackSpellFilter::Any),
        "Dispel" => Special::CounterTarget(StackSpellFilter::Instant),
        "Annul" => Special::CounterTarget(StackSpellFilter::ArtifactOrEnchantment),
        "Envelop" => Special::CounterTarget(StackSpellFilter::Sorcery),
        "Force Spike" => Special::CounterUnlessPaysGeneric {
            filter: StackSpellFilter::Any,
            generic: 1,
        },
        "Spell Pierce" => Special::CounterUnlessPaysGeneric {
            filter: StackSpellFilter::Noncreature,
            generic: 2,
        },
        "Steel Sabotage" => Special::SteelSabotage,
        "Piracy Charm" => Special::PiracyCharm,
        "Cast into the Fire" => Special::CastIntoTheFire,
        "Dust to Dust" => Special::DustToDust,
        "Thraben Charm" => Special::ThrabenCharm,
        "Blue Elemental Blast" => Special::ColorBlast {
            checked_color: BlastColor::Red,
            filter_timing: BlastFilterTiming::Targeting,
        },
        "Hydroblast" => Special::ColorBlast {
            checked_color: BlastColor::Red,
            filter_timing: BlastFilterTiming::Resolution,
        },
        "Pyroblast" => Special::ColorBlast {
            checked_color: BlastColor::Blue,
            filter_timing: BlastFilterTiming::Resolution,
        },
        "Red Elemental Blast" => Special::ColorBlast {
            checked_color: BlastColor::Blue,
            filter_timing: BlastFilterTiming::Targeting,
        },
        "End the Festivities" => Special::EndTheFestivities,
        "Galvanic Blast" => Special::GalvanicBlast,
        "Rally at the Hornburg" => Special::RallyAtTheHornburg,
        "Reckless Impulse" => Special::RecklessImpulse,
        "Winding Way" => Special::WindingWay,
        "Lead the Stampede" => Special::LookTopSelectByTypeToHandBottomRest {
            look: 5,
            card_type: "Creature",
        },
        "Mental Note" => Special::MillThenDraw {
            player: MillPlayer::Controller,
            mill: 2,
            draw: 1,
        },
        "Thought Scour" => Special::MillThenDraw {
            player: MillPlayer::Target0,
            mill: 2,
            draw: 1,
        },
        "Ponder" => Special::LookReorderMayShuffleThenDraw { look: 3, draw: 1 },
        "Brainstorm" => Special::DrawThenPutHandOnLibraryTop { draw: 3, put: 2 },
        "Preordain" => Special::ScryThenDraw { scry: 2, draw: 1 },
        "Deem Inferior" => Special::DeemInferior,
        "Sleep of the Dead" => Special::TapAndSkipNextUntap,
        "Cast Down" => Special::DestroyNonlegendaryCreature,
        "Pulse of Murasa" => Special::ReturnCreatureOrLandFromGraveyardAndGainLife { amount: 6 },
        "Breath Weapon" => Special::DamageEachCreatureWithoutSubtype {
            amount: 2,
            excluded_subtype: "Dragon",
        },
        "Dread Return" => Special::ReturnOwnGraveyardCreatureToBattlefield,
        "Land Grant" => Special::SearchForestToHand,
        "Unexpected Fangs" => Special::AddPlusOnePlusOneAndLifelinkCounters,
        "Bind the Monster" => Special::BindTheMonster,
        "Snap" => Special::Snap,
        "Flaring Pain" => Special::DamageCannotBePreventedThisTurn,
        "Prismatic Strands" => Special::PrismaticStrands,
        "Cleansing Wildfire" => Special::CleansingWildfire,
        "Duress" => Special::Duress,
        "Toxin Analysis" => Special::ToxinAnalysis,
        "Weather the Storm" => Special::WeatherTheStorm,
        "Monstrous Emergence" => Special::MonstrousEmergence,
        "Nyxborn Hydra" => Special::NyxbornHydra,
        _ => Special::None,
    }
}

/// Stable, gameplay-semantic description of the generated spell target,
/// effect, and mana-ability recipe selected for a card. The card-database hash
/// includes this alongside every other generated gameplay selector below.
/// Runtime implementation changes remain source/version gated; these tokens
/// bind the per-card program selection that was previously absent from the
/// data hash.
fn effect_recipe_for(card: &CardJson) -> String {
    match special_for(&card.name) {
        Special::None => {
            let executable = card.engine_capability != EngineCapabilityJson::NoEffect;
            let spell = if executable && is_ordinary_permanent(card) {
                "MoveObject(Battlefield)"
            } else {
                "None"
            };
            let mana = if executable {
                intrinsic_basic_mana_color(card)
                    .map(|color| format!("AddMana({color})"))
                    .unwrap_or_else(|| "None".to_string())
            } else {
                "None".to_string()
            };
            format!("target=None;spell={spell};mana={mana}")
        }
        Special::GreatFurnace => "target=None;spell=None;mana=AddMana(R)".to_string(),
        Special::DrawCards(count) => {
            format!("target=None;spell=DrawCards(Controller,{count});mana=None")
        }
        Special::BurnAnyTarget(amount) => {
            format!("target=AnyTarget;spell=DealDamage({amount});mana=None")
        }
        Special::ChainLightning => "target=AnyTarget;spell=ChainLightning;mana=None".to_string(),
        Special::TargetPlayerDraw { draw } => {
            format!("target=AnyPlayer;spell=DrawCards(Target0,{draw});mana=None")
        }
        Special::DrawThenDiscard { draw, discard } => {
            format!("target=None;spell=DrawThenDiscard(Controller,{draw},{discard});mana=None")
        }
        Special::ExtractAConfession => {
            "target=None;spell=SacrificeCreature(Opponent,GreatestPowerIfCollectEvidence6);mana=None"
                .to_string()
        }
        Special::GrabThePrize => "target=None;spell=GrabThePrize;mana=None".to_string(),
        Special::HighwayRobbery => "target=None;spell=HighwayRobbery;mana=None".to_string(),
        Special::SearingBlaze => {
            "target=PlayerThenTheirCreature;spell=SearingBlaze;mana=None".to_string()
        }
        Special::CounterTarget(filter) => format!(
            "target={};spell=CounterTarget;mana=None",
            filter
                .target_spec()
                .trim_start_matches("TargetSpec::")
        ),
        Special::CounterUnlessPaysGeneric { filter, generic } => format!(
            "target={};spell=CounterTargetUnlessPaysGeneric({generic});mana=None",
            filter
                .target_spec()
                .trim_start_matches("TargetSpec::")
        ),
        Special::SteelSabotage => {
            "target=ArtifactSpellOnStack;spell=CounterTarget;mode2=ReturnArtifactPermanentToOwnersHand;mana=None".to_string()
        }
        Special::PiracyCharm => "target=Creature;spell=GrantIslandwalk;mode2=PumpTarget(2,-1);mode3=DiscardCards(Target0,1);mana=None".to_string(),
        Special::CastIntoTheFire => "target=UpToTwoCreatures;spell=DamageAllTargets(1);mode2=MoveAllTargets(Exile);mana=None".to_string(),
        Special::DustToDust => "target=ExactlyTwoArtifactPermanents;spell=ExileAllArtifactTargets;mana=None".to_string(),
        Special::ThrabenCharm => "target=Creature;spell=DealDamageByControlledCreatureCount(2);mode2=DestroyEnchantment;mode3=ExileTargetPlayersGraveyards;mana=None".to_string(),
        Special::ColorBlast {
            checked_color,
            filter_timing: BlastFilterTiming::Targeting,
        } => format!(
            "target={};spell=CounterTarget;mana=None",
            checked_color
                .filtered_spell_target_spec()
                .trim_start_matches("TargetSpec::")
        ),
        Special::ColorBlast {
            checked_color,
            filter_timing: BlastFilterTiming::Resolution,
        } => format!(
            "target=AnySpellOnStack;spell=CounterTargetIfColor({});mana=None",
            checked_color.symbol()
        ),
        Special::EndTheFestivities => {
            "target=None;spell=DamageOpponentAndTheirCreatures(1);mana=None".to_string()
        }
        Special::GalvanicBlast => "target=AnyTarget;spell=GalvanicBlast;mana=None".to_string(),
        Special::RallyAtTheHornburg => "target=None;spell=RallyAtTheHornburg;mana=None".to_string(),
        Special::RecklessImpulse => "target=None;spell=RecklessImpulse;mana=None".to_string(),
        Special::WindingWay => "target=None;spell=WindingWay;mana=None".to_string(),
        Special::LookTopSelectByTypeToHandBottomRest { look, card_type } => format!(
            "target=None;spell=LookTopSelectByTypeToHandBottomRest(Controller,{look},{card_type});mana=None"
        ),
        Special::MillThenDraw { player, mill, draw } => {
            let (target, player) = match player {
                MillPlayer::Controller => ("None", "Controller"),
                MillPlayer::Target0 => ("AnyPlayer", "Target0"),
            };
            format!("target={target};spell=MillThenDraw({player},{mill},{draw});mana=None")
        }
        Special::LookReorderMayShuffleThenDraw { look, draw } => format!(
            "target=None;spell=LookReorderMayShuffleThenDraw(Controller,{look},{draw});mana=None"
        ),
        Special::DrawThenPutHandOnLibraryTop { draw, put } => format!(
            "target=None;spell=DrawThenPutHandOnLibraryTop(Controller,{draw},{put});mana=None"
        ),
        Special::ScryThenDraw { scry, draw } => {
            format!("target=None;spell=ScryThenDraw(Controller,{scry},{draw});mana=None")
        }
        Special::DeemInferior => "target=NonlandPermanent;spell=PutObjectInOwnersLibrarySecondOrBottom(Target0);mana=None".to_string(),
        Special::TapAndSkipNextUntap => {
            "target=Creature;spell=Sequence(TapObject(Target0),SkipNextUntap(Target0));mana=None"
                .to_string()
        }
        Special::DestroyNonlegendaryCreature => {
            "target=NonlegendaryCreature;spell=DestroyObject(Target0);mana=None".to_string()
        }
        Special::ReturnCreatureOrLandFromGraveyardAndGainLife { amount } => format!(
            "target=CreatureOrLandCardInGraveyard;spell=ReturnTargetToOwnersHandThenGainLife({amount});mana=None"
        ),
        Special::DamageEachCreatureWithoutSubtype {
            amount,
            excluded_subtype,
        } => format!(
            "target=None;spell=DamageEachCreatureWithoutSubtype({amount},{excluded_subtype});mana=None"
        ),
        Special::DrawThenCreateToken { draw, token } => {
            format!("target=None;spell=DrawThenCreateToken(Controller,{draw},{token});mana=None")
        }
        Special::GainPaidCostManaValueThenDraw { draw } => {
            format!("target=None;spell=GainPaidCostManaValueThenDraw(Controller,{draw});mana=None")
        }
        Special::ReturnOwnGraveyardCreatureToBattlefield => "target=CreatureCardInOwnGraveyard;spell=MoveObject(Target0,Battlefield);mana=None".to_string(),
        Special::SearchForestToHand => "target=None;spell=SearchLibraryToHand(Controller,LandWithSubtype(Forest));mana=None".to_string(),
        Special::AddPlusOnePlusOneAndLifelinkCounters => "target=Creature;spell=AddCounters(Target0,+1/+1=1,lifelink=1);mana=None".to_string(),
        Special::BindTheMonster => {
            "target=Creature;spell=PutSourceOntoBattlefieldAttachedToTarget(Target0);mana=None"
                .to_string()
        }
        Special::Snap => {
            "target=Creature;spell=Sequence(ReturnTargetToOwnersHand,UntapUpToLands(Controller,2));mana=None"
                .to_string()
        }
        Special::DamageCannotBePreventedThisTurn => {
            "target=None;spell=DamageCannotBePreventedThisTurn;mana=None".to_string()
        }
        Special::PrismaticStrands => "target=None;spell=PreventDamageFromChosenColorUntilEndOfTurn;mana=None".to_string(),
        Special::CleansingWildfire => "target=Land;spell=Sequence(DestroyTargetLandThenMaySearchBasicTapped(Target0),DrawCards(Controller,1));mana=None".to_string(),
        Special::Duress => "target=TargetOpponent;spell=RevealTargetHandChooseNoncreatureNonlandDiscard(Target0);mana=None".to_string(),
        Special::ToxinAnalysis => "target=Creature;spell=Sequence(GrantKeywordsTargetUntilEndOfTurn(Target0,Deathtouch|Lifelink),CreateToken(ClueToken));mana=None".to_string(),
        Special::WeatherTheStorm => {
            "target=None;spell=GainLife(Controller,3);trigger=CastSelf:Storm;mana=None".to_string()
        }
        Special::MonstrousEmergence => "target=Creature;spell=DealDamageToTargetEqualToChosenCostCreaturePower;mana=None".to_string(),
        Special::NyxbornHydra => "target=None;spell=PutSourceOntoBattlefieldWithXPlusOneCounters;bestow=Creature:XGG;mana=None".to_string(),
    }
}

/// Static combat/summoning-sickness keywords, verified against each card's
/// Java source (see the increment-3 report for the exact files read).
/// Masked Meower/Sneaky Snacker (Burn) and Clockwork Percussionist/Samurai
/// Token (Rally) carry an *unconditional* static keyword this way. Goblin
/// Bushwhacker's/Goblin Tomb Raider's haste is conditional (Kicker-gated,
/// or "as long as you control an artifact") and temporary/derived, so it is
/// deliberately NOT here -- see `engine::static_self_boost_for` and
/// `EffectOp::PumpControlled`'s `grant_haste` instead.
fn keywords_for(card: &CardJson) -> String {
    let mut keywords = Vec::new();
    if card
        .mechanics
        .iter()
        .any(|mechanic| mechanic == "indestructible")
    {
        keywords.push("Keywords::INDESTRUCTIBLE");
    }
    if card.mechanics.iter().any(|mechanic| mechanic == "defender") {
        keywords.push("Keywords::DEFENDER");
    }
    match card.name.as_str() {
        "Masked Meower" | "Clockwork Percussionist" => keywords.push("Keywords::HASTE"),
        "Sneaky Snacker"
        | "Bird Illusion Token"
        | "Faerie Miscreant"
        | "Faerie Seer"
        | "Faerie Macabre"
        | "Harrier Strix"
        | "Refurbished Familiar"
        | "Sagu Wildling"
        | "Squadron Hawk"
        | "Balustrade Spy"
        | "Spellstutter Sprite" => keywords.push("Keywords::FLYING"),
        "Generous Ent" | "Writhing Chrysalis" | "Vitu-Ghazi Inspector" => {
            keywords.push("Keywords::REACH")
        }
        "Spinewoods Paladin" | "Avenging Hunter" => keywords.push("Keywords::TRAMPLE"),
        "Outlaw Medic" | "Sacred Cat" | "Sacred Cat Embalmed Token" => {
            keywords.push("Keywords::LIFELINK")
        }
        "Guardian of the Guildpact" => keywords.push("Keywords::PROTECTION_FROM_MONOCOLORED"),
        "Samurai Token" => keywords.push("Keywords::VIGILANCE"),
        _ => {}
    }
    if card.name == "Nyxborn Hydra" {
        keywords.push("Keywords::REACH");
        keywords.push("Keywords::TRAMPLE");
    }
    if card.name == "Skeleton Token" {
        keywords.push("Keywords::MENACE");
    }
    if matches!(
        card.name.as_str(),
        "Humbling Elder" | "Saiba Cryptomancer" | "Spellstutter Sprite"
    ) {
        keywords.push("Keywords::FLASH");
    }
    if card.name == "Saiba Cryptomancer" {
        keywords.push("Keywords::HEXPROOF");
    }
    if keywords.is_empty() {
        "Keywords::NONE".to_string()
    } else if keywords.len() == 1 {
        keywords[0].to_string()
    } else {
        format!(
            "Keywords({})",
            keywords
                .iter()
                .map(|keyword| format!("{keyword}.0"))
                .collect::<Vec<_>>()
                .join(" | ")
        )
    }
}

/// Whether the registry describes an activated mana ability rather than
/// one-shot production such as Burning-Tree Emissary's ETB trigger. Richer
/// activated costs are carried by `mana_ability_def_for`; ordinary sources
/// continue to use the legacy tap-and-add-one substrate.
fn has_activated_mana_ability(card: &CardJson) -> bool {
    card.mechanics
        .iter()
        .any(|mechanic| mechanic == "mana_ability")
        // Burning-Tree Emissary's produced mana belongs to its ETB trigger,
        // not an activated mana ability. Other permanents can legitimately
        // have both an ETB trigger and a printed mana ability.
        && card.name != "Burning-Tree Emissary"
}

/// Fixed colors of the primary printed mana ability. Chosen-color Gates
/// expose only their fixed white/blue option here; the chosen option is
/// materialized from object state. Heap Gate's paid any-color ability is an
/// additional rich definition, leaving its free colorless ability primary.
fn primary_mana_ability_colors(card: &CardJson) -> Vec<&str> {
    match card.name.as_str() {
        "Citadel Gate" => vec!["W"],
        "Sea Gate" => vec!["U"],
        "Heap Gate" => vec!["C"],
        _ => card.produces_mana.iter().map(String::as_str).collect(),
    }
}

fn mana_ability_includes_chosen_color(name: &str) -> bool {
    matches!(name, "Citadel Gate" | "Sea Gate")
}

fn as_enters_choose_color_other_than(name: &str) -> &'static str {
    match name {
        "Citadel Gate" => "Some(ManaColor::W)",
        "Sea Gate" => "Some(ManaColor::U)",
        _ => "None",
    }
}

fn additional_mana_abilities_for(name: &str) -> &'static str {
    match name {
        "Heap Gate" => "&[AdditionalManaAbilityDef { colors: &[ManaColor::W, ManaColor::U, ManaColor::B, ManaColor::R, ManaColor::G], mana_cost: Cost { pips: &[], generic: 1, x_count: 0 }, ability: ManaAbilityDef { cost: ManaAbilityCostDef::TapSelf, amount: ManaAbilityAmountDef::Fixed(1), controller_damage: 0, max_activations_per_turn: None } }]",
        _ => "&[]",
    }
}

fn object_name_for(name: &str) -> &str {
    match name {
        "Sacred Cat Embalmed Token" => "Sacred Cat",
        _ => name,
    }
}

fn transform_face_for(name: &str) -> &'static str {
    match name {
        "The Modern Age" => "Some(TransformFaceDef { name: \"Vector Glider\", types: &[CardType::Enchantment, CardType::Creature], subtypes: &[Subtype::Spirit], colors: &[ManaColor::U], power: Some(2), toughness: Some(3), keywords: Keywords::FLYING })",
        _ => "None",
    }
}

fn saga_for(name: &str) -> &'static str {
    match name {
        "The Modern Age" => "Some(SagaDef { chapter_effects: &[saga_chapter_modern_age_loot, saga_chapter_modern_age_loot, saga_chapter_modern_age_transform] })",
        _ => "None",
    }
}

/// Source for a single printed mana ability that is not exactly tap-and-add
/// one. The runtime interprets these definitions generically and the same
/// source fragment is included in the card-database identity below.
fn mana_ability_def_for(name: &str) -> &'static str {
    match name {
        "Elves of Deep Shadow" => "Some(ManaAbilityDef { cost: ManaAbilityCostDef::TapSelf, amount: ManaAbilityAmountDef::Fixed(1), controller_damage: 1, max_activations_per_turn: None })",
        "Lotus Petal" => "Some(ManaAbilityDef { cost: ManaAbilityCostDef::SacrificeSelf, amount: ManaAbilityAmountDef::Fixed(1), controller_damage: 0, max_activations_per_turn: None })",
        "Overgrown Battlement" => "Some(ManaAbilityDef { cost: ManaAbilityCostDef::TapSelf, amount: ManaAbilityAmountDef::ControlledCreaturesWithKeyword(Keywords::DEFENDER), controller_damage: 0, max_activations_per_turn: None })",
        "Priest of Titania" => "Some(ManaAbilityDef { cost: ManaAbilityCostDef::TapSelf, amount: ManaAbilityAmountDef::Dynamic(DynamicValueDef::BattlefieldPermanentsWithSubtype(Subtype::Elf)), controller_damage: 0, max_activations_per_turn: None })",
        "Saruli Caretaker" => "Some(ManaAbilityDef { cost: ManaAbilityCostDef::TapSelfAndOtherUntappedControlledCreature, amount: ManaAbilityAmountDef::Fixed(1), controller_damage: 0, max_activations_per_turn: None })",
        "Tinder Wall" => "Some(ManaAbilityDef { cost: ManaAbilityCostDef::SacrificeSelf, amount: ManaAbilityAmountDef::Fixed(2), controller_damage: 0, max_activations_per_turn: None })",
        "Wall of Roots" => "Some(ManaAbilityDef { cost: ManaAbilityCostDef::PutMinus0Minus1CounterOnSelf, amount: ManaAbilityAmountDef::Fixed(1), controller_damage: 0, max_activations_per_turn: Some(1) })",
        "Treasure Token" => "Some(ManaAbilityDef { cost: ManaAbilityCostDef::TapAndSacrificeSelf, amount: ManaAbilityAmountDef::Fixed(1), controller_damage: 0, max_activations_per_turn: None })",
        "Eldrazi Spawn Token" => "Some(ManaAbilityDef { cost: ManaAbilityCostDef::SacrificeSelf, amount: ManaAbilityAmountDef::Fixed(1), controller_damage: 0, max_activations_per_turn: None })",
        _ => "None",
    }
}

fn enters_battlefield_tapped(card: &CardJson) -> bool {
    card.mechanics
        .iter()
        .any(|mechanic| mechanic == "enters_tapped")
        && card.name != "Gingerbread Cabin"
}

fn enters_battlefield_tapped_unless_for(name: &str) -> &'static str {
    match name {
        "Gingerbread Cabin" => "Some(EntersBattlefieldTappedUnlessDef { controller_controls_other_subtype: Subtype::Forest, minimum_count: 3 })",
        _ => "None",
    }
}

/// `Some` Kicker cost source text (`CardDef::kicker_cost`), verified against
/// Java (`KickerAbility`). Only Goblin Bushwhacker has one this increment
/// ("Kicker {R}").
fn kicker_cost_for(name: &str) -> String {
    match name {
        "Goblin Bushwhacker" => cost_src("{R}"),
        _ => "None".to_string(),
    }
}

/// `Some` alternative cost source text (`CardDef::alt_cost`), verified
/// against Java. Only Fireblast has one this increment ("You may
/// sacrifice two Mountains rather than pay Fireblast's mana cost.").
fn alt_cost_for(name: &str) -> &'static str {
    match name {
        "Fireblast" => "Some(&[CostComponent::SacrificeLands(2)])",
        "Land Grant" => "Some(&[CostComponent::RevealHandIfNoCardsWithType(CardType::Land)])",
        _ => "None",
    }
}

/// `Some` mandatory additional cost text (`CardDef::additional_cost`).
/// Only Grab the Prize has one this increment ("As an additional cost to
/// cast this spell, discard a card.").
fn additional_cost_for(name: &str) -> &'static str {
    match name {
        "Grab the Prize" => "Some(&[CostComponent::DiscardCards(1)])",
        "Fanatical Offering" | "Reckoner's Bargain" | "Eviscerator's Insight" => {
            "Some(&[CostComponent::SacrificeControlled { count: 1, filter: PermanentFilter::ArtifactOrCreature }])"
        }
        "Monstrous Emergence" => {
            "Some(&[CostComponent::ChooseControlledCreatureOrRevealCreatureCardFromHand])"
        }
        _ => "None",
    }
}

fn bestow_for(name: &str) -> String {
    match name {
        "Nyxborn Hydra" => {
            let (pips, generic, x_count) = parse_cost("{X}{G}{G}");
            format!(
                "Some(BestowDef {{ cost: Cost {{ pips: &[{}], generic: {generic}, x_count: {x_count} }}, target_spec: TargetSpec::Creature }})",
                pips.join(", ")
            )
        }
        _ => "None".to_string(),
    }
}

/// `Some` ordered flashback-cost definition, verified against Java. Faithless
/// Looting ("Flashback {2}{R}") pays mana; Lava Dart ("Flashback --
/// Sacrifice a Mountain.") sacrifices a land; Deep Analysis pays `{1}{U}`
/// followed by 3 life. All three use the same composable `CostComponent`
/// substrate as alternative, additional, and activated-ability costs.
fn flashback_for(name: &str) -> String {
    match name {
        "Faithless Looting" => {
            let (pips, generic, x_count) = parse_cost("{2}{R}");
            format!(
                "Some(FlashbackDef {{ cost: &[CostComponent::Mana(Cost {{ pips: &[{}], generic: {generic}, x_count: {x_count} }})] }})",
                pips.join(", ")
            )
        }
        "Lava Dart" => {
            "Some(FlashbackDef { cost: &[CostComponent::SacrificeLands(1)] })".to_string()
        }
        "Deep Analysis" => {
            let (pips, generic, x_count) = parse_cost("{1}{U}");
            format!(
                "Some(FlashbackDef {{ cost: &[CostComponent::Mana(Cost {{ pips: &[{}], generic: {generic}, x_count: {x_count} }}), CostComponent::PayLife(3)] }})",
                pips.join(", ")
            )
        }
        "Eviscerator's Insight" => {
            let (pips, generic, x_count) = parse_cost("{4}{B}");
            format!(
                "Some(FlashbackDef {{ cost: &[CostComponent::Mana(Cost {{ pips: &[{}], generic: {generic}, x_count: {x_count} }})] }})",
                pips.join(", ")
            )
        }
        "Dread Return" => "Some(FlashbackDef { cost: &[CostComponent::SacrificeControlled { count: 3, filter: PermanentFilter::Creature }] })".to_string(),
        "Flaring Pain" => {
            let (pips, generic, x_count) = parse_cost("{R}");
            format!(
                "Some(FlashbackDef {{ cost: &[CostComponent::Mana(Cost {{ pips: &[{}], generic: {generic}, x_count: {x_count} }})] }})",
                pips.join(", ")
            )
        }
        "Prismatic Strands" => "Some(FlashbackDef { cost: &[CostComponent::TapUntappedControlledPermanent(PermanentFilterDef::CreatureWithColor(ManaColor::W))] })".to_string(),
        _ => "None".to_string(),
    }
}

/// `Some` ordered escape-cost definition. Sleep of the Dead pays `{2}{U}`
/// and exiles three other cards from its owner's graveyard. The source is
/// excluded by the shared cast-cost candidate contract, not by a card-name
/// branch in the engine.
fn escape_for(name: &str) -> String {
    match name {
        "Sleep of the Dead" => {
            let (pips, generic, x_count) = parse_cost("{2}{U}");
            format!(
                "Some(EscapeDef {{ cost: &[CostComponent::Mana(Cost {{ pips: &[{}], generic: {generic}, x_count: {x_count} }}), CostComponent::ExileOtherCardsFromOwnGraveyard(3)] }})",
                pips.join(", ")
            )
        }
        _ => "None".to_string(),
    }
}

/// Non-mana activated abilities, verified against Java. Masked Meower
/// ("Discard a card, Sacrifice this creature: Draw a card.") and the Blood
/// token ("{1}, {T}, Discard a card, Sacrifice this artifact: Draw a
/// card.") both reduce to "discard/sacrifice/[cost]: draw a card", so both
/// share `ability_effect_draw_one`. Experimental Synthesizer's ("{2}{R},
/// Sacrifice Experimental Synthesizer: Create a 2/2 white Samurai creature
/// token with vigilance. Activate only as a sorcery.") is Rally's only
/// activated ability and the only one so far with `sorcery_speed_only:
/// true` -- see that field's doc in `card_def.rs`. Lorien Revealed adds the
/// hand-zone Islandcycling `{1}` shape: pay mana, discard the exact source,
/// then resolve the reusable typed library search.
fn activated_ability_recipes_for(name: &str) -> &'static [ActivatedAbilityRecipe] {
    match name {
        "Faerie Macabre" => &[ActivatedAbilityRecipe {
            cost: &[AbilityCostRecipe::DiscardSelf],
            effect: AbilityEffectRecipe::MoveAllTargetsToExile,
            activation_zone: "Hand",
            sorcery_speed_only: false,
            target_spec: "UpToTwoCardsInGraveyards",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Fume Spitter" => &[ActivatedAbilityRecipe {
            cost: &[AbilityCostRecipe::SacrificeSelf],
            effect: AbilityEffectRecipe::AddMinusOneMinusOneCounter,
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "Creature",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Krark-Clan Shaman" => &[ActivatedAbilityRecipe {
            cost: &[AbilityCostRecipe::SacrificeControlled {
                count: 1,
                filter: PermanentFilterRecipe::Artifact,
            }],
            effect: AbilityEffectRecipe::DamageAllCreatures {
                amount: 1,
                filter: CreatureEffectFilterRecipe::WithoutKeyword("FLYING"),
            },
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Makeshift Munitions" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: None,
                    generic: 1,
                },
                AbilityCostRecipe::SacrificeControlled {
                    count: 1,
                    filter: PermanentFilterRecipe::ArtifactOrCreature,
                },
            ],
            effect: AbilityEffectRecipe::DealDamageAnyTarget(1),
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "AnyTarget",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Nihil Spellbomb" => &[ActivatedAbilityRecipe {
            cost: &[AbilityCostRecipe::Tap, AbilityCostRecipe::SacrificeSelf],
            effect: AbilityEffectRecipe::ExileTargetPlayersGraveyard,
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "AnyPlayer",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Relic of Progenitus" => &[
            ActivatedAbilityRecipe {
                cost: &[AbilityCostRecipe::Tap],
                effect: AbilityEffectRecipe::ExileOneFromTargetPlayersGraveyard,
                activation_zone: "Battlefield",
                sorcery_speed_only: false,
                target_spec: "AnyPlayer",
                activation_target_filter: "TargetSpecOnly",
                max_activations_per_turn: None,
            },
            ActivatedAbilityRecipe {
                cost: &[
                    AbilityCostRecipe::Mana {
                        colored: None,
                        generic: 1,
                    },
                    AbilityCostRecipe::ExileSelf,
                ],
                effect: AbilityEffectRecipe::ExileAllGraveyardsThenDraw(1),
                activation_zone: "Battlefield",
                sorcery_speed_only: false,
                target_spec: "None",
                activation_target_filter: "TargetSpecOnly",
                max_activations_per_turn: None,
            },
        ],
        "Masked Meower" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::DiscardCards(1),
                AbilityCostRecipe::SacrificeSelf,
            ],
            effect: AbilityEffectRecipe::DrawCards(1),
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Blood Token" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: None,
                    generic: 1,
                },
                AbilityCostRecipe::Tap,
                AbilityCostRecipe::DiscardCards(1),
                AbilityCostRecipe::SacrificeSelf,
            ],
            effect: AbilityEffectRecipe::DrawCards(1),
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Food Token" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: None,
                    generic: 2,
                },
                AbilityCostRecipe::Tap,
                AbilityCostRecipe::SacrificeSelf,
            ],
            effect: AbilityEffectRecipe::GainLife(3),
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Lembas" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: None,
                    generic: 2,
                },
                AbilityCostRecipe::Tap,
                AbilityCostRecipe::SacrificeSelf,
            ],
            effect: AbilityEffectRecipe::GainLife(3),
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Clue Token" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: None,
                    generic: 2,
                },
                AbilityCostRecipe::SacrificeSelf,
            ],
            effect: AbilityEffectRecipe::DrawCards(1),
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Map Token" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: None,
                    generic: 1,
                },
                AbilityCostRecipe::Tap,
                AbilityCostRecipe::SacrificeSelf,
            ],
            effect: AbilityEffectRecipe::ExploreTarget,
            activation_zone: "Battlefield",
            sorcery_speed_only: true,
            target_spec: "ControlledCreature",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Blood Fountain" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: Some("B"),
                    generic: 3,
                },
                AbilityCostRecipe::Tap,
                AbilityCostRecipe::SacrificeSelf,
            ],
            effect: AbilityEffectRecipe::MoveAllTargetsToHand,
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "UpToTwoCreatureCardsInOwnGraveyard",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Experimental Synthesizer" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: Some("R"),
                    generic: 2,
                },
                AbilityCostRecipe::SacrificeSelf,
            ],
            effect: AbilityEffectRecipe::CreateToken("Samurai Token"),
            activation_zone: "Battlefield",
            sorcery_speed_only: true,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Lorien Revealed" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: None,
                    generic: 1,
                },
                AbilityCostRecipe::DiscardSelf,
            ],
            effect: AbilityEffectRecipe::SearchLibraryToHand {
                filter: LibrarySearchFilterRecipe::LandWithSubtype("Island"),
                min_targets: 0,
                max_targets: 1,
                reveal_selected: true,
                shuffle: true,
            },
            activation_zone: "Hand",
            sorcery_speed_only: false,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Generous Ent" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: None,
                    generic: 1,
                },
                AbilityCostRecipe::DiscardSelf,
            ],
            effect: AbilityEffectRecipe::SearchLibraryToHand {
                filter: LibrarySearchFilterRecipe::LandWithSubtype("Forest"),
                min_targets: 0,
                max_targets: 1,
                reveal_selected: true,
                shuffle: true,
            },
            activation_zone: "Hand",
            sorcery_speed_only: false,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Troll of Khazad-dum" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: None,
                    generic: 1,
                },
                AbilityCostRecipe::DiscardSelf,
            ],
            effect: AbilityEffectRecipe::SearchLibraryToHand {
                filter: LibrarySearchFilterRecipe::LandWithSubtype("Swamp"),
                min_targets: 0,
                max_targets: 1,
                reveal_selected: true,
                shuffle: true,
            },
            activation_zone: "Hand",
            sorcery_speed_only: false,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Tinder Wall" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: Some("R"),
                    generic: 0,
                },
                AbilityCostRecipe::SacrificeSelf,
            ],
            effect: AbilityEffectRecipe::DamageTarget(2),
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "Creature",
            activation_target_filter: "CreatureBlockedBySource",
            max_activations_per_turn: None,
        }],
        "Quirion Ranger" => &[ActivatedAbilityRecipe {
            cost: &[AbilityCostRecipe::ReturnControlledLandWithSubtype("Forest")],
            effect: AbilityEffectRecipe::UntapTarget,
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "Creature",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: Some(1),
        }],
        "Timberwatch Elf" => &[ActivatedAbilityRecipe {
            cost: &[AbilityCostRecipe::Tap],
            effect: AbilityEffectRecipe::PumpTargetBattlefieldSubtypeCount("Elf"),
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "Creature",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Wellwisher" => &[ActivatedAbilityRecipe {
            cost: &[AbilityCostRecipe::Tap],
            effect: AbilityEffectRecipe::GainLifeBattlefieldSubtypeCount("Elf"),
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Basilisk Gate" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: None,
                    generic: 2,
                },
                AbilityCostRecipe::Tap,
            ],
            effect: AbilityEffectRecipe::PumpTargetByControlledSubtypeCount("Gate"),
            activation_zone: "Battlefield",
            sorcery_speed_only: true,
            target_spec: "Creature",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Heap Gate" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: None,
                    generic: 1,
                },
                AbilityCostRecipe::Tap,
                AbilityCostRecipe::TapOtherUntappedControlledPermanentWithSubtype("Gate"),
            ],
            effect: AbilityEffectRecipe::CreateToken("Treasure Token"),
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Sacred Cat" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: Some("W"),
                    generic: 0,
                },
                AbilityCostRecipe::ExileSelf,
            ],
            effect: AbilityEffectRecipe::CreateToken("Sacred Cat Embalmed Token"),
            activation_zone: "Graveyard",
            sorcery_speed_only: true,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Black Mage's Rod" => &[ActivatedAbilityRecipe {
            cost: &[AbilityCostRecipe::Mana {
                colored: None,
                generic: 3,
            }],
            effect: AbilityEffectRecipe::AttachSourceToTarget,
            activation_zone: "Battlefield",
            sorcery_speed_only: true,
            target_spec: "ControlledCreature",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Hunter's Blowgun" => &[ActivatedAbilityRecipe {
            cost: &[AbilityCostRecipe::Mana {
                colored: None,
                generic: 2,
            }],
            effect: AbilityEffectRecipe::AttachSourceToTarget,
            activation_zone: "Battlefield",
            sorcery_speed_only: true,
            target_spec: "ControlledCreature",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Cryogen Relic" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: Some("U"),
                    generic: 1,
                },
                AbilityCostRecipe::SacrificeSelf,
            ],
            effect: AbilityEffectRecipe::AddStunCounterToOptionalTarget,
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "UpToOneTappedCreature",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Gorilla Shaman" => &[ActivatedAbilityRecipe {
            cost: &[AbilityCostRecipe::VariableMana {
                generic: 1,
                x_count: 2,
            }],
            effect: AbilityEffectRecipe::DestroyTarget,
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "NoncreatureArtifactPermanent",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Harrier Strix" => &[ActivatedAbilityRecipe {
            cost: &[AbilityCostRecipe::Mana {
                colored: Some("U"),
                generic: 2,
            }],
            effect: AbilityEffectRecipe::DrawThenDiscard {
                draw: 1,
                discard: 1,
            },
            activation_zone: "Battlefield",
            sorcery_speed_only: false,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Moon-Circuit Hacker" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: Some("U"),
                    generic: 0,
                },
                AbilityCostRecipe::ReturnControlledUnblockedAttacker,
            ],
            effect: AbilityEffectRecipe::PutSourceOntoBattlefieldTappedAndAttacking,
            activation_zone: "Hand",
            sorcery_speed_only: false,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Ninja of the Deep Hours" => &[ActivatedAbilityRecipe {
            cost: &[
                AbilityCostRecipe::Mana {
                    colored: Some("U"),
                    generic: 1,
                },
                AbilityCostRecipe::ReturnControlledUnblockedAttacker,
            ],
            effect: AbilityEffectRecipe::PutSourceOntoBattlefieldTappedAndAttacking,
            activation_zone: "Hand",
            sorcery_speed_only: false,
            target_spec: "None",
            activation_target_filter: "TargetSpecOnly",
            max_activations_per_turn: None,
        }],
        "Twisted Landscape" => &[
            ActivatedAbilityRecipe {
                cost: &[AbilityCostRecipe::Tap, AbilityCostRecipe::SacrificeSelf],
                effect: AbilityEffectRecipe::SearchLibraryToBattlefieldTapped {
                    filter: LibrarySearchFilterRecipe::BasicLandWithAnySubtype([
                        "Swamp", "Mountain", "Forest",
                    ]),
                },
                activation_zone: "Battlefield",
                sorcery_speed_only: false,
                target_spec: "None",
                activation_target_filter: "TargetSpecOnly",
                max_activations_per_turn: None,
            },
            ActivatedAbilityRecipe {
                cost: &[
                    AbilityCostRecipe::ManaCost("{B}{R}{G}"),
                    AbilityCostRecipe::DiscardSelf,
                ],
                effect: AbilityEffectRecipe::DrawCards(1),
                activation_zone: "Hand",
                sorcery_speed_only: false,
                target_spec: "None",
                activation_target_filter: "TargetSpecOnly",
                max_activations_per_turn: None,
            },
        ],
        _ => &[],
    }
}

fn ability_cost_src(cost: AbilityCostRecipe) -> String {
    match cost {
        AbilityCostRecipe::Mana { colored, generic } => {
            let pips = colored
                .map(|color| format!("Pip::Colored(ManaColor::{color})"))
                .unwrap_or_default();
            format!(
                "CostComponent::Mana(Cost {{ pips: &[{pips}], generic: {generic}, x_count: 0 }})"
            )
        }
        AbilityCostRecipe::VariableMana { generic, x_count } => format!(
            "CostComponent::Mana(Cost {{ pips: &[], generic: {generic}, x_count: {x_count} }})"
        ),
        AbilityCostRecipe::Tap => "CostComponent::Tap".to_string(),
        AbilityCostRecipe::DiscardCards(count) => {
            format!("CostComponent::DiscardCards({count})")
        }
        AbilityCostRecipe::DiscardSelf => "CostComponent::DiscardSelf".to_string(),
        AbilityCostRecipe::SacrificeSelf => "CostComponent::SacrificeSelf".to_string(),
        AbilityCostRecipe::ReturnControlledLandWithSubtype(subtype) => format!(
            "CostComponent::ReturnControlledPermanentToOwnersHand(PermanentFilterDef::LandWithSubtype(Subtype::{subtype}))"
        ),
        AbilityCostRecipe::ExileSelf => "CostComponent::ExileSelf".to_string(),
        AbilityCostRecipe::TapOtherUntappedControlledPermanentWithSubtype(subtype) => format!(
            "CostComponent::TapOtherUntappedControlledPermanentWithSubtype({})",
            subtype_variant(subtype)
        ),
        AbilityCostRecipe::SacrificeControlled { count, filter } => format!(
            "CostComponent::SacrificeControlled {{ count: {count}, filter: {} }}",
            permanent_filter_src(filter)
        ),
        AbilityCostRecipe::ReturnControlledUnblockedAttacker => {
            "CostComponent::ReturnControlledUnblockedAttackerToOwnersHand".to_string()
        }
        AbilityCostRecipe::ManaCost(cost) => {
            let (pips, generic, x_count) = parse_cost(cost);
            format!(
                "CostComponent::Mana(Cost {{ pips: &[{}], generic: {generic}, x_count: {x_count} }})",
                pips.join(", ")
            )
        }
    }
}

fn ability_cost_token(cost: AbilityCostRecipe) -> String {
    match cost {
        AbilityCostRecipe::Mana { colored, generic } => {
            format!("mana:{}:{generic}:0", colored.unwrap_or("-"))
        }
        AbilityCostRecipe::VariableMana { generic, x_count } => {
            format!("mana:-:{generic}:{x_count}")
        }
        AbilityCostRecipe::Tap => "tap".to_string(),
        AbilityCostRecipe::DiscardCards(count) => format!("discard_cards:{count}"),
        AbilityCostRecipe::DiscardSelf => "discard_self".to_string(),
        AbilityCostRecipe::SacrificeSelf => "sacrifice_self".to_string(),
        AbilityCostRecipe::ReturnControlledLandWithSubtype(subtype) => {
            format!("return_controlled_land_with_subtype:{subtype}")
        }
        AbilityCostRecipe::ExileSelf => "exile_self".to_string(),
        AbilityCostRecipe::TapOtherUntappedControlledPermanentWithSubtype(subtype) => {
            format!("tap_other_untapped_controlled_subtype:{subtype}")
        }
        AbilityCostRecipe::SacrificeControlled { count, filter } => {
            format!(
                "sacrifice_controlled:{count}:{}",
                permanent_filter_token(filter)
            )
        }
        AbilityCostRecipe::ReturnControlledUnblockedAttacker => {
            "return_controlled_unblocked_attacker".to_string()
        }
        AbilityCostRecipe::ManaCost(cost) => format!("mana_cost:{cost}"),
    }
}

fn permanent_filter_src(filter: PermanentFilterRecipe) -> &'static str {
    match filter {
        PermanentFilterRecipe::Artifact => "PermanentFilter::Artifact",
        PermanentFilterRecipe::ArtifactOrCreature => "PermanentFilter::ArtifactOrCreature",
    }
}

fn permanent_filter_token(filter: PermanentFilterRecipe) -> &'static str {
    match filter {
        PermanentFilterRecipe::Artifact => "artifact",
        PermanentFilterRecipe::ArtifactOrCreature => "artifact_or_creature",
    }
}

fn ability_effect_token(effect: AbilityEffectRecipe) -> String {
    match effect {
        AbilityEffectRecipe::DrawCards(count) => format!("draw_cards:{count}"),
        AbilityEffectRecipe::GainLife(amount) => format!("gain_life:{amount}"),
        AbilityEffectRecipe::CreateToken(name) => format!("create_token:{name}"),
        AbilityEffectRecipe::DamageTarget(amount) => format!("damage_target:{amount}"),
        AbilityEffectRecipe::MoveAllTargetsToHand => "move_all_targets_to_hand".to_string(),
        AbilityEffectRecipe::ExploreTarget => "explore_target".to_string(),
        AbilityEffectRecipe::DealDamageAnyTarget(amount) => {
            format!("deal_damage_any_target:{amount}")
        }
        AbilityEffectRecipe::DamageAllCreatures { amount, filter } => format!(
            "damage_all_creatures:{amount}:{}",
            creature_effect_filter_token(filter)
        ),
        AbilityEffectRecipe::ExileTargetPlayersGraveyard => {
            "exile_target_players_graveyard".to_string()
        }
        AbilityEffectRecipe::ExileOneFromTargetPlayersGraveyard => {
            "exile_one_from_target_players_graveyard".to_string()
        }
        AbilityEffectRecipe::ExileAllGraveyardsThenDraw(draw) => {
            format!("exile_all_graveyards_then_draw:{draw}")
        }
        AbilityEffectRecipe::SearchLibraryToHand {
            filter,
            min_targets,
            max_targets,
            reveal_selected,
            shuffle,
        } => format!(
            "search_library_to_hand:{}:min={min_targets}:max={max_targets}:reveal={reveal_selected}:shuffle={shuffle}",
            library_search_filter_token(filter)
        ),
        AbilityEffectRecipe::UntapTarget => "untap_target".to_string(),
        AbilityEffectRecipe::GainLifeBattlefieldSubtypeCount(subtype) => {
            format!("gain_life:battlefield_subtype_count:{subtype}")
        }
        AbilityEffectRecipe::PumpTargetBattlefieldSubtypeCount(subtype) => {
            format!("pump_target:battlefield_subtype_count:{subtype}")
        }
        AbilityEffectRecipe::PumpTargetByControlledSubtypeCount(subtype) => {
            format!("pump_target_by_controlled_subtype_count:{subtype}")
        }
        AbilityEffectRecipe::AttachSourceToTarget => "attach_source_to_target".to_string(),
        AbilityEffectRecipe::AddStunCounterToOptionalTarget => {
            "add_stun_counter_to_optional_target".to_string()
        }
        AbilityEffectRecipe::DestroyTarget => "destroy_target".to_string(),
        AbilityEffectRecipe::DrawThenDiscard { draw, discard } => {
            format!("draw_then_discard:{draw}:{discard}")
        }
        AbilityEffectRecipe::PutSourceOntoBattlefieldTappedAndAttacking => {
            "put_source_onto_battlefield_tapped_and_attacking".to_string()
        }
        AbilityEffectRecipe::MoveAllTargetsToExile => {
            "move_all_targets_to_exile".to_string()
        }
        AbilityEffectRecipe::AddMinusOneMinusOneCounter => {
            "add_minus_one_minus_one_counter".to_string()
        }
        AbilityEffectRecipe::SearchLibraryToBattlefieldTapped { filter } => format!(
            "search_library_to_battlefield_tapped:{}",
            library_search_filter_token(filter)
        ),
    }
}

fn creature_effect_filter_token(filter: CreatureEffectFilterRecipe) -> String {
    match filter {
        CreatureEffectFilterRecipe::WithoutKeyword(keyword) => {
            format!("without_keyword:{}", keyword.to_ascii_lowercase())
        }
    }
}

fn library_search_filter_token(filter: LibrarySearchFilterRecipe) -> String {
    match filter {
        LibrarySearchFilterRecipe::LandWithSubtype(subtype) => {
            format!("land_with_subtype:{subtype}")
        }
        LibrarySearchFilterRecipe::BasicLand => "basic_land".to_string(),
        LibrarySearchFilterRecipe::BasicLandWithAnySubtype(subtypes) => {
            format!("basic_land_with_any_subtype:{}", subtypes.join("|"))
        }
    }
}

fn library_search_filter_src(filter: LibrarySearchFilterRecipe) -> String {
    match filter {
        LibrarySearchFilterRecipe::LandWithSubtype(subtype) => {
            format!("LibraryCardFilter::LandWithSubtype(Subtype::{subtype})")
        }
        LibrarySearchFilterRecipe::BasicLand => "LibraryCardFilter::BasicLand".to_string(),
        LibrarySearchFilterRecipe::BasicLandWithAnySubtype(subtypes) => format!(
            "LibraryCardFilter::BasicLandWithAnySubtype([{}])",
            subtypes
                .iter()
                .map(|subtype| subtype_variant(subtype))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn ability_effect_fn_name(effect: AbilityEffectRecipe) -> String {
    match effect {
        AbilityEffectRecipe::DrawCards(count) => format!("ability_effect_draw_{count}"),
        AbilityEffectRecipe::GainLife(amount) => format!("ability_effect_gain_life_{amount}"),
        AbilityEffectRecipe::CreateToken("Samurai Token") => {
            "ability_effect_create_samurai_token".to_string()
        }
        AbilityEffectRecipe::CreateToken("Treasure Token") => {
            "ability_effect_create_treasure_token".to_string()
        }
        AbilityEffectRecipe::CreateToken("Sacred Cat Embalmed Token") => {
            "ability_effect_create_sacred_cat_embalmed_token".to_string()
        }
        AbilityEffectRecipe::CreateToken(name) => {
            panic!("no generated activated-ability token function for {name:?}")
        }
        AbilityEffectRecipe::DamageTarget(amount) => {
            format!("ability_effect_damage_target_{amount}")
        }
        AbilityEffectRecipe::MoveAllTargetsToHand => {
            "ability_effect_move_all_targets_to_hand".to_string()
        }
        AbilityEffectRecipe::ExploreTarget => "ability_effect_explore_target".to_string(),
        AbilityEffectRecipe::DealDamageAnyTarget(amount) => {
            format!("ability_effect_deal_damage_any_target_{amount}")
        }
        AbilityEffectRecipe::DamageAllCreatures {
            amount,
            filter: CreatureEffectFilterRecipe::WithoutKeyword(keyword),
        } => format!(
            "ability_effect_damage_all_creatures_without_{}_{}",
            keyword.to_ascii_lowercase(),
            amount
        ),
        AbilityEffectRecipe::ExileTargetPlayersGraveyard => {
            "ability_effect_exile_target_players_graveyard".to_string()
        }
        AbilityEffectRecipe::ExileOneFromTargetPlayersGraveyard => {
            "ability_effect_exile_one_from_target_players_graveyard".to_string()
        }
        AbilityEffectRecipe::ExileAllGraveyardsThenDraw(draw) => {
            format!("ability_effect_exile_all_graveyards_then_draw_{draw}")
        }
        AbilityEffectRecipe::SearchLibraryToHand {
            filter: LibrarySearchFilterRecipe::LandWithSubtype(subtype),
            min_targets: 0,
            max_targets: 1,
            reveal_selected: true,
            shuffle: true,
        } => format!("ability_effect_{}cycle", subtype.to_ascii_lowercase()),
        AbilityEffectRecipe::SearchLibraryToHand {
            filter: LibrarySearchFilterRecipe::BasicLand,
            min_targets: 0,
            max_targets: 1,
            reveal_selected: true,
            shuffle: true,
        } => "omen_effect_search_basic_land".to_string(),
        AbilityEffectRecipe::SearchLibraryToHand { .. } => panic!(
            "SearchLibraryToHand currently supports only optional single-card reveal+shuffle"
        ),
        AbilityEffectRecipe::UntapTarget => "ability_effect_untap_target".to_string(),
        AbilityEffectRecipe::GainLifeBattlefieldSubtypeCount(subtype) => format!(
            "ability_effect_gain_life_battlefield_{}_count",
            subtype.to_ascii_lowercase()
        ),
        AbilityEffectRecipe::PumpTargetBattlefieldSubtypeCount(subtype) => format!(
            "ability_effect_pump_target_battlefield_{}_count",
            subtype.to_ascii_lowercase()
        ),
        AbilityEffectRecipe::PumpTargetByControlledSubtypeCount(subtype) => format!(
            "ability_effect_pump_target_by_controlled_{}_count",
            subtype.to_ascii_lowercase()
        ),
        AbilityEffectRecipe::AttachSourceToTarget => {
            "ability_effect_attach_source_to_target".to_string()
        }
        AbilityEffectRecipe::AddStunCounterToOptionalTarget => {
            "ability_effect_add_stun_counter_to_optional_target".to_string()
        }
        AbilityEffectRecipe::DestroyTarget => "ability_effect_destroy_target".to_string(),
        AbilityEffectRecipe::DrawThenDiscard { draw, discard } => {
            format!("ability_effect_draw_{draw}_then_discard_{discard}")
        }
        AbilityEffectRecipe::PutSourceOntoBattlefieldTappedAndAttacking => {
            "ability_effect_put_source_onto_battlefield_tapped_and_attacking".to_string()
        }
        AbilityEffectRecipe::MoveAllTargetsToExile => {
            "ability_effect_move_all_targets_to_exile".to_string()
        }
        AbilityEffectRecipe::AddMinusOneMinusOneCounter => {
            "ability_effect_add_minus_one_minus_one_counter".to_string()
        }
        AbilityEffectRecipe::SearchLibraryToBattlefieldTapped { filter } => format!(
            "ability_effect_search_{}_to_battlefield_tapped",
            library_search_filter_token(filter)
                .replace([':', '|'], "_")
                .to_ascii_lowercase()
        ),
    }
}

fn activated_abilities_for(name: &str) -> String {
    let recipes = activated_ability_recipes_for(name);
    if recipes.is_empty() {
        return "&[]".to_string();
    }
    let abilities = recipes
        .iter()
        .map(|recipe| {
            let costs = recipe
                .cost
                .iter()
                .copied()
                .map(ability_cost_src)
                .collect::<Vec<_>>()
                .join(", ");
            let effect = ability_effect_fn_name(recipe.effect);
            let zone = recipe.activation_zone;
            let sorcery = recipe.sorcery_speed_only;
            let target_spec = recipe.target_spec;
            let activation_target_filter = recipe.activation_target_filter;
            let max_activations_per_turn = match recipe.max_activations_per_turn {
                Some(limit) => format!("Some({limit})"),
                None => "None".to_string(),
            };
            format!(
                "ActivatedAbilityDef {{ cost: &[{costs}], target_spec: TargetSpec::{target_spec}, effect: {effect}, activation_zone: Zone::{zone}, sorcery_speed_only: {sorcery}, activation_target_filter: ActivationTargetFilter::{activation_target_filter}, max_activations_per_turn: {max_activations_per_turn} }}"
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("&[{abilities}]")
}

fn activated_abilities_token(name: &str) -> String {
    activated_ability_recipes_for(name)
        .iter()
        .map(|recipe| {
            let costs = recipe
                .cost
                .iter()
                .copied()
                .map(ability_cost_token)
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "zone={};sorcery={};target={};activation_filter={};max_per_turn={};cost=[{}];effect={}",
                recipe.activation_zone.to_ascii_lowercase(),
                recipe.sorcery_speed_only,
                recipe.target_spec.to_ascii_lowercase(),
                recipe.activation_target_filter.to_ascii_lowercase(),
                recipe
                    .max_activations_per_turn
                    .map_or_else(|| "-".to_string(), |limit| limit.to_string()),
                costs,
                ability_effect_token(recipe.effect)
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

/// `Some` Plot cost source text (`CardDef::plot_cost`), verified against
/// Java (`PlotAbility`). Only Highway Robbery has one this increment
/// ("Plot {1}{R}").
fn plot_cost_for(name: &str) -> String {
    match name {
        "Highway Robbery" => cost_src("{1}{R}"),
        "Spinewoods Paladin" => cost_src("{3}{G}"),
        _ => "None".to_string(),
    }
}

/// `Some` Madness cost source text (`CardDef::madness_cost`), verified
/// against Java (`MadnessAbility`). Only Fiery Temper has one this
/// increment ("Madness {R}").
fn madness_cost_for(name: &str) -> String {
    match name {
        "Fiery Temper" => cost_src("{R}"),
        _ => "None".to_string(),
    }
}

/// `Some` second-mode source text (`CardDef::mode2`) for the symmetric
/// Elemental Blast/Pyroblast/Hydroblast family.
fn mode2_for(name: &str) -> String {
    match special_for(name) {
        Special::ColorBlast {
            checked_color,
            filter_timing: BlastFilterTiming::Targeting,
        } => format!(
            "Some(ModeDef {{ target_spec: {}, effect: mode2_effect_destroy_target_permanent }})",
            checked_color.filtered_permanent_target_spec()
        ),
        Special::ColorBlast {
            checked_color,
            filter_timing: BlastFilterTiming::Resolution,
        } => format!(
            "Some(ModeDef {{ target_spec: TargetSpec::AnyPermanent, effect: mode2_effect_destroy_target_permanent_if_{} }})",
            checked_color.suffix()
        ),
        Special::SteelSabotage => "Some(ModeDef { target_spec: TargetSpec::ArtifactPermanent, effect: mode2_effect_return_target_permanent_to_owners_hand })".to_string(),
        Special::PiracyCharm => "Some(ModeDef { target_spec: TargetSpec::Creature, effect: mode2_effect_piracy_charm_pump })".to_string(),
        Special::CastIntoTheFire => "Some(ModeDef { target_spec: TargetSpec::ArtifactPermanent, effect: mode2_effect_cast_into_the_fire_exile_artifact })".to_string(),
        Special::ThrabenCharm => "Some(ModeDef { target_spec: TargetSpec::EnchantmentPermanent, effect: mode2_effect_thraben_charm_destroy_enchantment })".to_string(),
        _ => "None".to_string(),
    }
}

/// Optional third printed mode, using the same stable mode index carried by
/// pending casts and stack items.
fn mode3_for(name: &str) -> String {
    match special_for(name) {
        Special::PiracyCharm => "Some(ModeDef { target_spec: TargetSpec::AnyPlayer, effect: mode3_effect_piracy_charm_discard })".to_string(),
        Special::ThrabenCharm => "Some(ModeDef { target_spec: TargetSpec::UpToTwoPlayers, effect: mode3_effect_thraben_charm_exile_graveyards })".to_string(),
        _ => "None".to_string(),
    }
}

/// Alternative spell characteristics for Omen cards. Sagu Wildling's
/// Roost Seek half is a green sorcery that searches for a basic land; the
/// shared engine cast-method path owns its successful source shuffle.
fn omen_for(name: &str) -> String {
    match name {
        "Sagu Wildling" => {
            "Some(OmenDef { cost: Cost { pips: &[Pip::Colored(ManaColor::G)], generic: 0, x_count: 0 }, types: &[CardType::Sorcery], target_spec: TargetSpec::None, effect: omen_effect_search_basic_land })".to_string()
        }
        _ => "None".to_string(),
    }
}

fn omen_effect_recipe_for(name: &str) -> Option<AbilityEffectRecipe> {
    match name {
        "Sagu Wildling" => Some(AbilityEffectRecipe::SearchLibraryToHand {
            filter: LibrarySearchFilterRecipe::BasicLand,
            min_targets: 0,
            max_targets: 1,
            reveal_selected: true,
            shuffle: true,
        }),
        _ => None,
    }
}

/// Minimum number of creatures required to block one attacker. The engine
/// treats zero/one as ordinary blocking and enforces larger values against
/// the complete declaration, with Troll of Khazad-dum requiring three.
fn minimum_blockers_for(name: &str) -> u8 {
    match name {
        "Troll of Khazad-dum" => 3,
        _ => 1,
    }
}

/// Renders a mana cost string straight to a `Cost { .. }` literal wrapped in
/// `Some(..)`, for the one-off cost tables above (`plot_cost_for`/
/// `madness_cost_for`) that aren't full `CardJson` records.
fn cost_src(mana_cost: &str) -> String {
    let (pips, generic, x_count) = parse_cost(mana_cost);
    format!(
        "Some(Cost {{ pips: &[{}], generic: {generic}, x_count: {x_count} }})",
        pips.join(", ")
    )
}

fn generic_cost_reduction_for(name: &str) -> &'static str {
    match name {
        "Myr Enforcer" | "Thoughtcast" | "Refurbished Familiar" => {
            "Some(GenericCostReductionDef { generic_per_count: 1, count: DynamicCountDef::ControllerBattlefieldAnyType(&[CardType::Artifact]) })"
        }
        "Cryptic Serpent" | "Tolarian Terror" => {
            "Some(GenericCostReductionDef { generic_per_count: 1, count: DynamicCountDef::ControllerGraveyardAnyType(&[CardType::Instant, CardType::Sorcery]) })"
        }
        "Deem Inferior" => {
            "Some(GenericCostReductionDef { generic_per_count: 1, count: DynamicCountDef::ControllerDrawsThisTurn })"
        }
        "Of One Mind" => {
            "Some(GenericCostReductionDef { generic_per_count: 2, count: DynamicCountDef::ControllerHasCreatureWithAndWithoutSubtype(Subtype::Human) })"
        }
        _ => "None",
    }
}

fn ward_cost_for(name: &str) -> &'static str {
    match name {
        "Tolarian Terror" => "Some(WardCostDef::Generic(2))",
        _ => "None",
    }
}

fn equipment_for(name: &str) -> &'static str {
    match name {
        "Black Mage's Rod" => "Some(EquipmentDef { power_delta: 1, toughness_delta: 0, add_subtype: Some(Subtype::Wizard), controller_turn_keywords: Keywords::NONE, other_turn_keywords: Keywords::NONE, noncreature_spell_damage_to_each_opponent: 1, job_select: true })",
        "Hunter's Blowgun" => "Some(EquipmentDef { power_delta: 1, toughness_delta: 1, add_subtype: None, controller_turn_keywords: Keywords::DEATHTOUCH, other_turn_keywords: Keywords::REACH, noncreature_spell_damage_to_each_opponent: 0, job_select: false })",
        _ => "None",
    }
}

fn attachment_for(name: &str) -> &'static str {
    match name {
        "Bind the Monster" => "Some(AttachmentDef::AuraCreature { prevents_untap: true })",
        _ => "None",
    }
}

fn optional_additional_cost_for(name: &str) -> &'static str {
    match name {
        "Extract a Confession" | "Vitu-Ghazi Inspector" => {
            "Some(OptionalAdditionalCostDef::CollectEvidence { minimum_mana_value: 6 })"
        }
        "Troublemaker Ouphe" => "Some(OptionalAdditionalCostDef::Bargain)",
        _ => "None",
    }
}

fn changeling_for(name: &str) -> bool {
    name == "Masked Vandal"
}

/// Stable semantic binding for definition-owned triggered abilities. Runtime
/// construction lives in trigger.rs, while this token makes each selected
/// event, target, and effect part of the generated card database identity.
fn trigger_recipe_for(name: &str) -> &'static str {
    match name {
        "Guttersnipe" => "cast_instant_or_sorcery:damage_opponent:2",
        "Murmuring Mystic" => "cast_instant_or_sorcery:create_bird_illusion",
        "Voldaren Epicure" => "etb:damage_opponent:1:create_blood",
        "Generous Ent" => "etb:create_food",
        "Blood Fountain" => "etb:create_blood",
        "Sagu Wildling" | "Healer of the Glade" | "Spinewoods Paladin" => "etb:gain_life:3",
        "Gatecreeper Vine" => "etb:search_basic_land_or_gate_to_hand",
        "Sneaky Snacker" => "third_draw:return_source_to_battlefield_tapped",
        "Burning-Tree Emissary" => "etb:add_r_g",
        "Clockwork Percussionist" => "dies:impulse_top_one_until_owners_next_turn",
        "Ichor Wellspring" => "etb_and_dies:draw:1",
        "Experimental Synthesizer" => "etb_and_leaves:impulse_top_one_end_of_turn",
        "Goblin Bushwhacker" => "etb_if_kicked:pump_controlled_creatures:1:0:haste",
        "Faerie Miscreant" => "etb_if_another_same_definition:draw:1",
        "Faerie Seer" => "etb:scry:2",
        "Outlaw Medic" => "dies:draw:1",
        "Refurbished Familiar" => "etb:opponent_discard_else_draw",
        "Squadron Hawk" => "etb:search_up_to_three_same_definition_reveal_shuffle",
        "Bind the Monster" => "etb:tap_attached_then_attached_deals_power_to_aura_controller",
        "Harrier Strix" => "etb:target_any_permanent:tap",
        "Humbling Elder" => "etb:target_opponent_creature:pump:-2:0:eot",
        "Moon-Circuit Hacker" => {
            "combat_damage_player:may_draw:discard_unless_source_entered_this_turn:lki"
        }
        "Ninja of the Deep Hours" => "combat_damage_player:may_draw:1",
        "Saiba Cryptomancer" => "etb:target_creature:backup_1:hexproof",
        "Spellstutter Sprite" => "etb:target_spell_mv_at_most_controlled_faerie_count:counter",
        "Lembas" => {
            "etb:scry:1:draw:1;left_battlefield_to_graveyard:shuffle_source_into_owners_library"
        }
        "Weather the Storm" => "cast_self:storm_copies:frozen_turn_cast_count",
        "Masked Vandal" => {
            "etb:target_opponent_artifact_or_enchantment:may_exile_controller_creature_card_then_exile_target"
        }
        "Troublemaker Ouphe" => {
            "etb_if_bargained:target_opponent_artifact_or_enchantment:exile_target"
        }
        "Vitu-Ghazi Inspector" => {
            "etb_if_collect_evidence_6:target_creature:plus_one_counter:gain_life_2"
        }
        "Avenging Hunter" => "etb:take_initiative:undercity",
        _ => "none",
    }
}

fn intrinsic_basic_mana_color(card: &CardJson) -> Option<&'static str> {
    let is_basic_land = card.is_land
        && card.types.iter().any(|card_type| card_type == "Land")
        && card.supertypes.iter().any(|supertype| supertype == "Basic");
    if !is_basic_land {
        return None;
    }

    let intrinsic: Vec<&'static str> = card
        .subtypes
        .iter()
        .filter_map(|subtype| match subtype.as_str() {
            "Plains" => Some("W"),
            "Island" => Some("U"),
            "Swamp" => Some("B"),
            "Mountain" => Some("R"),
            "Forest" => Some("G"),
            _ => None,
        })
        .collect();
    if intrinsic.len() != 1 {
        panic!(
            "cards_v1.json: intrinsic basic land {:?} must have exactly one basic land subtype, got {:?}",
            card.name, card.subtypes
        );
    }
    let expected = intrinsic[0];
    if card.produces_mana.len() != 1 || card.produces_mana[0] != expected {
        panic!(
            "cards_v1.json: intrinsic basic land {:?} subtype requires produces_mana {:?}, got {:?}",
            card.name, expected, card.produces_mana
        );
    }
    Some(expected)
}

fn is_ordinary_permanent(card: &CardJson) -> bool {
    !card.is_land
        && !card.is_token
        && card
            .types
            .iter()
            .any(|card_type| matches!(card_type.as_str(), "Artifact" | "Creature" | "Enchantment"))
}

fn capability_src(capability: EngineCapabilityJson) -> &'static str {
    match capability {
        EngineCapabilityJson::NoEffect => "CardCapability::NoEffect",
        EngineCapabilityJson::Partial => "CardCapability::Partial",
        EngineCapabilityJson::Full => "CardCapability::Full",
    }
}

fn codegen(cards: &[CardJson]) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "// GENERATED by build.rs from data/cards_v1.json. Do not edit by hand."
    )
    .unwrap();
    writeln!(out).unwrap();

    let mut draw_card_counts = Vec::new();
    for card in cards {
        if let Special::DrawCards(count) = special_for(&card.name) {
            if !draw_card_counts.contains(&count) {
                draw_card_counts.push(count);
            }
        }
    }
    for count in draw_card_counts {
        writeln!(out, "fn spell_effect_draw_{count}() -> Option<EffectOp> {{").unwrap();
        writeln!(
            out,
            "    Some(EffectOp::DrawCards {{ player: PlayerRef::Controller, count: {count} }})"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    for card in cards {
        match special_for(&card.name) {
            Special::DrawThenCreateToken { draw, token } => {
                let function = card.name.to_ascii_lowercase().replace([' ', '\''], "_");
                writeln!(out, "fn spell_effect_{function}() -> Option<EffectOp> {{").unwrap();
                writeln!(out, "    let token = crate::card_def::card_id_by_name({token:?}).expect(\"{token} in CARD_DEFS\");").unwrap();
                writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
                writeln!(out, "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: {draw} }},").unwrap();
                writeln!(out, "        EffectOp::CreateToken {{ token_def: token, controller: PlayerRef::Controller }},").unwrap();
                writeln!(out, "    ]))").unwrap();
                writeln!(out, "}}").unwrap();
                writeln!(out).unwrap();
            }
            Special::GainPaidCostManaValueThenDraw { draw } => {
                let function = card.name.to_ascii_lowercase().replace([' ', '\''], "_");
                writeln!(out, "fn spell_effect_{function}() -> Option<EffectOp> {{").unwrap();
                writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
                writeln!(out, "        EffectOp::GainLifeEqualToPaidCostManaValue {{ player: PlayerRef::Controller }},").unwrap();
                writeln!(out, "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: {draw} }},").unwrap();
                writeln!(out, "    ]))").unwrap();
                writeln!(out, "}}").unwrap();
                writeln!(out).unwrap();
            }
            _ => {}
        }
    }

    if cards.iter().any(|card| {
        matches!(
            special_for(&card.name),
            Special::AddPlusOnePlusOneAndLifelinkCounters
        )
    }) {
        writeln!(
            out,
            "fn spell_effect_add_plus_one_plus_one_and_lifelink_counters() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::AddCountersToTarget {{ target_index: 0, optional: false, plus1_plus1: 1, lifelink: 1, stun: 0 }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    // Shared/one-off effect-program functions. Function *pointers* (not
    // owned EffectOp values) are what make a `static [CardDef; N]` array
    // possible: EffectOp contains Vec/Box and can't live in a const
    // initializer directly, but a `fn() -> Option<EffectOp>` can, and it
    // builds the (small) tree fresh each call.
    for (suffix, color) in [
        ("w", "W"),
        ("u", "U"),
        ("b", "B"),
        ("r", "R"),
        ("g", "G"),
        ("c", "C"),
    ] {
        let used = cards.iter().any(|card| {
            card.engine_capability != EngineCapabilityJson::NoEffect
                && has_activated_mana_ability(card)
                && card.produces_mana.iter().any(|produced| produced == color)
        });
        if !used {
            continue;
        }
        writeln!(out, "fn mana_ability_add_{suffix}() -> Option<EffectOp> {{").unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "        EffectOp::TapObject {{ object: ObjectRef::ThisSource }},"
        )
        .unwrap();
        writeln!(out, "        EffectOp::AddMana {{ player: PlayerRef::Controller, colors: vec![ManaColor::{color}] }},").unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    writeln!(
        out,
        "fn spell_effect_ordinary_permanent() -> Option<EffectOp> {{"
    )
    .unwrap();
    writeln!(out, "    Some(EffectOp::MoveObject {{ object: ObjectRef::ThisSource, to_zone: Zone::Battlefield }})").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::ExtractAConfession))
    {
        writeln!(
            out,
            "fn spell_effect_extract_a_confession() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Conditional {{").unwrap();
        writeln!(out, "        cond: EffectCond::OptionalAdditionalCostPaid(OptionalAdditionalCostDef::CollectEvidence {{ minimum_mana_value: 6 }}),").unwrap();
        writeln!(out, "        then: Box::new(EffectOp::SacrificeCreature {{ player: PlayerRef::Opponent, filter: CreatureSacrificeFilter::GreatestPower }}),").unwrap();
        writeln!(out, "        else_: Box::new(EffectOp::SacrificeCreature {{ player: PlayerRef::Opponent, filter: CreatureSacrificeFilter::Any }}),").unwrap();
        writeln!(out, "    }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::BindTheMonster))
    {
        writeln!(
            out,
            "fn spell_effect_bind_the_monster() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::PutSourceOntoBattlefieldAttachedToTarget {{ target: ObjectRef::Target(0) }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::Snap))
    {
        writeln!(out, "fn spell_effect_snap() -> Option<EffectOp> {{").unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "        EffectOp::MoveObject {{ object: ObjectRef::Target(0), to_zone: Zone::Hand }},"
        )
        .unwrap();
        writeln!(out, "        EffectOp::UntapUpToLands {{ chooser: PlayerRef::Controller, max_targets: 2 }},").unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::CleansingWildfire))
    {
        writeln!(
            out,
            "fn spell_effect_cleansing_wildfire() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(out, "        EffectOp::DestroyTargetLandThenMaySearchBasicTapped {{ object: ObjectRef::Target(0) }},").unwrap();
        writeln!(
            out,
            "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: 1 }},"
        )
        .unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::Duress))
    {
        writeln!(out, "fn spell_effect_duress() -> Option<EffectOp> {{").unwrap();
        writeln!(out, "    Some(EffectOp::RevealTargetHandChooseNoncreatureNonlandDiscard {{ player: PlayerRef::Target(0) }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::ToxinAnalysis))
    {
        writeln!(
            out,
            "fn spell_effect_toxin_analysis() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    let clue = crate::card_def::card_id_by_name(\"Clue Token\").expect(\"Clue Token in CARD_DEFS\");").unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(out, "        EffectOp::GrantKeywordTargetUntilEndOfTurn {{ object: ObjectRef::Target(0), keyword: Keywords::DEATHTOUCH | Keywords::LIFELINK }},").unwrap();
        writeln!(out, "        EffectOp::CreateToken {{ token_def: clue, controller: PlayerRef::Controller }},").unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::WeatherTheStorm))
    {
        writeln!(
            out,
            "fn spell_effect_weather_the_storm() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(
            out,
            "    Some(EffectOp::GainLife {{ player: PlayerRef::Controller, amount: 3 }})"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    // Emit each structured activated-ability effect once. These are the same
    // recipes used for CardDef source and the v3 card-database hash tokens,
    // so a cost/zone/effect change cannot update one surface and leave the
    // others silently stale.
    let mut activated_effects = Vec::new();
    for card in cards {
        for recipe in activated_ability_recipes_for(&card.name) {
            if !activated_effects.contains(&recipe.effect) {
                activated_effects.push(recipe.effect);
            }
        }
        if let Some(effect) = omen_effect_recipe_for(&card.name) {
            if !activated_effects.contains(&effect) {
                activated_effects.push(effect);
            }
        }
    }
    for effect in activated_effects {
        let function_name = ability_effect_fn_name(effect);
        writeln!(out, "fn {function_name}() -> EffectOp {{").unwrap();
        match effect {
            AbilityEffectRecipe::DrawCards(count) => {
                writeln!(
                    out,
                    "    EffectOp::DrawCards {{ player: PlayerRef::Controller, count: {count} }}"
                )
                .unwrap();
            }
            AbilityEffectRecipe::GainLife(amount) => {
                writeln!(
                    out,
                    "    EffectOp::GainLife {{ player: PlayerRef::Controller, amount: {amount} }}"
                )
                .unwrap();
            }
            AbilityEffectRecipe::CreateToken(name) => {
                writeln!(out, "    let token = crate::card_def::card_id_by_name({name:?}).expect(\"{name} in CARD_DEFS\");").unwrap();
                writeln!(out, "    EffectOp::CreateToken {{ token_def: token, controller: PlayerRef::Controller }}").unwrap();
            }
            AbilityEffectRecipe::DamageTarget(amount) => {
                writeln!(
                    out,
                    "    EffectOp::DealDamage {{ target: TargetRef::Target(0), amount: {amount} }}"
                )
                .unwrap();
            }
            AbilityEffectRecipe::MoveAllTargetsToHand => {
                writeln!(
                    out,
                    "    EffectOp::MoveAllTargets {{ to_zone: Zone::Hand }}"
                )
                .unwrap();
            }
            AbilityEffectRecipe::ExploreTarget => {
                writeln!(
                    out,
                    "    EffectOp::ExploreTarget {{ object: ObjectRef::Target(0) }}"
                )
                .unwrap();
            }
            AbilityEffectRecipe::DealDamageAnyTarget(amount) => {
                writeln!(
                    out,
                    "    EffectOp::DealDamage {{ target: TargetRef::Target(0), amount: {amount} }}"
                )
                .unwrap();
            }
            AbilityEffectRecipe::DamageAllCreatures {
                amount,
                filter: CreatureEffectFilterRecipe::WithoutKeyword(keyword),
            } => {
                writeln!(out, "    EffectOp::DamageAllCreatures {{ filter: CreatureFilter::WithoutKeyword(Keywords::{keyword}), amount: {amount} }}").unwrap();
            }
            AbilityEffectRecipe::ExileTargetPlayersGraveyard => {
                writeln!(
                    out,
                    "    EffectOp::ExilePlayersGraveyard {{ player: PlayerRef::Target(0) }}"
                )
                .unwrap();
            }
            AbilityEffectRecipe::ExileOneFromTargetPlayersGraveyard => {
                writeln!(
                    out,
                    "    EffectOp::ExileOneFromPlayersGraveyard {{ player: PlayerRef::Target(0) }}"
                )
                .unwrap();
            }
            AbilityEffectRecipe::ExileAllGraveyardsThenDraw(draw) => {
                writeln!(out, "    EffectOp::Sequence(vec![").unwrap();
                writeln!(out, "        EffectOp::ExileAllGraveyards,").unwrap();
                writeln!(out, "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: {draw} }},").unwrap();
                writeln!(out, "    ])").unwrap();
            }
            AbilityEffectRecipe::SearchLibraryToHand {
                filter,
                min_targets: 0,
                max_targets: 1,
                reveal_selected: true,
                shuffle: true,
            } => {
                let filter = library_search_filter_src(filter);
                writeln!(out, "    EffectOp::SearchLibraryToHand {{ player: PlayerRef::Controller, filter: {filter} }}").unwrap();
            }
            AbilityEffectRecipe::SearchLibraryToHand { .. } => {
                unreachable!("ability_effect_fn_name rejects unsupported search recipes")
            }
            AbilityEffectRecipe::UntapTarget => {
                writeln!(
                    out,
                    "    EffectOp::UntapObject {{ object: ObjectRef::Target(0) }}"
                )
                .unwrap();
            }
            AbilityEffectRecipe::GainLifeBattlefieldSubtypeCount(subtype) => {
                writeln!(out, "    EffectOp::GainLifeDynamic {{ player: PlayerRef::Controller, amount: DynamicValueDef::BattlefieldPermanentsWithSubtype(Subtype::{subtype}) }}").unwrap();
            }
            AbilityEffectRecipe::PumpTargetBattlefieldSubtypeCount(subtype) => {
                writeln!(out, "    EffectOp::PumpTargetUntilEndOfTurnDynamic {{ target: TargetRef::Target(0), power: DynamicValueDef::BattlefieldPermanentsWithSubtype(Subtype::{subtype}), toughness: DynamicValueDef::BattlefieldPermanentsWithSubtype(Subtype::{subtype}) }}").unwrap();
            }
            AbilityEffectRecipe::PumpTargetByControlledSubtypeCount(subtype) => {
                writeln!(
                    out,
                    "    EffectOp::PumpTargetByControlledSubtypeCount {{ target: ObjectRef::Target(0), subtype: {} }}",
                    subtype_variant(subtype)
                )
                .unwrap();
            }
            AbilityEffectRecipe::AttachSourceToTarget => {
                writeln!(
                    out,
                    "    EffectOp::AttachSourceToTarget {{ object: ObjectRef::Target(0) }}"
                )
                .unwrap();
            }
            AbilityEffectRecipe::AddStunCounterToOptionalTarget => {
                writeln!(out, "    EffectOp::AddCountersToTarget {{ target_index: 0, optional: true, plus1_plus1: 0, lifelink: 0, stun: 1 }}").unwrap();
            }
            AbilityEffectRecipe::DestroyTarget => {
                writeln!(
                    out,
                    "    EffectOp::DestroyObject {{ object: ObjectRef::Target(0) }}"
                )
                .unwrap();
            }
            AbilityEffectRecipe::DrawThenDiscard { draw, discard } => {
                writeln!(out, "    EffectOp::Sequence(vec![").unwrap();
                writeln!(out, "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: {draw} }},").unwrap();
                writeln!(out, "        EffectOp::DiscardCards {{ player: PlayerRef::Controller, count: {discard} }},").unwrap();
                writeln!(out, "    ])").unwrap();
            }
            AbilityEffectRecipe::PutSourceOntoBattlefieldTappedAndAttacking => {
                writeln!(
                    out,
                    "    EffectOp::PutSourceOntoBattlefieldTappedAndAttacking"
                )
                .unwrap();
            }
            AbilityEffectRecipe::MoveAllTargetsToExile => {
                writeln!(
                    out,
                    "    EffectOp::MoveAllTargets {{ to_zone: Zone::Exile }}"
                )
                .unwrap();
            }
            AbilityEffectRecipe::AddMinusOneMinusOneCounter => {
                writeln!(
                    out,
                    "    EffectOp::AddMinusOneMinusOneCounter {{ object: ObjectRef::Target(0) }}"
                )
                .unwrap();
            }
            AbilityEffectRecipe::SearchLibraryToBattlefieldTapped { filter } => {
                let filter = library_search_filter_src(filter);
                writeln!(out, "    EffectOp::SearchLibraryToBattlefieldTapped {{ player: PlayerRef::Controller, filter: {filter} }}").unwrap();
            }
        }
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::ChainLightning))
    {
        // "Deals 3 damage to any target. Then that player or that
        // permanent's controller may pay {R}{R}. If the player does, they
        // may copy this spell...": mandatory damage first (identical shape
        // to `BurnAnyTarget(3)`), then the resolution-suspending copy offer
        // -- see `EffectOp::OfferAffectedPlayerSpellCopy`'s doc.
        writeln!(
            out,
            "fn spell_effect_chain_lightning() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "        EffectOp::DealDamage {{ target: TargetRef::Target(0), amount: 3 }},"
        )
        .unwrap();
        writeln!(
            out,
            "        EffectOp::OfferAffectedPlayerSpellCopy {{ affected: TargetRef::Target(0) }},"
        )
        .unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    let mut target_player_draw_shapes = Vec::new();
    for card in cards {
        if let Special::TargetPlayerDraw { draw } = special_for(&card.name) {
            if !target_player_draw_shapes.contains(&draw) {
                target_player_draw_shapes.push(draw);
            }
        }
    }
    for draw in target_player_draw_shapes {
        writeln!(
            out,
            "fn spell_effect_target_player_draw_{draw}() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(
            out,
            "    Some(EffectOp::DrawCards {{ player: PlayerRef::Target(0), count: {draw} }})"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::EndTheFestivities))
    {
        // "Deals 1 damage to each opponent and each creature and
        // planeswalker they control." No planeswalker exists in this pool,
        // so `DamageOpponentAndTheirCreatures` (which only hits the
        // opponent + their creatures) already covers 100% of the reachable
        // text -- see `EffectOp::DamageOpponentAndTheirCreatures`'s doc.
        writeln!(
            out,
            "fn spell_effect_end_the_festivities() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(
            out,
            "    Some(EffectOp::DamageOpponentAndTheirCreatures {{ amount: 1 }})"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::GalvanicBlast))
    {
        // Metalcraft -- 4 damage instead of 2 if you control 3+ artifacts.
        writeln!(
            out,
            "fn spell_effect_galvanic_blast() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Conditional {{").unwrap();
        writeln!(out, "        cond: EffectCond::ControlsArtifactCount(3),").unwrap();
        writeln!(out, "        then: Box::new(EffectOp::DealDamage {{ target: TargetRef::Target(0), amount: 4 }}),").unwrap();
        writeln!(out, "        else_: Box::new(EffectOp::DealDamage {{ target: TargetRef::Target(0), amount: 2 }}),").unwrap();
        writeln!(out, "    }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::RallyAtTheHornburg))
    {
        // "Create two 1/1 white Human Soldier creature tokens. Humans you
        // control gain haste until end of turn." The two just-created
        // tokens are themselves Human, so they're inside the "Humans you
        // control" set the pump snapshots -- see `EffectOp::PumpControlled`'s
        // doc for why sequencing (create both tokens, *then* pump) is what
        // makes that true.
        writeln!(
            out,
            "fn spell_effect_rally_at_the_hornburg() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    let human_soldier = crate::card_def::card_id_by_name(\"Human Soldier Token\").expect(\"Human Soldier Token in CARD_DEFS\");").unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(out, "        EffectOp::CreateToken {{ token_def: human_soldier, controller: PlayerRef::Controller }},").unwrap();
        writeln!(out, "        EffectOp::CreateToken {{ token_def: human_soldier, controller: PlayerRef::Controller }},").unwrap();
        writeln!(
            out,
            "        EffectOp::PumpControlled {{ filter: CreatureFilter::ControlledWithSubtype(Subtype::Human), power: 0, toughness: 0, grant_haste: true }},"
        )
        .unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::RecklessImpulse))
    {
        // "Exile the top two cards of your library. Until the end of your
        // next turn, you may play those cards."
        writeln!(
            out,
            "fn spell_effect_reckless_impulse() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::ImpulseDraw {{ count: 2, duration: ImpulseDuration::UntilOwnersNextTurn }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::WindingWay))
    {
        // Printed order is policy semantics: zero is Creature, one is Land.
        // The choice is made during resolution before the public reveal.
        writeln!(out, "fn spell_effect_winding_way() -> Option<EffectOp> {{").unwrap();
        writeln!(out, "    Some(EffectOp::Choice {{").unwrap();
        writeln!(out, "        controller: PlayerRef::Controller,").unwrap();
        writeln!(out, "        options: vec![").unwrap();
        for card_type in ["Creature", "Land"] {
            writeln!(out, "            EffectOp::RevealTopAndPartitionByType {{").unwrap();
            writeln!(out, "                player: PlayerRef::Controller,").unwrap();
            writeln!(out, "                count: 4,").unwrap();
            writeln!(out, "                card_type: CardType::{card_type},").unwrap();
            writeln!(out, "                matching_to: Zone::Hand,").unwrap();
            writeln!(out, "                rest_to: Zone::Graveyard,").unwrap();
            writeln!(out, "            }},").unwrap();
        }
        writeln!(out, "        ],").unwrap();
        writeln!(out, "    }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    let mut typed_top_partition_shapes = Vec::new();
    for card in cards {
        if let Special::LookTopSelectByTypeToHandBottomRest { look, card_type } =
            special_for(&card.name)
        {
            let shape = (look, card_type);
            if !typed_top_partition_shapes.contains(&shape) {
                typed_top_partition_shapes.push(shape);
            }
        }
    }
    for (look, card_type) in typed_top_partition_shapes {
        let suffix = card_type.to_ascii_lowercase();
        writeln!(
            out,
            "fn spell_effect_look_top_{look}_select_{suffix}_to_hand_bottom_rest() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::LookTopSelectByTypeToHandBottomRest {{ player: PlayerRef::Controller, count: {look}, card_type: CardType::{card_type} }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    let mut look_reorder_may_shuffle_then_draw_shapes = Vec::new();
    for card in cards {
        if let Special::LookReorderMayShuffleThenDraw { look, draw } = special_for(&card.name) {
            let shape = (look, draw);
            if !look_reorder_may_shuffle_then_draw_shapes.contains(&shape) {
                look_reorder_may_shuffle_then_draw_shapes.push(shape);
            }
        }
    }
    for (look, draw) in look_reorder_may_shuffle_then_draw_shapes {
        writeln!(
            out,
            "fn spell_effect_look_reorder_may_shuffle_then_draw_{look}_{draw}() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "        EffectOp::LookAtLibraryTopAndReorder {{ player: PlayerRef::Controller, count: {look} }},"
        )
        .unwrap();
        writeln!(
            out,
            "        EffectOp::MayShuffleLibrary {{ player: PlayerRef::Controller }},"
        )
        .unwrap();
        writeln!(
            out,
            "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: {draw} }},"
        )
        .unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    let mut draw_then_put_hand_on_library_top_shapes = Vec::new();
    for card in cards {
        if let Special::DrawThenPutHandOnLibraryTop { draw, put } = special_for(&card.name) {
            let shape = (draw, put);
            if !draw_then_put_hand_on_library_top_shapes.contains(&shape) {
                draw_then_put_hand_on_library_top_shapes.push(shape);
            }
        }
    }
    for (draw, put) in draw_then_put_hand_on_library_top_shapes {
        writeln!(
            out,
            "fn spell_effect_draw_then_put_hand_on_library_top_{draw}_{put}() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: {draw} }},"
        )
        .unwrap();
        writeln!(
            out,
            "        EffectOp::PutCardsFromHandOnLibraryTop {{ player: PlayerRef::Controller, count: {put} }},"
        )
        .unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    let mut scry_then_draw_shapes = Vec::new();
    for card in cards {
        if let Special::ScryThenDraw { scry, draw } = special_for(&card.name) {
            let shape = (scry, draw);
            if !scry_then_draw_shapes.contains(&shape) {
                scry_then_draw_shapes.push(shape);
            }
        }
    }
    for (scry, draw) in scry_then_draw_shapes {
        writeln!(
            out,
            "fn spell_effect_scry_then_draw_{scry}_{draw}() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "        EffectOp::Scry {{ player: PlayerRef::Controller, count: {scry} }},"
        )
        .unwrap();
        writeln!(
            out,
            "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: {draw} }},"
        )
        .unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::DeemInferior))
    {
        writeln!(
            out,
            "fn spell_effect_deem_inferior() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::PutObjectInOwnersLibrarySecondOrBottom {{ object: ObjectRef::Target(0) }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::TapAndSkipNextUntap))
    {
        writeln!(
            out,
            "fn spell_effect_tap_and_skip_next_untap() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "        EffectOp::TapObject {{ object: ObjectRef::Target(0) }},"
        )
        .unwrap();
        writeln!(
            out,
            "        EffectOp::SkipNextUntap {{ object: ObjectRef::Target(0) }},"
        )
        .unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards.iter().any(|card| {
        matches!(
            special_for(&card.name),
            Special::DestroyNonlegendaryCreature
        )
    }) {
        writeln!(
            out,
            "fn spell_effect_destroy_nonlegendary_creature() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Conditional {{").unwrap();
        writeln!(
            out,
            "        cond: EffectCond::TargetInZone(0, Zone::Battlefield),"
        )
        .unwrap();
        writeln!(
            out,
            "        then: Box::new(EffectOp::DestroyObject {{ object: ObjectRef::Target(0) }}),"
        )
        .unwrap();
        writeln!(out, "        else_: Box::new(EffectOp::Sequence(vec![])),").unwrap();
        writeln!(out, "    }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    let mut graveyard_return_life_amounts = Vec::new();
    for card in cards {
        if let Special::ReturnCreatureOrLandFromGraveyardAndGainLife { amount } =
            special_for(&card.name)
        {
            if !graveyard_return_life_amounts.contains(&amount) {
                graveyard_return_life_amounts.push(amount);
            }
        }
    }
    for amount in graveyard_return_life_amounts {
        writeln!(
            out,
            "fn spell_effect_return_creature_or_land_from_graveyard_and_gain_life_{amount}() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "        EffectOp::MoveObject {{ object: ObjectRef::Target(0), to_zone: Zone::Hand }},"
        )
        .unwrap();
        writeln!(
            out,
            "        EffectOp::GainLife {{ player: PlayerRef::Controller, amount: {amount} }},"
        )
        .unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    let mut damage_without_subtype_shapes = Vec::new();
    for card in cards {
        if let Special::DamageEachCreatureWithoutSubtype {
            amount,
            excluded_subtype,
        } = special_for(&card.name)
        {
            let shape = (amount, excluded_subtype);
            if !damage_without_subtype_shapes.contains(&shape) {
                damage_without_subtype_shapes.push(shape);
            }
        }
    }
    for (amount, excluded_subtype) in damage_without_subtype_shapes {
        let subtype = subtype_variant(excluded_subtype);
        let suffix = excluded_subtype.to_ascii_lowercase();
        writeln!(out, "fn spell_effect_damage_each_creature_without_{suffix}_{amount}() -> Option<EffectOp> {{").unwrap();
        writeln!(out, "    Some(EffectOp::DamageEachCreatureWithoutSubtype {{ amount: {amount}, excluded_subtype: {subtype} }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    let mut mill_then_draw_shapes = Vec::new();
    for card in cards {
        if let Special::MillThenDraw { player, mill, draw } = special_for(&card.name) {
            let shape = (player, mill, draw);
            if !mill_then_draw_shapes.contains(&shape) {
                mill_then_draw_shapes.push(shape);
            }
        }
    }
    for (player, mill, draw) in mill_then_draw_shapes {
        let (player_suffix, player_src) = match player {
            MillPlayer::Controller => ("controller", "PlayerRef::Controller"),
            MillPlayer::Target0 => ("target_0", "PlayerRef::Target(0)"),
        };
        writeln!(
            out,
            "fn spell_effect_mill_then_draw_{player_suffix}_{mill}_{draw}() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "        EffectOp::MillCards {{ player: {player_src}, count: {mill} }},"
        )
        .unwrap();
        writeln!(
            out,
            "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: {draw} }},"
        )
        .unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    let draw_then_discard_shapes: Vec<(i32, i32)> = cards
        .iter()
        .filter_map(|c| match special_for(&c.name) {
            Special::DrawThenDiscard { draw, discard } => Some((draw, discard)),
            _ => None,
        })
        .collect();
    for (draw, discard) in draw_then_discard_shapes {
        writeln!(
            out,
            "fn spell_effect_draw_then_discard_{draw}_{discard}() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: {draw} }},"
        )
        .unwrap();
        writeln!(
            out,
            "        EffectOp::DiscardCards {{ player: PlayerRef::Controller, count: {discard} }},"
        )
        .unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::GrabThePrize))
    {
        writeln!(
            out,
            "fn spell_effect_grab_the_prize() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: 2 }},"
        )
        .unwrap();
        writeln!(out, "        EffectOp::Conditional {{").unwrap();
        writeln!(
            out,
            "            cond: EffectCond::DiscardedNonLandForCost,"
        )
        .unwrap();
        writeln!(out, "            then: Box::new(EffectOp::DealDamage {{ target: TargetRef::Opponent, amount: 2 }}),").unwrap();
        writeln!(
            out,
            "            else_: Box::new(EffectOp::Sequence(vec![])),"
        )
        .unwrap();
        writeln!(out, "        }},").unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::HighwayRobbery))
    {
        // "You may discard a card or sacrifice a land. If you do, draw two
        // cards." -- DoIfCostPaid(OrCost(DiscardCardCost, SacrificeTargetCost)).
        writeln!(
            out,
            "fn spell_effect_highway_robbery() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::MayPayCostThen {{").unwrap();
        writeln!(out, "        discard: 1,").unwrap();
        writeln!(out, "        sacrifice_lands: 1,").unwrap();
        writeln!(out, "        then: Box::new(EffectOp::DrawCards {{ player: PlayerRef::Controller, count: 2 }}),").unwrap();
        writeln!(out, "    }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::SearingBlaze))
    {
        // 1 damage to target player + 1 damage to target creature that
        // player controls; landfall bumps both to 3. The creature-damage
        // leaf is individually fizzle-guarded (608.2b: Searing Blaze still
        // hits the player even if the creature target became illegal --
        // the player target can't, so the whole spell can never fully
        // fizzle in this pool).
        writeln!(
            out,
            "fn spell_effect_searing_blaze() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Conditional {{").unwrap();
        writeln!(out, "        cond: EffectCond::LandfallThisTurn,").unwrap();
        writeln!(out, "        then: Box::new(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "            EffectOp::DealDamage {{ target: TargetRef::Target(0), amount: 3 }},"
        )
        .unwrap();
        writeln!(out, "            EffectOp::Conditional {{").unwrap();
        writeln!(
            out,
            "                cond: EffectCond::TargetInZone(1, Zone::Battlefield),"
        )
        .unwrap();
        writeln!(out, "                then: Box::new(EffectOp::DealDamage {{ target: TargetRef::Target(1), amount: 3 }}),").unwrap();
        writeln!(
            out,
            "                else_: Box::new(EffectOp::Sequence(vec![])),"
        )
        .unwrap();
        writeln!(out, "            }},").unwrap();
        writeln!(out, "        ])),").unwrap();
        writeln!(out, "        else_: Box::new(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "            EffectOp::DealDamage {{ target: TargetRef::Target(0), amount: 1 }},"
        )
        .unwrap();
        writeln!(out, "            EffectOp::Conditional {{").unwrap();
        writeln!(
            out,
            "                cond: EffectCond::TargetInZone(1, Zone::Battlefield),"
        )
        .unwrap();
        writeln!(out, "                then: Box::new(EffectOp::DealDamage {{ target: TargetRef::Target(1), amount: 1 }}),").unwrap();
        writeln!(
            out,
            "                else_: Box::new(EffectOp::Sequence(vec![])),"
        )
        .unwrap();
        writeln!(out, "            }},").unwrap();
        writeln!(out, "        ])),").unwrap();
        writeln!(out, "    }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    // Reusable "counter target spell" resolution program. Target filters
    // belong to `TargetSpec`; the effect itself is identical for
    // Counterspell, Dispel, and both Elemental Blasts, and is nested under
    // Pyroblast/Hydroblast's resolution-time color check. The zone guard
    // makes a stale target a no-op, while `EffectOp::MoveObject` owns
    // physical-card, flashback, and virtual-copy departure semantics.
    writeln!(out, "fn counter_target_spell_effect() -> EffectOp {{").unwrap();
    writeln!(out, "    EffectOp::Conditional {{").unwrap();
    writeln!(
        out,
        "        cond: EffectCond::TargetInZone(0, Zone::Stack),"
    )
    .unwrap();
    writeln!(out, "        then: Box::new(EffectOp::MoveObject {{ object: ObjectRef::Target(0), to_zone: Zone::Graveyard }}),").unwrap();
    writeln!(out, "        else_: Box::new(EffectOp::Sequence(vec![])),").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "fn spell_effect_counter_target() -> Option<EffectOp> {{"
    )
    .unwrap();
    writeln!(out, "    Some(counter_target_spell_effect())").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    let mut counter_unless_generics = cards
        .iter()
        .filter_map(|card| match special_for(&card.name) {
            Special::CounterUnlessPaysGeneric { generic, .. } => Some(generic),
            _ => None,
        })
        .collect::<Vec<_>>();
    counter_unless_generics.sort_unstable();
    counter_unless_generics.dedup();
    for generic in counter_unless_generics {
        writeln!(
            out,
            "fn spell_effect_counter_target_unless_pays_generic_{generic}() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(
            out,
            "    Some(EffectOp::CounterTargetUnlessPaysGeneric {{ target: TargetRef::Target(0), generic: {generic} }})"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::SteelSabotage))
    {
        writeln!(
            out,
            "fn mode2_effect_return_target_permanent_to_owners_hand() -> EffectOp {{"
        )
        .unwrap();
        writeln!(out, "    EffectOp::Conditional {{").unwrap();
        writeln!(
            out,
            "        cond: EffectCond::TargetInZone(0, Zone::Battlefield),"
        )
        .unwrap();
        writeln!(out, "        then: Box::new(EffectOp::MoveObject {{ object: ObjectRef::Target(0), to_zone: Zone::Hand }}),").unwrap();
        writeln!(out, "        else_: Box::new(EffectOp::Sequence(vec![])),").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::PiracyCharm))
    {
        writeln!(
            out,
            "fn spell_effect_piracy_charm_islandwalk() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::GrantKeywordTargetUntilEndOfTurn {{ object: ObjectRef::Target(0), keyword: Keywords::ISLANDWALK }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "fn mode2_effect_piracy_charm_pump() -> EffectOp {{").unwrap();
        writeln!(out, "    EffectOp::PumpTargetUntilEndOfTurnDynamic {{ target: TargetRef::Target(0), power: DynamicValueDef::Fixed(2), toughness: DynamicValueDef::Fixed(-1) }}").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "fn mode3_effect_piracy_charm_discard() -> EffectOp {{").unwrap();
        writeln!(
            out,
            "    EffectOp::DiscardCards {{ player: PlayerRef::Target(0), count: 1 }}"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::CastIntoTheFire))
    {
        writeln!(
            out,
            "fn spell_effect_cast_into_the_fire_damage() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::DamageAllTargets {{ amount: 1 }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "fn mode2_effect_cast_into_the_fire_exile_artifact() -> EffectOp {{"
        )
        .unwrap();
        writeln!(
            out,
            "    EffectOp::MoveAllTargets {{ to_zone: Zone::Exile }}"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::DustToDust))
    {
        writeln!(out, "fn spell_effect_dust_to_dust() -> Option<EffectOp> {{").unwrap();
        writeln!(out, "    Some(EffectOp::ExileAllArtifactTargets)").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::ThrabenCharm))
    {
        writeln!(
            out,
            "fn spell_effect_thraben_charm_damage() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::DealDamageByControlledCreatureCount {{ target: TargetRef::Target(0), multiplier: 2 }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "fn mode2_effect_thraben_charm_destroy_enchantment() -> EffectOp {{"
        )
        .unwrap();
        writeln!(
            out,
            "    EffectOp::DestroyObject {{ object: ObjectRef::Target(0) }}"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "fn mode3_effect_thraben_charm_exile_graveyards() -> EffectOp {{"
        )
        .unwrap();
        writeln!(out, "    EffectOp::ExileTargetPlayersGraveyards").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::ColorBlast { .. }))
    {
        // The "if it's [color]" half used by Pyroblast/Hydroblast. Color is
        // evaluated during resolution; their TargetSpec remains permissive.
        writeln!(
            out,
            "fn counter_target_spell_if_color_effect(color: ManaColor) -> EffectOp {{"
        )
        .unwrap();
        writeln!(out, "    EffectOp::Conditional {{").unwrap();
        writeln!(out, "        cond: EffectCond::TargetIsColor(0, color),").unwrap();
        writeln!(
            out,
            "        then: Box::new(counter_target_spell_effect()),"
        )
        .unwrap();
        writeln!(out, "        else_: Box::new(EffectOp::Sequence(vec![])),").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();

        for color in [BlastColor::Blue, BlastColor::Red] {
            writeln!(
                out,
                "fn spell_effect_counter_target_if_{}() -> Option<EffectOp> {{",
                color.suffix()
            )
            .unwrap();
            writeln!(
                out,
                "    Some(counter_target_spell_if_color_effect({}))",
                color.mana_variant()
            )
            .unwrap();
            writeln!(out, "}}").unwrap();
            writeln!(out).unwrap();
        }

        // Shared destroy program for the Elemental Blasts, whose color is a
        // targeting restriction and therefore already revalidated by the
        // engine before resolution.
        writeln!(
            out,
            "fn mode2_effect_destroy_target_permanent() -> EffectOp {{"
        )
        .unwrap();
        writeln!(out, "    EffectOp::Conditional {{").unwrap();
        writeln!(
            out,
            "        cond: EffectCond::TargetInZone(0, Zone::Battlefield),"
        )
        .unwrap();
        writeln!(
            out,
            "        then: Box::new(EffectOp::DestroyObject {{ object: ObjectRef::Target(0) }}),"
        )
        .unwrap();
        writeln!(out, "        else_: Box::new(EffectOp::Sequence(vec![])),").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "fn destroy_target_permanent_if_color_effect(color: ManaColor) -> EffectOp {{"
        )
        .unwrap();
        writeln!(out, "    EffectOp::Conditional {{").unwrap();
        writeln!(out, "        cond: EffectCond::And(").unwrap();
        writeln!(
            out,
            "            Box::new(EffectCond::TargetInZone(0, Zone::Battlefield)),"
        )
        .unwrap();
        writeln!(
            out,
            "            Box::new(EffectCond::TargetIsColor(0, color)),"
        )
        .unwrap();
        writeln!(out, "        ),").unwrap();
        writeln!(
            out,
            "        then: Box::new(EffectOp::DestroyObject {{ object: ObjectRef::Target(0) }}),"
        )
        .unwrap();
        writeln!(out, "        else_: Box::new(EffectOp::Sequence(vec![])),").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();

        for color in [BlastColor::Blue, BlastColor::Red] {
            writeln!(
                out,
                "fn mode2_effect_destroy_target_permanent_if_{}() -> EffectOp {{",
                color.suffix()
            )
            .unwrap();
            writeln!(
                out,
                "    destroy_target_permanent_if_color_effect({})",
                color.mana_variant()
            )
            .unwrap();
            writeln!(out, "}}").unwrap();
            writeln!(out).unwrap();
        }
    }

    let burn_amounts: BTreeSetLike = cards
        .iter()
        .filter_map(|c| match special_for(&c.name) {
            Special::BurnAnyTarget(a) => Some(a),
            _ => None,
        })
        .collect();
    for amount in burn_amounts.values() {
        writeln!(
            out,
            "fn spell_effect_burn_any_target_{amount}() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(
            out,
            "    Some(EffectOp::DealDamage {{ target: TargetRef::Target(0), amount: {amount} }})"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards.iter().any(|card| {
        matches!(
            special_for(&card.name),
            Special::ReturnOwnGraveyardCreatureToBattlefield
        )
    }) {
        writeln!(
            out,
            "fn spell_effect_return_own_graveyard_creature_to_battlefield() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::MoveObject {{ object: ObjectRef::Target(0), to_zone: Zone::Battlefield }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::SearchForestToHand))
    {
        writeln!(
            out,
            "fn spell_effect_search_forest_to_hand() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::SearchLibraryToHand {{ player: PlayerRef::Controller, filter: LibraryCardFilter::LandWithSubtype(Subtype::Forest) }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards.iter().any(|card| {
        matches!(
            special_for(&card.name),
            Special::DamageCannotBePreventedThisTurn
        )
    }) {
        writeln!(
            out,
            "fn spell_effect_damage_cannot_be_prevented_this_turn() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::DamageCannotBePreventedThisTurn)").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::PrismaticStrands))
    {
        writeln!(
            out,
            "fn spell_effect_prismatic_strands() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::PreventDamageFromChosenColorUntilEndOfTurn {{ player: PlayerRef::Controller }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::MonstrousEmergence))
    {
        writeln!(
            out,
            "fn spell_effect_monstrous_emergence() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::DealDamageToTargetEqualToChosenCostCreaturePower {{ target: TargetRef::Target(0) }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|card| matches!(special_for(&card.name), Special::NyxbornHydra))
    {
        writeln!(
            out,
            "fn spell_effect_nyxborn_hydra() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(
            out,
            "    Some(EffectOp::PutSourceOntoBattlefieldWithXPlusOneCounters)"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards.iter().any(|card| {
        card.name == "The Modern Age" && card.engine_capability != EngineCapabilityJson::NoEffect
    }) {
        writeln!(out, "fn saga_chapter_modern_age_loot() -> EffectOp {{").unwrap();
        writeln!(out, "    EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: 1 }},"
        )
        .unwrap();
        writeln!(
            out,
            "        EffectOp::DiscardCards {{ player: PlayerRef::Controller, count: 1 }},"
        )
        .unwrap();
        writeln!(out, "    ])").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "fn saga_chapter_modern_age_transform() -> EffectOp {{").unwrap();
        writeln!(out, "    EffectOp::TransformSagaSource").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    // ---- CARD_DEFS -------------------------------------------------
    writeln!(out, "pub static CARD_DEFS: [CardDef; {}] = [", cards.len()).unwrap();
    for c in cards {
        let (pips, generic, x_count) = parse_cost(&c.mana_cost);
        let special = special_for(&c.name);

        let types_src = c
            .types
            .iter()
            .map(|t| format!("CardType::{}", card_type_variant(t)))
            .collect::<Vec<_>>()
            .join(", ");
        let subtypes_src = c
            .subtypes
            .iter()
            .map(|s| subtype_variant(s))
            .collect::<Vec<_>>()
            .join(", ");
        let supertypes_src = c
            .supertypes
            .iter()
            .map(|t| format!("Supertype::{}", supertype_variant(t)))
            .collect::<Vec<_>>()
            .join(", ");
        let produces_src = c
            .produces_mana
            .iter()
            .map(|m| format!("ManaColor::{}", color_variant(m)))
            .collect::<Vec<_>>()
            .join(", ");
        let colors_src = c
            .colors
            .iter()
            .map(|m| format!("ManaColor::{}", color_variant(m)))
            .collect::<Vec<_>>()
            .join(", ");
        let pips_src = pips.join(", ");
        let power_src = match c.power {
            Some(p) => format!("Some({p})"),
            None => "None".to_string(),
        };
        let toughness_src = match c.toughness {
            Some(t) => format!("Some({t})"),
            None => "None".to_string(),
        };

        let (target_spec_src, mut spell_effect_src, mut mana_ability_src) = match special {
            Special::None => (
                "TargetSpec::None",
                "no_effect".to_string(),
                "no_effect".to_string(),
            ),
            Special::GreatFurnace => (
                "TargetSpec::None",
                "no_effect".to_string(),
                "mana_ability_add_r".to_string(),
            ),
            Special::DrawCards(count) => (
                "TargetSpec::None",
                format!("spell_effect_draw_{count}"),
                "no_effect".to_string(),
            ),
            Special::BurnAnyTarget(amount) => (
                "TargetSpec::AnyTarget",
                format!("spell_effect_burn_any_target_{amount}"),
                "no_effect".to_string(),
            ),
            Special::ChainLightning => (
                "TargetSpec::AnyTarget",
                "spell_effect_chain_lightning".to_string(),
                "no_effect".to_string(),
            ),
            Special::TargetPlayerDraw { draw } => (
                "TargetSpec::AnyPlayer",
                format!("spell_effect_target_player_draw_{draw}"),
                "no_effect".to_string(),
            ),
            Special::DrawThenDiscard { draw, discard } => (
                "TargetSpec::None",
                format!("spell_effect_draw_then_discard_{draw}_{discard}"),
                "no_effect".to_string(),
            ),
            Special::ExtractAConfession => (
                "TargetSpec::None",
                "spell_effect_extract_a_confession".to_string(),
                "no_effect".to_string(),
            ),
            Special::GrabThePrize => (
                "TargetSpec::None",
                "spell_effect_grab_the_prize".to_string(),
                "no_effect".to_string(),
            ),
            Special::HighwayRobbery => (
                "TargetSpec::None",
                "spell_effect_highway_robbery".to_string(),
                "no_effect".to_string(),
            ),
            Special::SearingBlaze => (
                "TargetSpec::PlayerThenTheirCreature",
                "spell_effect_searing_blaze".to_string(),
                "no_effect".to_string(),
            ),
            Special::CounterTarget(filter) => (
                filter.target_spec(),
                "spell_effect_counter_target".to_string(),
                "no_effect".to_string(),
            ),
            Special::SteelSabotage => (
                StackSpellFilter::Artifact.target_spec(),
                "spell_effect_counter_target".to_string(),
                "no_effect".to_string(),
            ),
            Special::PiracyCharm => (
                "TargetSpec::Creature",
                "spell_effect_piracy_charm_islandwalk".to_string(),
                "no_effect".to_string(),
            ),
            Special::CastIntoTheFire => (
                "TargetSpec::UpToTwoCreatures",
                "spell_effect_cast_into_the_fire_damage".to_string(),
                "no_effect".to_string(),
            ),
            Special::DustToDust => (
                "TargetSpec::ExactlyTwoArtifactPermanents",
                "spell_effect_dust_to_dust".to_string(),
                "no_effect".to_string(),
            ),
            Special::ThrabenCharm => (
                "TargetSpec::Creature",
                "spell_effect_thraben_charm_damage".to_string(),
                "no_effect".to_string(),
            ),
            Special::CounterUnlessPaysGeneric { filter, generic } => (
                filter.target_spec(),
                format!("spell_effect_counter_target_unless_pays_generic_{generic}"),
                "no_effect".to_string(),
            ),
            Special::ColorBlast {
                checked_color,
                filter_timing: BlastFilterTiming::Targeting,
            } => (
                checked_color.filtered_spell_target_spec(),
                "spell_effect_counter_target".to_string(),
                "no_effect".to_string(),
            ),
            Special::ColorBlast {
                checked_color,
                filter_timing: BlastFilterTiming::Resolution,
            } => (
                "TargetSpec::AnySpellOnStack",
                format!("spell_effect_counter_target_if_{}", checked_color.suffix()),
                "no_effect".to_string(),
            ),
            Special::EndTheFestivities => (
                "TargetSpec::None",
                "spell_effect_end_the_festivities".to_string(),
                "no_effect".to_string(),
            ),
            Special::GalvanicBlast => (
                "TargetSpec::AnyTarget",
                "spell_effect_galvanic_blast".to_string(),
                "no_effect".to_string(),
            ),
            Special::RallyAtTheHornburg => (
                "TargetSpec::None",
                "spell_effect_rally_at_the_hornburg".to_string(),
                "no_effect".to_string(),
            ),
            Special::RecklessImpulse => (
                "TargetSpec::None",
                "spell_effect_reckless_impulse".to_string(),
                "no_effect".to_string(),
            ),
            Special::WindingWay => (
                "TargetSpec::None",
                "spell_effect_winding_way".to_string(),
                "no_effect".to_string(),
            ),
            Special::LookTopSelectByTypeToHandBottomRest { look, card_type } => (
                "TargetSpec::None",
                format!(
                    "spell_effect_look_top_{look}_select_{}_to_hand_bottom_rest",
                    card_type.to_ascii_lowercase()
                ),
                "no_effect".to_string(),
            ),
            Special::LookReorderMayShuffleThenDraw { look, draw } => (
                "TargetSpec::None",
                format!("spell_effect_look_reorder_may_shuffle_then_draw_{look}_{draw}"),
                "no_effect".to_string(),
            ),
            Special::DrawThenPutHandOnLibraryTop { draw, put } => (
                "TargetSpec::None",
                format!("spell_effect_draw_then_put_hand_on_library_top_{draw}_{put}"),
                "no_effect".to_string(),
            ),
            Special::ScryThenDraw { scry, draw } => (
                "TargetSpec::None",
                format!("spell_effect_scry_then_draw_{scry}_{draw}"),
                "no_effect".to_string(),
            ),
            Special::DeemInferior => (
                "TargetSpec::NonlandPermanent",
                "spell_effect_deem_inferior".to_string(),
                "no_effect".to_string(),
            ),
            Special::TapAndSkipNextUntap => (
                "TargetSpec::Creature",
                "spell_effect_tap_and_skip_next_untap".to_string(),
                "no_effect".to_string(),
            ),
            Special::DestroyNonlegendaryCreature => (
                "TargetSpec::NonlegendaryCreature",
                "spell_effect_destroy_nonlegendary_creature".to_string(),
                "no_effect".to_string(),
            ),
            Special::ReturnCreatureOrLandFromGraveyardAndGainLife { amount } => (
                "TargetSpec::CreatureOrLandCardInGraveyard",
                format!(
                    "spell_effect_return_creature_or_land_from_graveyard_and_gain_life_{amount}"
                ),
                "no_effect".to_string(),
            ),
            Special::DamageEachCreatureWithoutSubtype {
                amount,
                excluded_subtype,
            } => (
                "TargetSpec::None",
                format!(
                    "spell_effect_damage_each_creature_without_{}_{}",
                    excluded_subtype.to_ascii_lowercase(),
                    amount
                ),
                "no_effect".to_string(),
            ),
            Special::DrawThenCreateToken { .. } | Special::GainPaidCostManaValueThenDraw { .. } => {
                (
                    "TargetSpec::None",
                    format!(
                        "spell_effect_{}",
                        c.name.to_ascii_lowercase().replace([' ', '\''], "_")
                    ),
                    "no_effect".to_string(),
                )
            }
            Special::MillThenDraw { player, mill, draw } => {
                let (target_spec, player_suffix) = match player {
                    MillPlayer::Controller => ("TargetSpec::None", "controller"),
                    MillPlayer::Target0 => ("TargetSpec::AnyPlayer", "target_0"),
                };
                (
                    target_spec,
                    format!("spell_effect_mill_then_draw_{player_suffix}_{mill}_{draw}"),
                    "no_effect".to_string(),
                )
            }
            Special::ReturnOwnGraveyardCreatureToBattlefield => (
                "TargetSpec::CreatureCardInOwnGraveyard",
                "spell_effect_return_own_graveyard_creature_to_battlefield".to_string(),
                "no_effect".to_string(),
            ),
            Special::SearchForestToHand => (
                "TargetSpec::None",
                "spell_effect_search_forest_to_hand".to_string(),
                "no_effect".to_string(),
            ),
            Special::AddPlusOnePlusOneAndLifelinkCounters => (
                "TargetSpec::Creature",
                "spell_effect_add_plus_one_plus_one_and_lifelink_counters".to_string(),
                "no_effect".to_string(),
            ),
            Special::BindTheMonster => (
                "TargetSpec::Creature",
                "spell_effect_bind_the_monster".to_string(),
                "no_effect".to_string(),
            ),
            Special::Snap => (
                "TargetSpec::Creature",
                "spell_effect_snap".to_string(),
                "no_effect".to_string(),
            ),
            Special::DamageCannotBePreventedThisTurn => (
                "TargetSpec::None",
                "spell_effect_damage_cannot_be_prevented_this_turn".to_string(),
                "no_effect".to_string(),
            ),
            Special::PrismaticStrands => (
                "TargetSpec::None",
                "spell_effect_prismatic_strands".to_string(),
                "no_effect".to_string(),
            ),
            Special::CleansingWildfire => (
                "TargetSpec::Land",
                "spell_effect_cleansing_wildfire".to_string(),
                "no_effect".to_string(),
            ),
            Special::Duress => (
                "TargetSpec::TargetOpponent",
                "spell_effect_duress".to_string(),
                "no_effect".to_string(),
            ),
            Special::ToxinAnalysis => (
                "TargetSpec::Creature",
                "spell_effect_toxin_analysis".to_string(),
                "no_effect".to_string(),
            ),
            Special::WeatherTheStorm => (
                "TargetSpec::None",
                "spell_effect_weather_the_storm".to_string(),
                "no_effect".to_string(),
            ),
            Special::MonstrousEmergence => (
                "TargetSpec::Creature",
                "spell_effect_monstrous_emergence".to_string(),
                "no_effect".to_string(),
            ),
            Special::NyxbornHydra => (
                "TargetSpec::None",
                "spell_effect_nyxborn_hydra".to_string(),
                "no_effect".to_string(),
            ),
        };

        let executable = c.engine_capability != EngineCapabilityJson::NoEffect;
        if executable && matches!(special, Special::None) && is_ordinary_permanent(c) {
            spell_effect_src = "spell_effect_ordinary_permanent".to_string();
        }

        // Keep the historical function-pointer program for single-color
        // sources. Multi-color sources expose one action per printed mana
        // ability through `CardDef::mana_ability_choices` below.
        let mana_ability_colors = if executable && has_activated_mana_ability(c) {
            primary_mana_ability_colors(c)
        } else {
            Vec::new()
        };
        if mana_ability_colors.len() == 1 {
            let color = mana_ability_colors[0];
            let suffix = color.to_ascii_lowercase();
            color_variant(color);
            mana_ability_src = format!("mana_ability_add_{suffix}");
        }

        let has_spell_program = spell_effect_src != "no_effect";
        let has_mana_program = !mana_ability_colors.is_empty();
        if executable && !c.is_token {
            if c.is_land && !has_mana_program {
                panic!(
                    "cards_v1.json: executable land {:?} has no generated mana program",
                    c.name
                );
            }
            if !c.is_land && !has_spell_program {
                panic!(
                    "cards_v1.json: executable nonland {:?} has no generated spell program",
                    c.name
                );
            }
        }
        if !executable && (has_spell_program || has_mana_program) {
            panic!(
                "cards_v1.json: no-effect card {:?} unexpectedly received an executable program",
                c.name
            );
        }

        writeln!(out, "    CardDef {{").unwrap();
        writeln!(out, "        name: {:?},", c.name).unwrap();
        writeln!(
            out,
            "        capability: {},",
            capability_src(c.engine_capability)
        )
        .unwrap();
        writeln!(
            out,
            "        cost: Cost {{ pips: &[{pips_src}], generic: {generic}, x_count: {x_count} }},"
        )
        .unwrap();
        writeln!(
            out,
            "        generic_cost_reduction: {},",
            generic_cost_reduction_for(&c.name)
        )
        .unwrap();
        writeln!(out, "        ward_cost: {},", ward_cost_for(&c.name)).unwrap();
        writeln!(out, "        equipment: {},", equipment_for(&c.name)).unwrap();
        writeln!(out, "        types: &[{types_src}],").unwrap();
        writeln!(out, "        subtypes: &[{subtypes_src}],").unwrap();
        writeln!(out, "        supertypes: &[{supertypes_src}],").unwrap();
        writeln!(out, "        power: {power_src},").unwrap();
        writeln!(out, "        toughness: {toughness_src},").unwrap();
        writeln!(out, "        is_land: {},", c.is_land).unwrap();
        writeln!(out, "        produces_mana: &[{produces_src}],").unwrap();
        writeln!(out, "        colors: &[{colors_src}],").unwrap();
        writeln!(out, "        target_spec: {target_spec_src},").unwrap();
        writeln!(out, "        keywords: {},", keywords_for(c)).unwrap();
        writeln!(out, "        spell_effect: {spell_effect_src},").unwrap();
        writeln!(out, "        mana_ability: {mana_ability_src},").unwrap();
        writeln!(out, "        alt_cost: {},", alt_cost_for(&c.name)).unwrap();
        writeln!(out, "        kicker_cost: {},", kicker_cost_for(&c.name)).unwrap();
        writeln!(
            out,
            "        additional_cost: {},",
            additional_cost_for(&c.name)
        )
        .unwrap();
        writeln!(out, "        flashback: {},", flashback_for(&c.name)).unwrap();
        writeln!(
            out,
            "        activated_abilities: {},",
            activated_abilities_for(&c.name)
        )
        .unwrap();
        writeln!(out, "        plot_cost: {},", plot_cost_for(&c.name)).unwrap();
        writeln!(out, "        madness_cost: {},", madness_cost_for(&c.name)).unwrap();
        writeln!(out, "        mode2: {},", mode2_for(&c.name)).unwrap();
        writeln!(out, "        mode3: {},", mode3_for(&c.name)).unwrap();
        writeln!(out, "        is_token: {},", c.is_token).unwrap();
        writeln!(out, "        escape: {},", escape_for(&c.name)).unwrap();
        let mana_ability_choices_src = mana_ability_colors
            .iter()
            .map(|color| format!("ManaColor::{}", color_variant(color)))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            out,
            "        mana_ability_choices: &[{mana_ability_choices_src}],"
        )
        .unwrap();
        writeln!(
            out,
            "        enters_battlefield_tapped: {},",
            executable && enters_battlefield_tapped(c)
        )
        .unwrap();
        writeln!(
            out,
            "        mana_ability_def: {},",
            if executable {
                mana_ability_def_for(&c.name)
            } else {
                "None"
            }
        )
        .unwrap();
        writeln!(
            out,
            "        minimum_blockers: {},",
            minimum_blockers_for(&c.name)
        )
        .unwrap();
        writeln!(out, "        omen: {},", omen_for(&c.name)).unwrap();
        writeln!(out, "        mana_value: {},", c.mana_value).unwrap();
        writeln!(
            out,
            "        mana_ability_includes_chosen_color: {},",
            executable && mana_ability_includes_chosen_color(&c.name)
        )
        .unwrap();
        writeln!(
            out,
            "        as_enters_choose_color_other_than: {},",
            if executable {
                as_enters_choose_color_other_than(&c.name)
            } else {
                "None"
            }
        )
        .unwrap();
        writeln!(
            out,
            "        additional_mana_abilities: {},",
            if executable {
                additional_mana_abilities_for(&c.name)
            } else {
                "&[]"
            }
        )
        .unwrap();
        writeln!(out, "        object_name: {:?},", object_name_for(&c.name)).unwrap();
        writeln!(
            out,
            "        enters_battlefield_tapped_unless: {},",
            if executable {
                enters_battlefield_tapped_unless_for(&c.name)
            } else {
                "None"
            }
        )
        .unwrap();
        writeln!(out, "        attachment: {},", attachment_for(&c.name)).unwrap();
        writeln!(
            out,
            "        transform_face: {},",
            if executable {
                transform_face_for(&c.name)
            } else {
                "None"
            }
        )
        .unwrap();
        writeln!(
            out,
            "        saga: {},",
            if executable {
                saga_for(&c.name)
            } else {
                "None"
            }
        )
        .unwrap();
        writeln!(
            out,
            "        optional_additional_cost: {},",
            if executable {
                optional_additional_cost_for(&c.name)
            } else {
                "None"
            }
        )
        .unwrap();
        writeln!(
            out,
            "        changeling: {},",
            executable && changeling_for(&c.name)
        )
        .unwrap();
        writeln!(
            out,
            "        bestow: {},",
            if executable {
                bestow_for(&c.name)
            } else {
                "None".to_string()
            }
        )
        .unwrap();
        writeln!(out, "    }},").unwrap();
    }
    writeln!(out, "];").unwrap();
    writeln!(out).unwrap();

    // ---- name -> id --------------------------------------------------
    writeln!(out, "pub fn card_id_by_name(name: &str) -> Option<u16> {{").unwrap();
    writeln!(out, "    match name {{").unwrap();
    for (i, c) in cards.iter().enumerate() {
        writeln!(out, "        {:?} => Some({i}),", c.name).unwrap();
    }
    writeln!(out, "        _ => None,").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    // ---- content + executable-recipe hash ------------------------------
    // v30 hashes every generated CardDef selector plus semantic tokens from
    // the same `Special` and structured activated-ability recipes that emit
    // executable definitions. Lorien's Draw3/search and Deep Analysis's
    // target-player draw/ordered flashback and Sleep's ordered Escape cost
    // remain bound alongside each Blast's checked color and
    // targeting-versus-resolution filter timing. Wildfire utility effects,
    // typed battlefield searches, Storm, and Clue remain bound too.
    // Metadata-only registry fields (timestamps, java_file paths, complexity
    // tags) remain intentionally outside the contract.
    let mut canon = String::from("kernel_carddb/v32\n");
    for c in cards {
        canon.push_str(&c.name);
        canon.push('|');
        canon.push_str(&c.mana_cost);
        canon.push('|');
        canon.push_str(&c.mana_value.to_string());
        canon.push('|');
        canon.push_str(&c.types.join(","));
        canon.push('|');
        canon.push_str(&c.subtypes.join(","));
        canon.push('|');
        canon.push_str(&c.supertypes.join(","));
        canon.push('|');
        canon.push_str(&c.power.map(|p| p.to_string()).unwrap_or_default());
        canon.push('|');
        canon.push_str(&c.toughness.map(|t| t.to_string()).unwrap_or_default());
        canon.push('|');
        canon.push_str(if c.is_land { "L" } else { "-" });
        canon.push('|');
        canon.push_str(&c.produces_mana.join(","));
        canon.push('|');
        canon.push_str(&c.colors.join(","));
        canon.push('|');
        canon.push_str(if c.is_token { "T" } else { "-" });
        canon.push('|');
        canon.push_str(match c.engine_capability {
            EngineCapabilityJson::NoEffect => "no_effect",
            EngineCapabilityJson::Partial => "partial",
            EngineCapabilityJson::Full => "full",
        });
        canon.push('|');
        // Reuse the exact source fragment that generates CardDef so a change
        // to the generated reducer definition necessarily changes the frozen
        // database identity. Evaluator semantics remain source/version gated.
        // `None` is included too, preserving positional separation and making
        // addition/removal equally visible.
        canon.push_str(generic_cost_reduction_for(&c.name));
        canon.push('|');
        canon.push_str(ward_cost_for(&c.name));
        canon.push('|');
        canon.push_str("equipment=");
        canon.push_str(equipment_for(&c.name));
        canon.push('|');
        // Reuse the exact generated source fragments plus the stable spell
        // recipe token. This covers every field selected by the generator for
        // `CardDef`; runtime primitive implementation changes remain pinned
        // separately by the source revision.
        canon.push_str(&keywords_for(c));
        canon.push('|');
        canon.push_str("mana_ability_choices=");
        if c.engine_capability != EngineCapabilityJson::NoEffect && has_activated_mana_ability(c) {
            canon.push_str(&primary_mana_ability_colors(c).join(","));
        }
        canon.push('|');
        canon.push_str("enters_battlefield_tapped=");
        canon.push_str(
            if c.engine_capability != EngineCapabilityJson::NoEffect && enters_battlefield_tapped(c)
            {
                "true"
            } else {
                "false"
            },
        );
        canon.push('|');
        canon.push_str("mana_ability_def=");
        canon.push_str(if c.engine_capability != EngineCapabilityJson::NoEffect {
            mana_ability_def_for(&c.name)
        } else {
            "None"
        });
        canon.push('|');
        canon.push_str("mana_ability_includes_chosen_color=");
        canon.push_str(
            if c.engine_capability != EngineCapabilityJson::NoEffect
                && mana_ability_includes_chosen_color(&c.name)
            {
                "true"
            } else {
                "false"
            },
        );
        canon.push('|');
        canon.push_str("as_enters_choose_color_other_than=");
        canon.push_str(if c.engine_capability != EngineCapabilityJson::NoEffect {
            as_enters_choose_color_other_than(&c.name)
        } else {
            "None"
        });
        canon.push('|');
        canon.push_str("additional_mana_abilities=");
        canon.push_str(if c.engine_capability != EngineCapabilityJson::NoEffect {
            additional_mana_abilities_for(&c.name)
        } else {
            "&[]"
        });
        canon.push('|');
        canon.push_str("object_name=");
        canon.push_str(object_name_for(&c.name));
        canon.push('|');
        canon.push_str("enters_battlefield_tapped_unless=");
        canon.push_str(if c.engine_capability != EngineCapabilityJson::NoEffect {
            enters_battlefield_tapped_unless_for(&c.name)
        } else {
            "None"
        });
        canon.push('|');
        canon.push_str("attachment=");
        canon.push_str(attachment_for(&c.name));
        canon.push('|');
        canon.push_str("transform_face=");
        canon.push_str(if c.engine_capability != EngineCapabilityJson::NoEffect {
            transform_face_for(&c.name)
        } else {
            "None"
        });
        canon.push('|');
        canon.push_str("saga=");
        canon.push_str(if c.engine_capability != EngineCapabilityJson::NoEffect {
            saga_for(&c.name)
        } else {
            "None"
        });
        canon.push('|');
        canon.push_str("optional_additional_cost=");
        canon.push_str(if c.engine_capability != EngineCapabilityJson::NoEffect {
            optional_additional_cost_for(&c.name)
        } else {
            "None"
        });
        canon.push('|');
        canon.push_str("changeling=");
        canon.push_str(
            if c.engine_capability != EngineCapabilityJson::NoEffect && changeling_for(&c.name) {
                "true"
            } else {
                "false"
            },
        );
        canon.push('|');
        canon.push_str("bestow=");
        let bestow = if c.engine_capability != EngineCapabilityJson::NoEffect {
            bestow_for(&c.name)
        } else {
            "None".to_string()
        };
        canon.push_str(&bestow);
        canon.push('|');
        canon.push_str("trigger=");
        canon.push_str(if c.engine_capability != EngineCapabilityJson::NoEffect {
            trigger_recipe_for(&c.name)
        } else {
            "none"
        });
        canon.push('|');
        canon.push_str(alt_cost_for(&c.name));
        canon.push('|');
        canon.push_str(&kicker_cost_for(&c.name));
        canon.push('|');
        canon.push_str(additional_cost_for(&c.name));
        canon.push('|');
        canon.push_str(&flashback_for(&c.name));
        canon.push('|');
        canon.push_str(&activated_abilities_for(&c.name));
        canon.push('|');
        canon.push_str(&plot_cost_for(&c.name));
        canon.push('|');
        canon.push_str(&madness_cost_for(&c.name));
        canon.push('|');
        canon.push_str(&mode2_for(&c.name));
        canon.push('|');
        canon.push_str(&mode3_for(&c.name));
        canon.push('|');
        canon.push_str("minimum_blockers=");
        canon.push_str(&minimum_blockers_for(&c.name).to_string());
        canon.push('|');
        canon.push_str("omen=");
        canon.push_str(&omen_for(&c.name));
        canon.push('|');
        canon.push_str("omen_effect=");
        if let Some(effect) = omen_effect_recipe_for(&c.name) {
            canon.push_str(&ability_effect_token(effect));
        } else {
            canon.push_str("none");
        }
        canon.push('|');
        canon.push_str(&effect_recipe_for(c));
        canon.push('|');
        canon.push_str(&c.decks.join(","));
        canon.push('|');
        canon.push_str("special=");
        canon.push_str(&special_for(&c.name).canonical_token());
        canon.push('|');
        canon.push_str("activated=");
        canon.push_str(&activated_abilities_token(&c.name));
        canon.push('|');
        canon.push_str("escape=");
        canon.push_str(&escape_for(&c.name));
        canon.push('\n');
    }
    let hash = fnv1a64(canon.as_bytes());
    writeln!(out, "pub const KERNEL_CARDDB_HASH: u64 = 0x{hash:016x};").unwrap();

    out
}

/// Minimal ordered-unique-values collection so this file doesn't need a
/// `BTreeSet` import just for one dedup+sort of a handful of i32s.
struct BTreeSetLike(Vec<i32>);
impl FromIterator<i32> for BTreeSetLike {
    fn from_iter<T: IntoIterator<Item = i32>>(iter: T) -> Self {
        let mut v: Vec<i32> = iter.into_iter().collect();
        v.sort_unstable();
        v.dedup();
        BTreeSetLike(v)
    }
}
impl BTreeSetLike {
    fn values(&self) -> &[i32] {
        &self.0
    }
}

fn card_type_variant(t: &str) -> &'static str {
    match t {
        "Land" => "Land",
        "Creature" => "Creature",
        "Instant" => "Instant",
        "Sorcery" => "Sorcery",
        "Artifact" => "Artifact",
        "Enchantment" => "Enchantment",
        other => panic!("cards_v1.json: unknown card type {other:?}"),
    }
}

fn supertype_variant(t: &str) -> &'static str {
    match t {
        "Basic" => "Basic",
        "Snow" => "Snow",
        "Legendary" => "Legendary",
        other => panic!("cards_v1.json: unknown supertype {other:?}"),
    }
}

/// Maps a `cards_v1.json` subtype string to a `card_def::Subtype` variant --
/// a fully closed set (one variant per distinct string across the whole
/// 136-definition pool this increment's data covers), so this panics on an
/// unrecognized value same as `card_type_variant`/`supertype_variant`/
/// `color_variant` -- see `Subtype`'s own doc for why it's closed rather
/// than named-variants-plus-string-fallback (a `&'static str` payload can't
/// derive `Deserialize`, and `Subtype` is embedded in `effect::EffectOp`,
/// which needs to).
fn subtype_variant(t: &str) -> &'static str {
    match t {
        "Ape" => "Subtype::Ape",
        "Aura" => "Subtype::Aura",
        "BIRD" => "Subtype::BirdAllCaps",
        "Bird" => "Subtype::Bird",
        "Blood" => "Subtype::Blood",
        "Cat" => "Subtype::Cat",
        "Detective" => "Subtype::Detective",
        "Dragon" => "Subtype::Dragon",
        "Drone" => "Subtype::Drone",
        "Druid" => "Subtype::Druid",
        "Eldrazi" => "Subtype::Eldrazi",
        "Elf" => "Subtype::Elf",
        "Equipment" => "Subtype::Equipment",
        "FAERIE" => "Subtype::FaerieAllCaps",
        "Faerie" => "Subtype::Faerie",
        "Food" => "Subtype::Food",
        "Forest" => "Subtype::Forest",
        "Gate" => "Subtype::Gate",
        "Goblin" => "Subtype::Goblin",
        "HUMAN" => "Subtype::HumanAllCaps",
        "Hero" => "Subtype::Hero",
        "Human" => "Subtype::Human",
        "Hydra" => "Subtype::Hydra",
        "Island" => "Subtype::Island",
        "Knight" => "Subtype::Knight",
        "MONK" => "Subtype::Monk",
        "MOONFOLK" => "Subtype::Moonfolk",
        "Monkey" => "Subtype::Monkey",
        "Mountain" => "Subtype::Mountain",
        "Myr" => "Subtype::Myr",
        "NINJA" => "Subtype::NinjaAllCaps",
        "Ninja" => "Subtype::Ninja",
        "Ouphe" => "Subtype::Ouphe",
        "Pirate" => "Subtype::Pirate",
        "Plains" => "Subtype::Plains",
        "ROGUE" => "Subtype::RogueAllCaps",
        "Ranger" => "Subtype::Ranger",
        "Rat" => "Subtype::Rat",
        "Rogue" => "Subtype::Rogue",
        "SERPENT" => "Subtype::Serpent",
        "Saga" => "Subtype::Saga",
        "Samurai" => "Subtype::Samurai",
        "Shaman" => "Subtype::Shaman",
        "Shapeshifter" => "Subtype::Shapeshifter",
        "Soldier" => "Subtype::Soldier",
        "Spider" => "Subtype::Spider",
        "Spirit" => "Subtype::Spirit",
        "Swamp" => "Subtype::Swamp",
        "Toy" => "Subtype::Toy",
        "Treefolk" => "Subtype::Treefolk",
        "Vampire" => "Subtype::Vampire",
        "WIZARD" => "Subtype::WizardAllCaps",
        "Warrior" => "Subtype::Warrior",
        "Wizard" => "Subtype::Wizard",
        "Zombie" => "Subtype::Zombie",
        "Illusion" => "Subtype::Illusion",
        "Dryad" => "Subtype::Dryad",
        "Plant" => "Subtype::Plant",
        "Wall" => "Subtype::Wall",
        "Troll" => "Subtype::Troll",
        "Elemental" => "Subtype::Elemental",
        "Map" => "Subtype::Map",
        "Treasure" => "Subtype::Treasure",
        "Giant" => "Subtype::Giant",
        "Spawn" => "Subtype::Spawn",
        "Phyrexian" => "Subtype::Phyrexian",
        "Horror" => "Subtype::Horror",
        "Nightmare" => "Subtype::Nightmare",
        "Clue" => "Subtype::Clue",
        "Skeleton" => "Subtype::Skeleton",
        other => panic!("cards_v1.json: unknown subtype {other:?}"),
    }
}

fn color_variant(c: &str) -> &'static str {
    match c {
        "W" => "W",
        "U" => "U",
        "B" => "B",
        "R" => "R",
        "G" => "G",
        "C" => "C",
        other => panic!("cards_v1.json: unknown mana color {other:?}"),
    }
}

/// Parses a mana cost string like `"{1}{R}{R}"` into (pip expressions,
/// generic amount, X count). Pip expressions are emitted as literal Rust
/// source (e.g. `"Pip::Colored(ManaColor::R)"`) ready to splice into a
/// `&[...]` slice literal.
fn parse_cost(mana_cost: &str) -> (Vec<String>, u8, u8) {
    let mut pips = Vec::new();
    let mut generic: u32 = 0;
    let mut x_count: u8 = 0;

    let mut chars = mana_cost.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '{' {
            continue;
        }
        let mut tok = String::new();
        for c2 in chars.by_ref() {
            if c2 == '}' {
                break;
            }
            tok.push(c2);
        }

        if let Ok(n) = tok.parse::<u32>() {
            generic += n;
            continue;
        }
        if tok.eq_ignore_ascii_case("X") {
            x_count += 1;
            continue;
        }
        if let Some((a, b)) = tok.split_once('/') {
            if b.eq_ignore_ascii_case("P") {
                let color = color_variant(a);
                pips.push(format!("Pip::Phyrexian(ManaColor::{color})"));
            } else if a.chars().all(|ch| ch.is_ascii_digit()) {
                // Twobrid ({2/R}): not in the current pool. Approximate as
                // a colored pip; a future increment that needs twobrid
                // costs should give this its own Pip variant instead.
                let color = color_variant(b);
                pips.push(format!("Pip::Colored(ManaColor::{color})"));
            } else {
                let ca = color_variant(a);
                let cb = color_variant(b);
                pips.push(format!("Pip::Hybrid(ManaColor::{ca}, ManaColor::{cb})"));
            }
            continue;
        }
        let color = color_variant(tok.as_str());
        pips.push(format!("Pip::Colored(ManaColor::{color})"));
    }

    if generic > u8::MAX as u32 {
        panic!("cards_v1.json: generic cost {generic} overflows u8 in {mana_cost:?}");
    }
    (pips, generic as u8, x_count)
}

fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
