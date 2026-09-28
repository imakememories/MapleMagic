pub use mtg_kernel;

pub mod game;
pub mod ismcts;
pub mod features;
pub mod selfplay;

// Game-state clones allocate heavily; the system allocator on Windows
// serializes threads on it. mimalloc scales across threads.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
