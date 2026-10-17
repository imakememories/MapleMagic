pub use mtg_kernel;

pub mod board;
pub mod commander;
pub mod events;
pub mod game;
pub mod history;
pub mod play;
pub mod ismcts;
pub mod features;
pub mod selfplay;

// Game-state clones allocate heavily, and the Windows allocator doesn't scale across threads.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
