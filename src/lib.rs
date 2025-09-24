#![cfg_attr(not(feature = "std"), no_std)]

//! # Multi-Architecture JIT Assembler
//!
//! A multi-architecture JIT assembler library for runtime code generation.
//!
//! ## Features
//!
//! - **Multi-architecture support**: RISC-V, x86-64, ARM64 (planned)
//! - **Host-independent**: Runs on any host architecture to generate target code
//! - **No-std compatible**: Works in both `std` and `no_std` environments
//! - **Macro-based DSL**: Convenient syntax for writing assembly
//!
//! ## Supported Architectures
//!
//! - **RISC-V 64-bit** (`riscv` feature, enabled by default)
//! - **x86-64** (`x86_64` feature) - Coming soon
//! - **ARM64** (`arm64` feature) - Coming soon
//!
//! ## Usage
//!
//! ```rust
//! use jit_assembler::riscv::{reg, csr, InstructionBuilder};
//!
//! // Method chaining style (recommended)
//! let mut builder = InstructionBuilder::new();
//! let instructions = builder
//!     .csrrw(reg::RA, csr::MSTATUS, reg::SP)
//!     .addi(reg::A0, reg::ZERO, 100)
//!     .add(reg::A1, reg::A0, reg::SP)
//!     .ret()
//!     .instructions();
//!
//! // Traditional style
//! let mut builder2 = InstructionBuilder::new();
//! builder2.csrrw(reg::RA, csr::MSTATUS, reg::SP);
//! builder2.addi(reg::A0, reg::ZERO, 100);
//! builder2.ret();
//! let instructions2 = builder2.instructions();
//!
//! // Convert to bytes for execution
//! for instr in instructions {
//!     let bytes = instr.bytes();
//!     // Write to executable memory...
//! }
//! ```

#[cfg(not(feature = "std"))]
extern crate alloc;

// Common types and traits shared across architectures
pub mod common;

// Architecture-specific modules
#[cfg(feature = "riscv")]
pub mod riscv;

#[cfg(feature = "x86_64")]
pub mod x86_64;

#[cfg(feature = "arm64")]
pub mod arm64;

// Re-export for convenience (default to RISC-V if available)
#[cfg(feature = "riscv")]
pub use riscv as default_arch;

/// Generic JIT assembler macro - Reference implementation
///
/// This is a reference implementation that can be used by each architecture.
/// Each architecture can provide their own specialized version by specifying
/// the appropriate InstructionBuilder type.
///
/// Usage pattern for architecture-specific implementations:
/// ```rust
/// #[macro_export]
/// macro_rules! arch_asm {
///     ($($method:ident($($args:expr),*);)*) => {{
///         $crate::jit_asm_generic!(YourArchInstructionBuilder, $($method($($args),*);)*)
///     }};
/// }
/// ```
#[macro_export]
macro_rules! jit_asm_generic {
    ($builder_type:ty, $($method:ident($($args:expr),*);)*) => {{
        let mut builder = <$builder_type>::new();
        $(
            builder.$method($($args),*);
        )*
        builder.instructions().to_vec()
    }};
}