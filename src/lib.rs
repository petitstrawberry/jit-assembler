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
//!     .csrrw(reg::X1, csr::MSTATUS, reg::X2)
//!     .addi(reg::X3, reg::X1, 100)
//!     .add(reg::X4, reg::X1, reg::X2)
//!     .instructions();
//!
//! // Traditional style
//! let mut builder2 = InstructionBuilder::new();
//! builder2.csrrw(reg::X1, csr::MSTATUS, reg::X2);
//! builder2.addi(reg::X3, reg::X1, 100);
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