#![cfg_attr(not(feature = "std"), no_std)]

//! # Multi-Architecture JIT Assembler
//!
//! A multi-architecture JIT assembler library for runtime code generation.
//!
//! ## Features
//!
//! - **Multi-architecture support**: RISC-V, AArch64, x86-64 (planned)
//! - **Host-independent**: Runs on any host architecture to generate target code
//! - **No-std compatible**: Works in both `std` and `no_std` environments
//! - **Macro-based DSL**: Convenient syntax for writing assembly
//! - **JIT execution**: Direct execution of assembled code as functions (std-only)
//!
//! ## Supported Architectures
//!
//! - **RISC-V 64-bit** (`riscv` feature, enabled by default)
//! - **AArch64** (`aarch64` feature, enabled by default) - Basic arithmetic and logical operations
//! - **x86-64** (`x86_64` feature) - Coming soon
//!
//! ## Usage
//!
//! ```rust
//! # #[cfg(feature = "riscv")]
//! # {
//! use jit_assembler::riscv64::{reg, csr, Riscv64InstructionBuilder};
//! use jit_assembler::common::InstructionBuilder;
//!
//! // Method chaining style (recommended)
//! let mut builder = Riscv64InstructionBuilder::new();
//! let instructions = builder
//!     .csrrw(reg::RA, csr::MSTATUS, reg::SP)
//!     .addi(reg::A0, reg::ZERO, 100)
//!     .add(reg::A1, reg::A0, reg::SP)
//!     .ret()
//!     .instructions();
//!
//! // Macro style (concise and assembly-like)
//! let instructions3 = jit_assembler::riscv64_asm! {
//!     csrrw(reg::RA, csr::MSTATUS, reg::SP);
//!     addi(reg::A0, reg::ZERO, 100);
//!     add(reg::A1, reg::A0, reg::SP);
//!     ret();
//! };
//!
//! // Traditional style  
//! let mut builder2 = Riscv64InstructionBuilder::new();
//! builder2.csrrw(reg::RA, csr::MSTATUS, reg::SP);
//! builder2.addi(reg::A0, reg::ZERO, 100);
//! builder2.ret();
//! let instructions2 = builder2.instructions();
//! // InstructionCollection provides convenient methods
//! let bytes = instructions.to_bytes();     // Convert all to bytes
//! let size = instructions.total_size();    // Get total size
//! let count = instructions.len();          // Get instruction count
//!
//! // Iterate over instructions
//! for instr in instructions {
//!     let bytes = instr.bytes();
//!     // Write to executable memory...
//! }
//! # }
//! ```
//!
//! ## AArch64 Usage
//!
//! ```rust
//! # #[cfg(feature = "aarch64")]
//! # {
//! use jit_assembler::aarch64::{reg, Aarch64InstructionBuilder};
//! use jit_assembler::common::InstructionBuilder;
//!
//! // Create an AArch64 function that adds two numbers
//! let mut builder = Aarch64InstructionBuilder::new();
//! let instructions = builder
//!     .add(reg::X0, reg::X0, reg::X1)  // Add first two arguments (X0 + X1 -> X0)
//!     .ret()                           // Return
//!     .instructions();
//!
//! // Macro style (concise and assembly-like)
//! let instructions3 = jit_assembler::aarch64_asm! {
//!     add(reg::X0, reg::X0, reg::X1);  // Add first two arguments
//!     mov_imm(reg::X1, 42);            // Load immediate 42 into X1
//!     mul(reg::X0, reg::X0, reg::X1);  // Multiply X0 by 42
//!     ret();                           // Return
//! };
//!
//! // More complex AArch64 example with immediate values
//! let mut builder2 = Aarch64InstructionBuilder::new();
//! let instructions2 = builder2
//!     .mov_imm(reg::X1, 42)            // Load immediate 42 into X1
//!     .mul(reg::X0, reg::X0, reg::X1)  // Multiply X0 by 42
//!     .addi(reg::X0, reg::X0, 100)     // Add 100 to result
//!     .ret()                           // Return
//!     .instructions();
//! # }
//! ```
//!
//! ## JIT Execution (std-only)
//!
//! ```rust,no_run
//! # #[cfg(feature = "riscv")]
//! # {
//! use jit_assembler::riscv64::{reg, Riscv64InstructionBuilder};
//! use jit_assembler::common::InstructionBuilder;
//! 
//! // Create a JIT function that adds two numbers
//! let add_func = unsafe {
//!     Riscv64InstructionBuilder::new()
//!         .add(reg::A0, reg::A0, reg::A1) // Add first two arguments
//!         .ret()                          // Return result
//!         .function::<fn(u64, u64) -> u64>()
//! }.expect("Failed to create JIT function");
//!
//! // Call the JIT function naturally!
//! let result = add_func.call(10, 20);
//! assert_eq!(result, 30);
//! # }
//! ```

#[cfg(not(feature = "std"))]
extern crate alloc;

// Common types and traits shared across architectures
pub mod common;

// Re-export JIT functionality when std is available
#[cfg(feature = "std")]
pub use common::jit::{CallableJitFunction, JitError};

// Architecture-specific modules
#[cfg(feature = "riscv")]
pub mod riscv64;

#[cfg(feature = "x86_64")]
pub mod x86_64;

#[cfg(feature = "aarch64")]
pub mod aarch64;

// Re-export for convenience (default to RISC-V if available)
#[cfg(feature = "riscv")]
pub use riscv64 as default_arch;

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