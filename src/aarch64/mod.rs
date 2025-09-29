//! AArch64 instruction set support for the JIT assembler.
//!
//! This module provides AArch64 specific instruction encoding and a builder-based
//! interface for generating AArch64 machine code at runtime.
//!
//! ## Features
//!
//! - **64-bit instructions**: All instructions are 32-bit fixed-width
//! - **Register support**: 31 general-purpose registers (X0-X30) plus SP/XZR
//! - **Basic arithmetic**: ADD, SUB, MUL, DIV operations
//! - **Logical operations**: AND, OR, XOR operations
//! - **Register tracking**: Optional register usage analysis
//! - **JIT execution**: Direct execution of assembled code as functions
//!
//! ## Register Conventions (AAPCS64)
//!
//! - **X0-X7**: Argument/result registers (caller-saved)
//! - **X8-X15**: Caller-saved temporary registers
//! - **X16-X17**: Intra-procedure-call registers (caller-saved)
//! - **X18**: Platform register (caller-saved on most platforms)
//! - **X19-X28**: Callee-saved registers
//! - **X29**: Frame pointer (FP)
//! - **X30**: Link register (LR)
//! - **X31**: Stack pointer (SP) or zero register (XZR/WZR)
//!
//! ## Examples
//!
//! ```rust,no_run
//! use jit_assembler::aarch64::{reg, Aarch64InstructionBuilder};
//! use jit_assembler::common::InstructionBuilder;
//!
//! let mut builder = Aarch64InstructionBuilder::new();
//! builder
//!     .add(reg::X0, reg::X0, reg::X1)  // X0 = X0 + X1
//!     .ret();                          // Return
//!
//! // On AArch64 hosts, you can JIT compile and execute:
//! // let func = unsafe { builder.function::<fn(u64, u64) -> u64>() }?;
//! // let result = func.call(10, 20); // Returns 30
//! ```

pub mod instruction;
pub mod builder;
pub mod macros;

#[cfg(test)]
mod tests;

// Re-export commonly used items
pub use instruction::{Register, Instruction, reg};
pub use builder::Aarch64InstructionBuilder;