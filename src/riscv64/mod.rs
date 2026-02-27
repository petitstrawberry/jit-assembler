//! RISC-V instruction set support for the JIT assembler.
//!
//! This module provides RISC-V specific instruction encoding and a macro-based
//! DSL for generating RISC-V machine code at runtime.

pub mod builder;
pub mod instruction;
pub mod macros;

#[cfg(test)]
mod tests;

// Re-export commonly used items
pub use builder::Riscv64InstructionBuilder;
pub use instruction::{csr, reg, Csr, Instruction, Register};
