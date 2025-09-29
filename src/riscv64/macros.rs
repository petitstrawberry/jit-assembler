/// JIT assembler macro interface
///
/// This module provides a macro-based interface for writing RISC-V assembly
/// in a convenient syntax that gets compiled to machine code instructions.

/// RISC-V JIT assembler macro
///
/// Provides a clean interface for generating RISC-V instructions.
/// Uses the generic jit_asm_generic! macro with RISC-V InstructionBuilder.
///
/// Usage:
/// ```rust
/// use jit_assembler::riscv64::{reg, csr};
/// use jit_assembler::common::InstructionBuilder;
///
/// let instructions = jit_assembler::riscv64_asm! {
///     csrrw(reg::X1, csr::MSTATUS, reg::X2);
///     addi(reg::X3, reg::X1, 100);
///     add(reg::X4, reg::X1, reg::X3);
/// };
/// ```
#[macro_export]
macro_rules! riscv64_asm {
    ($($method:ident($($args:expr),*);)*) => {{
        $crate::jit_asm_generic!($crate::riscv64::Riscv64InstructionBuilder, $($method($($args),*);)*)
    }};
}

// The complex jit_asm_chain! macro has been removed.
// Now we use direct method calls on InstructionBuilder for better IDE support and maintainability.