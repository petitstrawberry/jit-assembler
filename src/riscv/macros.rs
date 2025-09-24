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
/// use jit_assembler::riscv::{reg, csr};
///
/// let instructions = jit_assembler::jit_asm! {
///     csrrw(reg::X1, csr::MSTATUS, reg::X2);
///     addi(reg::X3, reg::X1, 100);
///     add(reg::X4, reg::X1, reg::X3);
/// };
/// ```
#[macro_export]
macro_rules! jit_asm {
    ($($method:ident($($args:expr),*);)*) => {{
        $crate::jit_asm_generic!($crate::riscv::InstructionBuilder, $($method($($args),*);)*)
    }};
}

// The complex jit_asm_chain! macro has been removed.
// Now we use direct method calls on InstructionBuilder for better IDE support and maintainability.