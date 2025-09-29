/// JIT assembler macro interface
///
/// This module provides a macro-based interface for writing AArch64 assembly
/// in a convenient syntax that gets compiled to machine code instructions.

/// AArch64 JIT assembler macro
///
/// Provides a clean interface for generating AArch64 instructions.
/// Uses the generic jit_asm_generic! macro with AArch64 InstructionBuilder.
///
/// Usage:
/// ```rust
/// use jit_assembler::aarch64::{reg};
/// use jit_assembler::common::InstructionBuilder;
///
/// let instructions = jit_assembler::aarch64_asm! {
///     add(reg::X0, reg::X0, reg::X1);
///     mov_imm(reg::X1, 42);
///     ret();
/// };
/// ```
#[macro_export]
macro_rules! aarch64_asm {
    ($($method:ident($($args:expr),*);)*) => {{
        $crate::jit_asm_generic!($crate::aarch64::Aarch64InstructionBuilder, $($method($($args),*);)*)
    }};
}