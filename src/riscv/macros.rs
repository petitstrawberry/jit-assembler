/// JIT assembler macro interface
/// 
/// This module provides a macro-based interface for writing RISC-V assembly
/// in a convenient syntax that gets compiled to machine code instructions.

/// JIT assembler macro
/// 
/// Provides a fluent interface for generating RISC-V instructions.
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
    ($($instr:tt)*) => {{
        $crate::jit_asm_chain!($crate::riscv::InstructionBuilder::new(), $($instr)*)
            .instructions()
            .to_vec()
    }};
}

/// Internal implementation macro for processing individual instructions
#[macro_export]
macro_rules! jit_asm_chain {
    // Base case
    ($builder_expr:expr,) => {
        $builder_expr
    };

    // Special case: CSR read (alias)
    ($builder_expr:expr, csrr($rd:expr, $csr:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.csrrs($rd, $csr, $crate::riscv::reg::X0), $($rest)*)
    };

    // CSR instructions (rd, csr, rs1)
    ($builder_expr:expr, csrrw($rd:expr, $csr:expr, $rs1:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.csrrw($rd, $csr, $rs1), $($rest)*)
    };
    ($builder_expr:expr, csrrs($rd:expr, $csr:expr, $rs1:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.csrrs($rd, $csr, $rs1), $($rest)*)
    };
    ($builder_expr:expr, csrrc($rd:expr, $csr:expr, $rs1:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.csrrc($rd, $csr, $rs1), $($rest)*)
    };

    // CSR immediate instructions (rd, csr, uimm)
    ($builder_expr:expr, csrrwi($rd:expr, $csr:expr, $uimm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.csrrwi($rd, $csr, $uimm), $($rest)*)
    };
    ($builder_expr:expr, csrrsi($rd:expr, $csr:expr, $uimm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.csrrsi($rd, $csr, $uimm), $($rest)*)
    };
    ($builder_expr:expr, csrrci($rd:expr, $csr:expr, $uimm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.csrrci($rd, $csr, $uimm), $($rest)*)
    };

    // R-type arithmetic instructions (rd, rs1, rs2)
    ($builder_expr:expr, add($rd:expr, $rs1:expr, $rs2:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.add($rd, $rs1, $rs2), $($rest)*)
    };
    ($builder_expr:expr, sub($rd:expr, $rs1:expr, $rs2:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.sub($rd, $rs1, $rs2), $($rest)*)
    };
    ($builder_expr:expr, xor($rd:expr, $rs1:expr, $rs2:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.xor($rd, $rs1, $rs2), $($rest)*)
    };
    ($builder_expr:expr, or($rd:expr, $rs1:expr, $rs2:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.or($rd, $rs1, $rs2), $($rest)*)
    };
    ($builder_expr:expr, and($rd:expr, $rs1:expr, $rs2:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.and($rd, $rs1, $rs2), $($rest)*)
    };

    // I-type instructions (rd, rs1, imm)
    ($builder_expr:expr, addi($rd:expr, $rs1:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.addi($rd, $rs1, $imm), $($rest)*)
    };
    ($builder_expr:expr, jalr($rd:expr, $rs1:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.jalr($rd, $rs1, $imm), $($rest)*)
    };

    // J-type instructions (rd, imm)
    ($builder_expr:expr, jal($rd:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.jal($rd, $imm), $($rest)*)
    };

    // B-type branch instructions (rs1, rs2, imm)
    ($builder_expr:expr, beq($rs1:expr, $rs2:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.beq($rs1, $rs2, $imm), $($rest)*)
    };
    ($builder_expr:expr, bne($rs1:expr, $rs2:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.bne($rs1, $rs2, $imm), $($rest)*)
    };
    ($builder_expr:expr, blt($rs1:expr, $rs2:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.blt($rs1, $rs2, $imm), $($rest)*)
    };
    ($builder_expr:expr, bge($rs1:expr, $rs2:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.bge($rs1, $rs2, $imm), $($rest)*)
    };
    ($builder_expr:expr, bltu($rs1:expr, $rs2:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.bltu($rs1, $rs2, $imm), $($rest)*)
    };
    ($builder_expr:expr, bgeu($rs1:expr, $rs2:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.bgeu($rs1, $rs2, $imm), $($rest)*)
    };

    // Load instructions (rd, rs1, imm)
    ($builder_expr:expr, ld($rd:expr, $rs1:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.ld($rd, $rs1, $imm), $($rest)*)
    };
    ($builder_expr:expr, lw($rd:expr, $rs1:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.lw($rd, $rs1, $imm), $($rest)*)
    };
    ($builder_expr:expr, lh($rd:expr, $rs1:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.lh($rd, $rs1, $imm), $($rest)*)
    };
    ($builder_expr:expr, lb($rd:expr, $rs1:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.lb($rd, $rs1, $imm), $($rest)*)
    };

    // Store instructions (rs1, rs2, imm)
    ($builder_expr:expr, sd($rs1:expr, $rs2:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.sd($rs1, $rs2, $imm), $($rest)*)
    };
    ($builder_expr:expr, sw($rs1:expr, $rs2:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.sw($rs1, $rs2, $imm), $($rest)*)
    };
    ($builder_expr:expr, sh($rs1:expr, $rs2:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.sh($rs1, $rs2, $imm), $($rest)*)
    };
    ($builder_expr:expr, sb($rs1:expr, $rs2:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.sb($rs1, $rs2, $imm), $($rest)*)
    };

    // Shift instructions (rd, rs1, rs2)
    ($builder_expr:expr, sll($rd:expr, $rs1:expr, $rs2:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.sll($rd, $rs1, $rs2), $($rest)*)
    };
    ($builder_expr:expr, srl($rd:expr, $rs1:expr, $rs2:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.srl($rd, $rs1, $rs2), $($rest)*)
    };
    ($builder_expr:expr, sra($rd:expr, $rs1:expr, $rs2:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.sra($rd, $rs1, $rs2), $($rest)*)
    };

    // Shift immediate instructions (rd, rs1, shamt)
    ($builder_expr:expr, slli($rd:expr, $rs1:expr, $shamt:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.slli($rd, $rs1, $shamt), $($rest)*)
    };
    ($builder_expr:expr, srli($rd:expr, $rs1:expr, $shamt:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.srli($rd, $rs1, $shamt), $($rest)*)
    };
    ($builder_expr:expr, srai($rd:expr, $rs1:expr, $shamt:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.srai($rd, $rs1, $shamt), $($rest)*)
    };

    // Upper immediate instructions (rd, imm)
    ($builder_expr:expr, lui($rd:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.lui($rd, $imm), $($rest)*)
    };
    ($builder_expr:expr, auipc($rd:expr, $imm:expr); $($rest:tt)*) => {
        $crate::jit_asm_chain!($builder_expr.auipc($rd, $imm), $($rest)*)
    };
}