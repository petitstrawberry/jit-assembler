use crate::common::{Instruction as InstructionTrait, Register as RegisterTrait};
/// RISC-V instruction formats and encoding
use core::fmt;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// RISC-V register representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Register(pub u8);

impl Register {
    pub const fn new(reg: u8) -> Self {
        Self(reg)
    }

    pub fn value(self) -> u8 {
        self.0
    }
}

impl RegisterTrait for Register {
    fn id(&self) -> u32 {
        self.0 as u32
    }

    fn abi_class(&self) -> crate::common::AbiClass {
        use crate::common::AbiClass;

        match self.0 {
            // Caller-saved registers (do not need to be preserved across calls)
            1 => AbiClass::CallerSaved, // RA (return address) - caller-saved
            5..=7 | 28..=31 => AbiClass::CallerSaved, // T0-T2, T3-T6 (temporaries)
            10..=17 => AbiClass::CallerSaved, // A0-A7 (arguments/return values)

            // Callee-saved registers (must be preserved across calls)
            2 => AbiClass::CalleeSaved, // SP (stack pointer) - callee-saved
            8..=9 | 18..=27 => AbiClass::CalleeSaved, // S0-S1, S2-S11 (saved registers)

            // Special-purpose registers
            0 => AbiClass::Special, // X0 (zero register) - hardwired to zero
            3 => AbiClass::Special, // GP (global pointer) - special purpose
            4 => AbiClass::Special, // TP (thread pointer) - special purpose

            // Default to Special for any unhandled registers
            _ => AbiClass::Special,
        }
    }
}

/// RISC-V CSR representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Csr(pub u16);

impl Csr {
    pub const fn new(csr: u16) -> Self {
        Self(csr)
    }

    pub fn value(self) -> u16 {
        self.0
    }
}

/// RISC-V instruction representation (supports both 32-bit and 16-bit compressed instructions)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instruction {
    /// 32-bit standard instruction
    Standard(u32),
    /// 16-bit compressed instruction (C extension)
    Compressed(u16),
}

impl Instruction {
    /// Create a new 32-bit instruction
    pub fn new(value: u32) -> Self {
        Self::Standard(value)
    }

    /// Create a new 16-bit compressed instruction
    pub fn new_compressed(value: u16) -> Self {
        Self::Compressed(value)
    }

    /// Get the instruction value as u64 (for compatibility)
    pub fn value(self) -> u32 {
        match self {
            Self::Standard(val) => val,
            Self::Compressed(val) => val as u32,
        }
    }

    /// Get the instruction as bytes with proper length
    pub fn bytes(self) -> Vec<u8> {
        match self {
            Self::Standard(val) => val.to_le_bytes().to_vec(),
            Self::Compressed(val) => val.to_le_bytes().to_vec(),
        }
    }

    /// Get the size of this instruction in bytes
    pub fn size(&self) -> usize {
        match self {
            Self::Standard(_) => 4,
            Self::Compressed(_) => 2,
        }
    }

    /// Check if this is a compressed instruction
    pub fn is_compressed(&self) -> bool {
        matches!(self, Self::Compressed(_))
    }
}

impl InstructionTrait for Instruction {
    fn value(&self) -> u64 {
        match self {
            Self::Standard(val) => *val as u64,
            Self::Compressed(val) => *val as u64,
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            Self::Standard(val) => val.to_le_bytes().to_vec(),
            Self::Compressed(val) => val.to_le_bytes().to_vec(),
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::Standard(_) => 4,
            Self::Compressed(_) => 2,
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard(val) => write!(f, "0x{:08x}", val),
            Self::Compressed(val) => write!(f, "0x{:04x}c", val),
        }
    }
}

/// I-type instruction encoding
pub fn encode_i_type(opcode: u8, rd: Register, funct3: u8, rs1: Register, imm: i16) -> Instruction {
    let imm = imm as u32 & 0xfff;
    let instr = (imm << 20)
        | ((rs1.value() as u32) << 15)
        | ((funct3 as u32) << 12)
        | ((rd.value() as u32) << 7)
        | (opcode as u32);
    Instruction::Standard(instr)
}

/// CSR instruction encoding (I-type variant)
pub fn encode_csr_type(
    opcode: u8,
    rd: Register,
    funct3: u8,
    rs1: Register,
    csr: Csr,
) -> Instruction {
    let csr_val = csr.value() as u32;
    let instr = (csr_val << 20)
        | ((rs1.value() as u32) << 15)
        | ((funct3 as u32) << 12)
        | ((rd.value() as u32) << 7)
        | (opcode as u32);
    Instruction::Standard(instr)
}

/// CSR immediate instruction encoding
pub fn encode_csr_imm_type(
    opcode: u8,
    rd: Register,
    funct3: u8,
    uimm: u8,
    csr: Csr,
) -> Instruction {
    let csr_val = csr.value() as u32;
    let instr = (csr_val << 20)
        | ((uimm as u32) << 15)
        | ((funct3 as u32) << 12)
        | ((rd.value() as u32) << 7)
        | (opcode as u32);
    Instruction::Standard(instr)
}

/// Privileged instruction encoding (special SYSTEM instructions)
pub fn encode_privileged_type(opcode: u8, funct12: u16) -> Instruction {
    let instr = ((funct12 as u32) << 20) | (opcode as u32);
    Instruction::Standard(instr)
}

/// R-type instruction encoding
pub fn encode_r_type(
    opcode: u8,
    rd: Register,
    funct3: u8,
    rs1: Register,
    rs2: Register,
    funct7: u8,
) -> Instruction {
    let instr = ((funct7 as u32) << 25)
        | ((rs2.value() as u32) << 20)
        | ((rs1.value() as u32) << 15)
        | ((funct3 as u32) << 12)
        | ((rd.value() as u32) << 7)
        | (opcode as u32);
    Instruction::Standard(instr)
}

/// S-type instruction encoding (Store)
pub fn encode_s_type(
    opcode: u8,
    funct3: u8,
    rs1: Register,
    rs2: Register,
    imm: i16,
) -> Instruction {
    let imm = imm as u32 & 0xfff;
    let imm_11_5 = (imm >> 5) & 0x7f;
    let imm_4_0 = imm & 0x1f;
    let instr = (imm_11_5 << 25)
        | ((rs2.value() as u32) << 20)
        | ((rs1.value() as u32) << 15)
        | ((funct3 as u32) << 12)
        | (imm_4_0 << 7)
        | (opcode as u32);
    Instruction::Standard(instr)
}

/// B-type instruction encoding (Branch)
pub fn encode_b_type(
    opcode: u8,
    funct3: u8,
    rs1: Register,
    rs2: Register,
    imm: i16,
) -> Instruction {
    let imm = (imm as u32) & 0x1ffe; // 13-bit signed immediate
    let imm_12 = (imm >> 12) & 0x1;
    let imm_10_5 = (imm >> 5) & 0x3f;
    let imm_4_1 = (imm >> 1) & 0xf;
    let imm_11 = (imm >> 11) & 0x1;
    let instr = (imm_12 << 31)
        | (imm_10_5 << 25)
        | ((rs2.value() as u32) << 20)
        | ((rs1.value() as u32) << 15)
        | ((funct3 as u32) << 12)
        | (imm_4_1 << 8)
        | (imm_11 << 7)
        | (opcode as u32);
    Instruction::Standard(instr)
}

/// U-type instruction encoding (Upper immediate)
pub fn encode_u_type(opcode: u8, rd: Register, imm: u32) -> Instruction {
    let imm = (imm & 0xfffff) << 12; // 20-bit immediate shifted to upper bits
    let instr = imm | ((rd.value() as u32) << 7) | (opcode as u32);
    Instruction::Standard(instr)
}

/// J-type instruction encoding (Jump)
pub fn encode_j_type(opcode: u8, rd: Register, imm: i32) -> Instruction {
    let imm = (imm as u32) & 0x1fffff; // 21-bit signed immediate
    let imm_20 = (imm >> 20) & 0x1;
    let imm_10_1 = (imm >> 1) & 0x3ff;
    let imm_11 = (imm >> 11) & 0x1;
    let imm_19_12 = (imm >> 12) & 0xff;
    let instr = (imm_20 << 31)
        | (imm_10_1 << 21)
        | (imm_11 << 20)
        | (imm_19_12 << 12)
        | ((rd.value() as u32) << 7)
        | (opcode as u32);
    Instruction::Standard(instr)
}

/// Common RISC-V opcodes
pub mod opcodes {
    pub const SYSTEM: u8 = 0x73;
    pub const OP_IMM: u8 = 0x13;
    pub const OP: u8 = 0x33;
    pub const LOAD: u8 = 0x03;
    pub const STORE: u8 = 0x23;
    pub const BRANCH: u8 = 0x63;
    pub const JAL: u8 = 0x6f;
    pub const JALR: u8 = 0x67;
    pub const LUI: u8 = 0x37;
    pub const AUIPC: u8 = 0x17;
    pub const OP_IMM_32: u8 = 0x1b;
    pub const OP_32: u8 = 0x3b;
}

/// SYSTEM instruction function codes
pub mod system_funct3 {
    pub const CSRRW: u8 = 0x1;
    pub const CSRRS: u8 = 0x2;
    pub const CSRRC: u8 = 0x3;
    pub const CSRRWI: u8 = 0x5;
    pub const CSRRSI: u8 = 0x6;
    pub const CSRRCI: u8 = 0x7;
}

/// Privileged instruction function codes (funct12 field)
pub mod privileged_funct12 {
    pub const ECALL: u16 = 0x000; // Environment call
    pub const EBREAK: u16 = 0x001; // Environment break
    pub const SRET: u16 = 0x102; // Supervisor return
    pub const MRET: u16 = 0x302; // Machine return
    pub const WFI: u16 = 0x105; // Wait for interrupt
}

/// Branch instruction function codes
pub mod branch_funct3 {
    pub const BEQ: u8 = 0x0;
    pub const BNE: u8 = 0x1;
    pub const BLT: u8 = 0x4;
    pub const BGE: u8 = 0x5;
    pub const BLTU: u8 = 0x6;
    pub const BGEU: u8 = 0x7;
}

/// Load instruction function codes
pub mod load_funct3 {
    pub const LB: u8 = 0x0;
    pub const LH: u8 = 0x1;
    pub const LW: u8 = 0x2;
    pub const LD: u8 = 0x3;
    pub const LBU: u8 = 0x4;
    pub const LHU: u8 = 0x5;
    pub const LWU: u8 = 0x6;
}

/// Store instruction function codes
pub mod store_funct3 {
    pub const SB: u8 = 0x0;
    pub const SH: u8 = 0x1;
    pub const SW: u8 = 0x2;
    pub const SD: u8 = 0x3;
}

/// Arithmetic/Logic instruction function codes
pub mod alu_funct3 {
    pub const ADD_SUB: u8 = 0x0;
    pub const SLL: u8 = 0x1;
    pub const SLT: u8 = 0x2;
    pub const SLTU: u8 = 0x3;
    pub const XOR: u8 = 0x4;
    pub const SRL_SRA: u8 = 0x5;
    pub const OR: u8 = 0x6;
    pub const AND: u8 = 0x7;
}

/// M Extension (Multiply/Divide) instruction function codes
/// All M-extension instructions use funct7 = 0x01 and opcode = OP (0x33)
pub mod m_funct3 {
    pub const MUL: u8 = 0x0; // Multiply (low 64 bits)
    pub const MULH: u8 = 0x1; // Multiply high (signed × signed)
    pub const MULHSU: u8 = 0x2; // Multiply high (signed × unsigned)
    pub const MULHU: u8 = 0x3; // Multiply high (unsigned × unsigned)
    pub const DIV: u8 = 0x4; // Divide (signed)
    pub const DIVU: u8 = 0x5; // Divide (unsigned)
    pub const REM: u8 = 0x6; // Remainder (signed)
    pub const REMU: u8 = 0x7; // Remainder (unsigned)
}

/// M Extension funct7 code
pub mod m_funct7 {
    pub const M_EXT: u8 = 0x01;
}

/// Common CSR addresses
pub mod csr {
    use super::Csr;

    // Machine Information Registers
    pub const MVENDORID: Csr = Csr::new(0xf11);
    pub const MARCHID: Csr = Csr::new(0xf12);
    pub const MIMPID: Csr = Csr::new(0xf13);
    pub const MHARTID: Csr = Csr::new(0xf14);
    pub const MCONFIGPTR: Csr = Csr::new(0xf15);

    // Machine Trap Setup
    pub const MSTATUS: Csr = Csr::new(0x300);
    pub const MISA: Csr = Csr::new(0x301);
    pub const MEDELEG: Csr = Csr::new(0x302);
    pub const MIDELEG: Csr = Csr::new(0x303);
    pub const MIE: Csr = Csr::new(0x304);
    pub const MTVEC: Csr = Csr::new(0x305);
    pub const MCOUNTEREN: Csr = Csr::new(0x306);
    pub const MSTATUSH: Csr = Csr::new(0x310);

    // Machine Trap Handling
    pub const MSCRATCH: Csr = Csr::new(0x340);
    pub const MEPC: Csr = Csr::new(0x341);
    pub const MCAUSE: Csr = Csr::new(0x342);
    pub const MTVAL: Csr = Csr::new(0x343);
    pub const MIP: Csr = Csr::new(0x344);
    pub const MTINST: Csr = Csr::new(0x34a);
    pub const MTVAL2: Csr = Csr::new(0x34b);

    // Machine Configuration
    pub const MENVCFG: Csr = Csr::new(0x30a);
    pub const MENVCFGH: Csr = Csr::new(0x31a);
    pub const MSECCFG: Csr = Csr::new(0x747);
    pub const MSECCFGH: Csr = Csr::new(0x757);

    // Machine Memory Protection - Configuration
    pub const PMPCFG0: Csr = Csr::new(0x3a0);
    pub const PMPCFG1: Csr = Csr::new(0x3a1);
    pub const PMPCFG2: Csr = Csr::new(0x3a2);
    pub const PMPCFG3: Csr = Csr::new(0x3a3);
    pub const PMPCFG4: Csr = Csr::new(0x3a4);
    pub const PMPCFG5: Csr = Csr::new(0x3a5);
    pub const PMPCFG6: Csr = Csr::new(0x3a6);
    pub const PMPCFG7: Csr = Csr::new(0x3a7);
    pub const PMPCFG8: Csr = Csr::new(0x3a8);
    pub const PMPCFG9: Csr = Csr::new(0x3a9);
    pub const PMPCFG10: Csr = Csr::new(0x3aa);
    pub const PMPCFG11: Csr = Csr::new(0x3ab);
    pub const PMPCFG12: Csr = Csr::new(0x3ac);
    pub const PMPCFG13: Csr = Csr::new(0x3ad);
    pub const PMPCFG14: Csr = Csr::new(0x3ae);
    pub const PMPCFG15: Csr = Csr::new(0x3af);

    // Machine Memory Protection - Address
    pub const PMPADDR0: Csr = Csr::new(0x3b0);
    pub const PMPADDR1: Csr = Csr::new(0x3b1);
    pub const PMPADDR2: Csr = Csr::new(0x3b2);
    pub const PMPADDR3: Csr = Csr::new(0x3b3);
    pub const PMPADDR4: Csr = Csr::new(0x3b4);
    pub const PMPADDR5: Csr = Csr::new(0x3b5);
    pub const PMPADDR6: Csr = Csr::new(0x3b6);
    pub const PMPADDR7: Csr = Csr::new(0x3b7);
    pub const PMPADDR8: Csr = Csr::new(0x3b8);
    pub const PMPADDR9: Csr = Csr::new(0x3b9);
    pub const PMPADDR10: Csr = Csr::new(0x3ba);
    pub const PMPADDR11: Csr = Csr::new(0x3bb);
    pub const PMPADDR12: Csr = Csr::new(0x3bc);
    pub const PMPADDR13: Csr = Csr::new(0x3bd);
    pub const PMPADDR14: Csr = Csr::new(0x3be);
    pub const PMPADDR15: Csr = Csr::new(0x3bf);
    pub const PMPADDR16: Csr = Csr::new(0x3c0);
    pub const PMPADDR17: Csr = Csr::new(0x3c1);
    pub const PMPADDR18: Csr = Csr::new(0x3c2);
    pub const PMPADDR19: Csr = Csr::new(0x3c3);
    pub const PMPADDR20: Csr = Csr::new(0x3c4);
    pub const PMPADDR21: Csr = Csr::new(0x3c5);
    pub const PMPADDR22: Csr = Csr::new(0x3c6);
    pub const PMPADDR23: Csr = Csr::new(0x3c7);
    pub const PMPADDR24: Csr = Csr::new(0x3c8);
    pub const PMPADDR25: Csr = Csr::new(0x3c9);
    pub const PMPADDR26: Csr = Csr::new(0x3ca);
    pub const PMPADDR27: Csr = Csr::new(0x3cb);
    pub const PMPADDR28: Csr = Csr::new(0x3cc);
    pub const PMPADDR29: Csr = Csr::new(0x3cd);
    pub const PMPADDR30: Csr = Csr::new(0x3ce);
    pub const PMPADDR31: Csr = Csr::new(0x3cf);
    pub const PMPADDR32: Csr = Csr::new(0x3d0);
    pub const PMPADDR33: Csr = Csr::new(0x3d1);
    pub const PMPADDR34: Csr = Csr::new(0x3d2);
    pub const PMPADDR35: Csr = Csr::new(0x3d3);
    pub const PMPADDR36: Csr = Csr::new(0x3d4);
    pub const PMPADDR37: Csr = Csr::new(0x3d5);
    pub const PMPADDR38: Csr = Csr::new(0x3d6);
    pub const PMPADDR39: Csr = Csr::new(0x3d7);
    pub const PMPADDR40: Csr = Csr::new(0x3d8);
    pub const PMPADDR41: Csr = Csr::new(0x3d9);
    pub const PMPADDR42: Csr = Csr::new(0x3da);
    pub const PMPADDR43: Csr = Csr::new(0x3db);
    pub const PMPADDR44: Csr = Csr::new(0x3dc);
    pub const PMPADDR45: Csr = Csr::new(0x3dd);
    pub const PMPADDR46: Csr = Csr::new(0x3de);
    pub const PMPADDR47: Csr = Csr::new(0x3df);
    pub const PMPADDR48: Csr = Csr::new(0x3e0);
    pub const PMPADDR49: Csr = Csr::new(0x3e1);
    pub const PMPADDR50: Csr = Csr::new(0x3e2);
    pub const PMPADDR51: Csr = Csr::new(0x3e3);
    pub const PMPADDR52: Csr = Csr::new(0x3e4);
    pub const PMPADDR53: Csr = Csr::new(0x3e5);
    pub const PMPADDR54: Csr = Csr::new(0x3e6);
    pub const PMPADDR55: Csr = Csr::new(0x3e7);
    pub const PMPADDR56: Csr = Csr::new(0x3e8);
    pub const PMPADDR57: Csr = Csr::new(0x3e9);
    pub const PMPADDR58: Csr = Csr::new(0x3ea);
    pub const PMPADDR59: Csr = Csr::new(0x3eb);
    pub const PMPADDR60: Csr = Csr::new(0x3ec);
    pub const PMPADDR61: Csr = Csr::new(0x3ed);
    pub const PMPADDR62: Csr = Csr::new(0x3ee);
    pub const PMPADDR63: Csr = Csr::new(0x3ef);

    // Machine Counter/Timers
    pub const MCYCLE: Csr = Csr::new(0xb00);
    pub const MINSTRET: Csr = Csr::new(0xb02);
    pub const MHPMCOUNTER3: Csr = Csr::new(0xb03);
    pub const MHPMCOUNTER4: Csr = Csr::new(0xb04);
    pub const MHPMCOUNTER5: Csr = Csr::new(0xb05);
    pub const MHPMCOUNTER6: Csr = Csr::new(0xb06);
    pub const MHPMCOUNTER7: Csr = Csr::new(0xb07);
    pub const MHPMCOUNTER8: Csr = Csr::new(0xb08);
    pub const MHPMCOUNTER9: Csr = Csr::new(0xb09);
    pub const MHPMCOUNTER10: Csr = Csr::new(0xb0a);
    pub const MHPMCOUNTER11: Csr = Csr::new(0xb0b);
    pub const MHPMCOUNTER12: Csr = Csr::new(0xb0c);
    pub const MHPMCOUNTER13: Csr = Csr::new(0xb0d);
    pub const MHPMCOUNTER14: Csr = Csr::new(0xb0e);
    pub const MHPMCOUNTER15: Csr = Csr::new(0xb0f);
    pub const MHPMCOUNTER16: Csr = Csr::new(0xb10);
    pub const MHPMCOUNTER17: Csr = Csr::new(0xb11);
    pub const MHPMCOUNTER18: Csr = Csr::new(0xb12);
    pub const MHPMCOUNTER19: Csr = Csr::new(0xb13);
    pub const MHPMCOUNTER20: Csr = Csr::new(0xb14);
    pub const MHPMCOUNTER21: Csr = Csr::new(0xb15);
    pub const MHPMCOUNTER22: Csr = Csr::new(0xb16);
    pub const MHPMCOUNTER23: Csr = Csr::new(0xb17);
    pub const MHPMCOUNTER24: Csr = Csr::new(0xb18);
    pub const MHPMCOUNTER25: Csr = Csr::new(0xb19);
    pub const MHPMCOUNTER26: Csr = Csr::new(0xb1a);
    pub const MHPMCOUNTER27: Csr = Csr::new(0xb1b);
    pub const MHPMCOUNTER28: Csr = Csr::new(0xb1c);
    pub const MHPMCOUNTER29: Csr = Csr::new(0xb1d);
    pub const MHPMCOUNTER30: Csr = Csr::new(0xb1e);
    pub const MHPMCOUNTER31: Csr = Csr::new(0xb1f);

    // Machine Counter/Timers - High (for RV32)
    pub const MCYCLEH: Csr = Csr::new(0xb80);
    pub const MINSTRETH: Csr = Csr::new(0xb82);
    pub const MHPMCOUNTER3H: Csr = Csr::new(0xb83);
    pub const MHPMCOUNTER4H: Csr = Csr::new(0xb84);
    pub const MHPMCOUNTER5H: Csr = Csr::new(0xb85);
    pub const MHPMCOUNTER6H: Csr = Csr::new(0xb86);
    pub const MHPMCOUNTER7H: Csr = Csr::new(0xb87);
    pub const MHPMCOUNTER8H: Csr = Csr::new(0xb88);
    pub const MHPMCOUNTER9H: Csr = Csr::new(0xb89);
    pub const MHPMCOUNTER10H: Csr = Csr::new(0xb8a);
    pub const MHPMCOUNTER11H: Csr = Csr::new(0xb8b);
    pub const MHPMCOUNTER12H: Csr = Csr::new(0xb8c);
    pub const MHPMCOUNTER13H: Csr = Csr::new(0xb8d);
    pub const MHPMCOUNTER14H: Csr = Csr::new(0xb8e);
    pub const MHPMCOUNTER15H: Csr = Csr::new(0xb8f);
    pub const MHPMCOUNTER16H: Csr = Csr::new(0xb90);
    pub const MHPMCOUNTER17H: Csr = Csr::new(0xb91);
    pub const MHPMCOUNTER18H: Csr = Csr::new(0xb92);
    pub const MHPMCOUNTER19H: Csr = Csr::new(0xb93);
    pub const MHPMCOUNTER20H: Csr = Csr::new(0xb94);
    pub const MHPMCOUNTER21H: Csr = Csr::new(0xb95);
    pub const MHPMCOUNTER22H: Csr = Csr::new(0xb96);
    pub const MHPMCOUNTER23H: Csr = Csr::new(0xb97);
    pub const MHPMCOUNTER24H: Csr = Csr::new(0xb98);
    pub const MHPMCOUNTER25H: Csr = Csr::new(0xb99);
    pub const MHPMCOUNTER26H: Csr = Csr::new(0xb9a);
    pub const MHPMCOUNTER27H: Csr = Csr::new(0xb9b);
    pub const MHPMCOUNTER28H: Csr = Csr::new(0xb9c);
    pub const MHPMCOUNTER29H: Csr = Csr::new(0xb9d);
    pub const MHPMCOUNTER30H: Csr = Csr::new(0xb9e);
    pub const MHPMCOUNTER31H: Csr = Csr::new(0xb9f);

    // Machine Counter Setup
    pub const MCOUNTINHIBIT: Csr = Csr::new(0x320);
    pub const MHPMEVENT3: Csr = Csr::new(0x323);
    pub const MHPMEVENT4: Csr = Csr::new(0x324);
    pub const MHPMEVENT5: Csr = Csr::new(0x325);
    pub const MHPMEVENT6: Csr = Csr::new(0x326);
    pub const MHPMEVENT7: Csr = Csr::new(0x327);
    pub const MHPMEVENT8: Csr = Csr::new(0x328);
    pub const MHPMEVENT9: Csr = Csr::new(0x329);
    pub const MHPMEVENT10: Csr = Csr::new(0x32a);
    pub const MHPMEVENT11: Csr = Csr::new(0x32b);
    pub const MHPMEVENT12: Csr = Csr::new(0x32c);
    pub const MHPMEVENT13: Csr = Csr::new(0x32d);
    pub const MHPMEVENT14: Csr = Csr::new(0x32e);
    pub const MHPMEVENT15: Csr = Csr::new(0x32f);
    pub const MHPMEVENT16: Csr = Csr::new(0x330);
    pub const MHPMEVENT17: Csr = Csr::new(0x331);
    pub const MHPMEVENT18: Csr = Csr::new(0x332);
    pub const MHPMEVENT19: Csr = Csr::new(0x333);
    pub const MHPMEVENT20: Csr = Csr::new(0x334);
    pub const MHPMEVENT21: Csr = Csr::new(0x335);
    pub const MHPMEVENT22: Csr = Csr::new(0x336);
    pub const MHPMEVENT23: Csr = Csr::new(0x337);
    pub const MHPMEVENT24: Csr = Csr::new(0x338);
    pub const MHPMEVENT25: Csr = Csr::new(0x339);
    pub const MHPMEVENT26: Csr = Csr::new(0x33a);
    pub const MHPMEVENT27: Csr = Csr::new(0x33b);
    pub const MHPMEVENT28: Csr = Csr::new(0x33c);
    pub const MHPMEVENT29: Csr = Csr::new(0x33d);
    pub const MHPMEVENT30: Csr = Csr::new(0x33e);
    pub const MHPMEVENT31: Csr = Csr::new(0x33f);

    // Supervisor-mode CSRs
    pub const SSTATUS: Csr = Csr::new(0x100);
    pub const SIE: Csr = Csr::new(0x104);
    pub const STVEC: Csr = Csr::new(0x105);
    pub const SSCRATCH: Csr = Csr::new(0x140);
    pub const SEPC: Csr = Csr::new(0x141);
    pub const SCAUSE: Csr = Csr::new(0x142);
    pub const STVAL: Csr = Csr::new(0x143);
    pub const SIP: Csr = Csr::new(0x144);
}

/// Common registers
pub mod reg {
    use super::Register;

    // Standard register names (x0-x31)
    pub const X0: Register = Register::new(0);
    pub const X1: Register = Register::new(1);
    pub const X2: Register = Register::new(2);
    pub const X3: Register = Register::new(3);
    pub const X4: Register = Register::new(4);
    pub const X5: Register = Register::new(5);
    pub const X6: Register = Register::new(6);
    pub const X7: Register = Register::new(7);
    pub const X8: Register = Register::new(8);
    pub const X9: Register = Register::new(9);
    pub const X10: Register = Register::new(10);
    pub const X11: Register = Register::new(11);
    pub const X12: Register = Register::new(12);
    pub const X13: Register = Register::new(13);
    pub const X14: Register = Register::new(14);
    pub const X15: Register = Register::new(15);
    pub const X16: Register = Register::new(16);
    pub const X17: Register = Register::new(17);
    pub const X18: Register = Register::new(18);
    pub const X19: Register = Register::new(19);
    pub const X20: Register = Register::new(20);
    pub const X21: Register = Register::new(21);
    pub const X22: Register = Register::new(22);
    pub const X23: Register = Register::new(23);
    pub const X24: Register = Register::new(24);
    pub const X25: Register = Register::new(25);
    pub const X26: Register = Register::new(26);
    pub const X27: Register = Register::new(27);
    pub const X28: Register = Register::new(28);
    pub const X29: Register = Register::new(29);
    pub const X30: Register = Register::new(30);
    pub const X31: Register = Register::new(31);

    // RISC-V ABI register aliases
    pub const ZERO: Register = X0; // Hard-wired zero
    pub const RA: Register = X1; // Return address
    pub const SP: Register = X2; // Stack pointer
    pub const GP: Register = X3; // Global pointer
    pub const TP: Register = X4; // Thread pointer
    pub const T0: Register = X5; // Temporary register 0
    pub const T1: Register = X6; // Temporary register 1
    pub const T2: Register = X7; // Temporary register 2
    pub const S0: Register = X8; // Saved register 0 / Frame pointer
    pub const FP: Register = X8; // Frame pointer (alias for s0)
    pub const S1: Register = X9; // Saved register 1
    pub const A0: Register = X10; // Function argument 0 / Return value 0
    pub const A1: Register = X11; // Function argument 1 / Return value 1
    pub const A2: Register = X12; // Function argument 2
    pub const A3: Register = X13; // Function argument 3
    pub const A4: Register = X14; // Function argument 4
    pub const A5: Register = X15; // Function argument 5
    pub const A6: Register = X16; // Function argument 6
    pub const A7: Register = X17; // Function argument 7
    pub const S2: Register = X18; // Saved register 2
    pub const S3: Register = X19; // Saved register 3
    pub const S4: Register = X20; // Saved register 4
    pub const S5: Register = X21; // Saved register 5
    pub const S6: Register = X22; // Saved register 6
    pub const S7: Register = X23; // Saved register 7
    pub const S8: Register = X24; // Saved register 8
    pub const S9: Register = X25; // Saved register 9
    pub const S10: Register = X26; // Saved register 10
    pub const S11: Register = X27; // Saved register 11
    pub const T3: Register = X28; // Temporary register 3
    pub const T4: Register = X29; // Temporary register 4
    pub const T5: Register = X30; // Temporary register 5
    pub const T6: Register = X31; // Temporary register 6
}
