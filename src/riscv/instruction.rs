/// RISC-V instruction formats and encoding
use core::fmt;
use crate::common::{
    Instruction as InstructionTrait,
    Register as RegisterTrait,
};

#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// RISC-V register representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    let instr = (imm << 20) | ((rs1.value() as u32) << 15) | ((funct3 as u32) << 12) | ((rd.value() as u32) << 7) | (opcode as u32);
    Instruction::Standard(instr)
}

/// CSR instruction encoding (I-type variant)
pub fn encode_csr_type(opcode: u8, rd: Register, funct3: u8, rs1: Register, csr: Csr) -> Instruction {
    let csr_val = csr.value() as u32;
    let instr = (csr_val << 20) | ((rs1.value() as u32) << 15) | ((funct3 as u32) << 12) | ((rd.value() as u32) << 7) | (opcode as u32);
    Instruction::Standard(instr)
}

/// CSR immediate instruction encoding
pub fn encode_csr_imm_type(opcode: u8, rd: Register, funct3: u8, uimm: u8, csr: Csr) -> Instruction {
    let csr_val = csr.value() as u32;
    let instr = (csr_val << 20) | ((uimm as u32) << 15) | ((funct3 as u32) << 12) | ((rd.value() as u32) << 7) | (opcode as u32);
    Instruction::Standard(instr)
}

/// Privileged instruction encoding (special SYSTEM instructions)
pub fn encode_privileged_type(opcode: u8, funct12: u16) -> Instruction {
    let instr = ((funct12 as u32) << 20) | (opcode as u32);
    Instruction::Standard(instr)
}

/// R-type instruction encoding
pub fn encode_r_type(opcode: u8, rd: Register, funct3: u8, rs1: Register, rs2: Register, funct7: u8) -> Instruction {
    let instr = ((funct7 as u32) << 25) | ((rs2.value() as u32) << 20) | ((rs1.value() as u32) << 15) | ((funct3 as u32) << 12) | ((rd.value() as u32) << 7) | (opcode as u32);
    Instruction::Standard(instr)
}

/// S-type instruction encoding (Store)
pub fn encode_s_type(opcode: u8, funct3: u8, rs1: Register, rs2: Register, imm: i16) -> Instruction {
    let imm = imm as u32 & 0xfff;
    let imm_11_5 = (imm >> 5) & 0x7f;
    let imm_4_0 = imm & 0x1f;
    let instr = (imm_11_5 << 25) | ((rs2.value() as u32) << 20) | ((rs1.value() as u32) << 15) | ((funct3 as u32) << 12) | (imm_4_0 << 7) | (opcode as u32);
    Instruction::Standard(instr)
}

/// B-type instruction encoding (Branch)
pub fn encode_b_type(opcode: u8, funct3: u8, rs1: Register, rs2: Register, imm: i16) -> Instruction {
    let imm = (imm as u32) & 0x1ffe; // 13-bit signed immediate
    let imm_12 = (imm >> 12) & 0x1;
    let imm_10_5 = (imm >> 5) & 0x3f;
    let imm_4_1 = (imm >> 1) & 0xf;
    let imm_11 = (imm >> 11) & 0x1;
    let instr = (imm_12 << 31) | (imm_10_5 << 25) | ((rs2.value() as u32) << 20) | ((rs1.value() as u32) << 15) | ((funct3 as u32) << 12) | (imm_4_1 << 8) | (imm_11 << 7) | (opcode as u32);
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
    let instr = (imm_20 << 31) | (imm_10_1 << 21) | (imm_11 << 20) | (imm_19_12 << 12) | ((rd.value() as u32) << 7) | (opcode as u32);
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
    pub const ECALL: u16 = 0x000;  // Environment call
    pub const EBREAK: u16 = 0x001; // Environment break
    pub const SRET: u16 = 0x102;   // Supervisor return
    pub const MRET: u16 = 0x302;   // Machine return
    pub const WFI: u16 = 0x105;    // Wait for interrupt
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

/// Common CSR addresses
pub mod csr {
    use super::Csr;
    
    // Machine-mode CSRs
    pub const MSTATUS: Csr = Csr::new(0x300);
    pub const MISA: Csr = Csr::new(0x301);
    pub const MEDELEG: Csr = Csr::new(0x302);
    pub const MIDELEG: Csr = Csr::new(0x303);
    pub const MIE: Csr = Csr::new(0x304);
    pub const MTVEC: Csr = Csr::new(0x305);
    pub const MSCRATCH: Csr = Csr::new(0x340);
    pub const MEPC: Csr = Csr::new(0x341);
    pub const MCAUSE: Csr = Csr::new(0x342);
    pub const MTVAL: Csr = Csr::new(0x343);
    pub const MIP: Csr = Csr::new(0x344);
    pub const MHARTID: Csr = Csr::new(0xf14);
    
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
    pub const ZERO: Register = X0;  // Hard-wired zero
    pub const RA: Register = X1;    // Return address
    pub const SP: Register = X2;    // Stack pointer
    pub const GP: Register = X3;    // Global pointer
    pub const TP: Register = X4;    // Thread pointer
    pub const T0: Register = X5;    // Temporary register 0
    pub const T1: Register = X6;    // Temporary register 1
    pub const T2: Register = X7;    // Temporary register 2
    pub const S0: Register = X8;    // Saved register 0 / Frame pointer
    pub const FP: Register = X8;    // Frame pointer (alias for s0)
    pub const S1: Register = X9;    // Saved register 1
    pub const A0: Register = X10;   // Function argument 0 / Return value 0
    pub const A1: Register = X11;   // Function argument 1 / Return value 1
    pub const A2: Register = X12;   // Function argument 2
    pub const A3: Register = X13;   // Function argument 3
    pub const A4: Register = X14;   // Function argument 4
    pub const A5: Register = X15;   // Function argument 5
    pub const A6: Register = X16;   // Function argument 6
    pub const A7: Register = X17;   // Function argument 7
    pub const S2: Register = X18;   // Saved register 2
    pub const S3: Register = X19;   // Saved register 3
    pub const S4: Register = X20;   // Saved register 4
    pub const S5: Register = X21;   // Saved register 5
    pub const S6: Register = X22;   // Saved register 6
    pub const S7: Register = X23;   // Saved register 7
    pub const S8: Register = X24;   // Saved register 8
    pub const S9: Register = X25;   // Saved register 9
    pub const S10: Register = X26;  // Saved register 10
    pub const S11: Register = X27;  // Saved register 11
    pub const T3: Register = X28;   // Temporary register 3
    pub const T4: Register = X29;   // Temporary register 4
    pub const T5: Register = X30;   // Temporary register 5
    pub const T6: Register = X31;   // Temporary register 6
}