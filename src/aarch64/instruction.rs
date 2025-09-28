/// AArch64 instruction formats and encoding
use core::fmt;
use crate::common::{
    Instruction as InstructionTrait,
    Register as RegisterTrait,
};

#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// AArch64 register representation (32 general-purpose registers)
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
            0..=7 => AbiClass::CallerSaved,     // X0-X7: Argument/return value registers
            8..=15 => AbiClass::CallerSaved,    // X8-X15: Caller-saved temporary registers
            16..=17 => AbiClass::CallerSaved,   // X16-X17: Intra-procedure-call registers
            18 => AbiClass::CallerSaved,        // X18: Platform register (caller-saved on most platforms)
            
            // Callee-saved registers (must be preserved across calls)
            19..=28 => AbiClass::CalleeSaved,   // X19-X28: Callee-saved registers
            
            // Special-purpose registers
            29 => AbiClass::Special,  // X29: Frame pointer (FP)
            30 => AbiClass::Special,  // X30: Link register (LR)
            31 => AbiClass::Special,  // X31: Stack pointer (SP) or zero register (XZR)

            // Default to Special for any unhandled registers
            _ => AbiClass::Special,
        }
    }
}

/// AArch64 instruction representation (32-bit fixed-width instructions)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instruction(pub u32);

impl Instruction {
    /// Create a new 32-bit instruction
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    /// Get the instruction value as u32
    pub fn value(self) -> u32 {
        self.0
    }

    /// Get the instruction as bytes (little-endian)
    pub fn bytes(self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }
}

impl InstructionTrait for Instruction {
    fn value(&self) -> u64 {
        self.0 as u64
    }
    
    fn bytes(&self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }
    
    fn size(&self) -> usize {
        4
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:08x}", self.0)
    }
}

/// Data Processing - Register instruction encoding (3-operand)  
pub fn encode_add_sub_reg(sf: u8, op: u8, s: u8, rm: Register, imm6: u8, rn: Register, rd: Register) -> Instruction {
    // ADD/SUB (shifted register) encoding according to AArch64 ISA
    // 31: sf (0=32-bit, 1=64-bit)
    // 30: op (0=ADD, 1=SUB) 
    // 29: s (0=don't set flags, 1=set flags)
    // 28-24: 01011 (fixed for shifted register)
    // 23-22: shift (00=LSL, 01=LSR, 10=ASR, 11=reserved)
    // 21: 0 (fixed)
    // 20-16: Rm 
    // 15-10: imm6 (shift amount)
    // 9-5: Rn
    // 4-0: Rd
    let instr = ((sf as u32) << 31) |
                ((op as u32) << 30) |
                ((s as u32) << 29) |
                (0b01011 << 24) |     // Fixed bits for shifted register format
                (0b00 << 22) |        // shift = 00 (LSL)
                (0 << 21) |           // Fixed bit
                ((rm.value() as u32) << 16) |
                ((imm6 as u32) << 10) |
                ((rn.value() as u32) << 5) |
                (rd.value() as u32);
    Instruction::new(instr)
}

/// Data Processing - Immediate instruction encoding (ADD/SUB immediate)
pub fn encode_add_sub_imm(sf: u8, op: u8, s: u8, sh: u8, imm12: u16, rn: Register, rd: Register) -> Instruction {
    let instr = ((sf as u32) << 31) |
                ((op as u32) << 30) |
                ((s as u32) << 29) |
                (0b10001 << 24) |  // Fixed bits for data processing immediate
                ((sh as u32) << 22) |
                ((imm12 as u32) << 10) |
                ((rn.value() as u32) << 5) |
                (rd.value() as u32);
    Instruction::new(instr)
}

/// Logical instruction encoding (register)
pub fn encode_logical_reg(sf: u8, opc: u8, shift: u8, n: u8, rm: Register, imm6: u8, rn: Register, rd: Register) -> Instruction {
    let instr = ((sf as u32) << 31) |
                ((opc as u32) << 29) |
                (0b01010 << 24) |  // Fixed bits for logical register
                ((shift as u32) << 22) |
                ((n as u32) << 21) |
                ((rm.value() as u32) << 16) |
                ((imm6 as u32) << 10) |
                ((rn.value() as u32) << 5) |
                (rd.value() as u32);
    Instruction::new(instr)
}

/// Multiply instruction encoding
pub fn encode_multiply(sf: u8, op31: u8, rm: Register, o0: u8, ra: Register, rn: Register, rd: Register) -> Instruction {
    // For MUL x0, x1, x2 -> encoding should be 0x9b027c20
    // sf=1 (64-bit), op31=00, rm=2, o0=0, ra=31(XZR for MUL), rn=1, rd=0
    let instr = ((sf as u32) << 31) |
                (0b00011011 << 23) |  // Data processing 3-source
                ((op31 as u32) << 21) |
                ((rm.value() as u32) << 16) |
                ((o0 as u32) << 15) |
                ((ra.value() as u32) << 10) |
                ((rn.value() as u32) << 5) |
                (rd.value() as u32);
    Instruction::new(instr)
}

/// Division instruction encoding (same as multiply but with different op31/o0)
pub fn encode_divide(sf: u8, op31: u8, rm: Register, o0: u8, rn: Register, rd: Register) -> Instruction {
    // Division uses ra=31 (XZR) since it's a 2-operand instruction
    encode_multiply(sf, op31, rm, o0, Register::new(31), rn, rd)
}

/// Move instruction encoding (ORR with XZR)
pub fn encode_move_reg(sf: u8, rm: Register, rd: Register) -> Instruction {
    // MOV Rd, Rm -> ORR Rd, XZR, Rm
    encode_logical_reg(sf, 0b01, 0b00, 0, rm, 0, Register::new(31), rd)
}

/// Return instruction encoding (RET)
pub fn encode_ret(rn: Register) -> Instruction {
    // RET instruction -> encoding should be 0xd65f03c0 for ret (X30 implied)
    // Unconditional branch (register) format: 1101011 0 Z M 11111 000000 Rn 00000
    // For RET: Z=1, M=0, Rn=30 (LR)
    let instr = (0b1101011 << 25) |   // bits 31-25: 1101011
                (0b0 << 24) |         // bit 24: 0
                (0b1 << 23) |         // bit 23: Z=1 for RET
                (0b0 << 22) |         // bit 22: M=0
                (0b11111 << 16) |     // bits 21-16: 11111
                (0b000000 << 10) |    // bits 15-10: 000000
                ((rn.value() as u32) << 5) |  // bits 9-5: Rn
                0b00000;              // bits 4-0: 00000
    Instruction::new(instr)
}

/// Branch register instruction encoding (BR)
pub fn encode_branch_reg(opc: u8, op2: u8, op3: u8, rn: Register, op4: u8) -> Instruction {
    let instr = (0b1101011 << 25) |
                ((opc as u32) << 21) |
                ((op2 as u32) << 16) |
                ((op3 as u32) << 10) |
                ((rn.value() as u32) << 5) |
                (op4 as u32);
    Instruction::new(instr)
}

/// Common registers
pub mod reg {
    use super::Register;
    
    // Standard register names (X0-X30)
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
    
    // X31 is special - it's SP in some contexts, XZR/WZR in others
    pub const SP: Register = Register::new(31);   // Stack pointer
    pub const XZR: Register = Register::new(31);  // Zero register (64-bit)
    pub const WZR: Register = Register::new(31);  // Zero register (32-bit)

    // AArch64 ABI register aliases
    pub const FP: Register = X29;    // Frame pointer
    pub const LR: Register = X30;    // Link register
}