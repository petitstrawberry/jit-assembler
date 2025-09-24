/// Instruction builder interface for RISC-V assembly generation
use super::instruction::*;
use crate::common::InstructionBuilder as BuilderTrait;

#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Instruction builder for generating RISC-V instructions
pub struct InstructionBuilder {
    instructions: Vec<Instruction>,
}

impl InstructionBuilder {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    pub fn push(&mut self, instr: Instruction) -> &mut Self {
        self.instructions.push(instr);
        self
    }

    pub fn clear(&mut self) -> &mut Self {
        self.instructions.clear();
        self
    }
}

impl BuilderTrait<Instruction> for InstructionBuilder {
    fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    fn push(&mut self, instr: Instruction) {
        self.instructions.push(instr);
    }

    fn clear(&mut self) {
        self.instructions.clear();
    }
}

impl InstructionBuilder {
    /// Generate CSR read-write instruction
    pub fn csrrw(&mut self, rd: Register, csr: Csr, rs1: Register) -> &mut Self {
        let instr = encode_csr_type(opcodes::SYSTEM, rd, system_funct3::CSRRW, rs1, csr);
        self.push(instr);
        self
    }

    /// Generate CSR read-set instruction
    pub fn csrrs(&mut self, rd: Register, csr: Csr, rs1: Register) -> &mut Self {
        let instr = encode_csr_type(opcodes::SYSTEM, rd, system_funct3::CSRRS, rs1, csr);
        self.push(instr);
        self
    }

    /// Generate CSR read-clear instruction
    pub fn csrrc(&mut self, rd: Register, csr: Csr, rs1: Register) -> &mut Self {
        let instr = encode_csr_type(opcodes::SYSTEM, rd, system_funct3::CSRRC, rs1, csr);
        self.push(instr);
        self
    }

    /// Generate CSR read-write immediate instruction
    pub fn csrrwi(&mut self, rd: Register, csr: Csr, uimm: u8) -> &mut Self {
        let instr = encode_csr_imm_type(opcodes::SYSTEM, rd, system_funct3::CSRRWI, uimm, csr);
        self.push(instr);
        self
    }

    /// Generate CSR read-set immediate instruction
    pub fn csrrsi(&mut self, rd: Register, csr: Csr, uimm: u8) -> &mut Self {
        let instr = encode_csr_imm_type(opcodes::SYSTEM, rd, system_funct3::CSRRSI, uimm, csr);
        self.push(instr);
        self
    }

    /// Generate CSR read-clear immediate instruction
    pub fn csrrci(&mut self, rd: Register, csr: Csr, uimm: u8) -> &mut Self {
        let instr = encode_csr_imm_type(opcodes::SYSTEM, rd, system_funct3::CSRRCI, uimm, csr);
        self.push(instr);
        self
    }

    /// CSR read (alias for csrrs with rs1=x0)
    /// This is a common alias in RISC-V assembly
    pub fn csrr(&mut self, rd: Register, csr: Csr) -> &mut Self {
        self.csrrs(rd, csr, super::reg::X0)
    }

    /// Generate add immediate instruction
    pub fn addi(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::OP_IMM, rd, alu_funct3::ADD_SUB, rs1, imm);
        self.push(instr);
        self
    }

    /// Generate add instruction
    pub fn add(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, alu_funct3::ADD_SUB, rs1, rs2, 0x0);
        self.push(instr);
        self
    }

    /// Generate subtract instruction
    pub fn sub(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, alu_funct3::ADD_SUB, rs1, rs2, 0x20);
        self.push(instr);
        self
    }

    /// Generate XOR instruction
    pub fn xor(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, alu_funct3::XOR, rs1, rs2, 0x0);
        self.push(instr);
        self
    }

    /// Generate OR instruction
    pub fn or(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, alu_funct3::OR, rs1, rs2, 0x0);
        self.push(instr);
        self
    }

    /// Generate AND instruction
    pub fn and(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, alu_funct3::AND, rs1, rs2, 0x0);
        self.push(instr);
        self
    }

    /// Generate JAL (Jump and Link) instruction
    pub fn jal(&mut self, rd: Register, imm: i32) -> &mut Self {
        let instr = encode_j_type(opcodes::JAL, rd, imm);
        self.push(instr);
        self
    }

    /// Generate JALR (Jump and Link Register) instruction
    pub fn jalr(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::JALR, rd, 0x0, rs1, imm);
        self.push(instr);
        self
    }

    /// Generate BEQ (Branch if Equal) instruction
    pub fn beq(&mut self, rs1: Register, rs2: Register, imm: i16) -> &mut Self {
        let instr = encode_b_type(opcodes::BRANCH, branch_funct3::BEQ, rs1, rs2, imm);
        self.push(instr);
        self
    }

    /// Generate BNE (Branch if Not Equal) instruction
    pub fn bne(&mut self, rs1: Register, rs2: Register, imm: i16) -> &mut Self {
        let instr = encode_b_type(opcodes::BRANCH, branch_funct3::BNE, rs1, rs2, imm);
        self.push(instr);
        self
    }

    /// Generate BLT (Branch if Less Than) instruction
    pub fn blt(&mut self, rs1: Register, rs2: Register, imm: i16) -> &mut Self {
        let instr = encode_b_type(opcodes::BRANCH, branch_funct3::BLT, rs1, rs2, imm);
        self.push(instr);
        self
    }

    /// Generate BGE (Branch if Greater or Equal) instruction
    pub fn bge(&mut self, rs1: Register, rs2: Register, imm: i16) -> &mut Self {
        let instr = encode_b_type(opcodes::BRANCH, branch_funct3::BGE, rs1, rs2, imm);
        self.push(instr);
        self
    }

    /// Generate BLTU (Branch if Less Than Unsigned) instruction
    pub fn bltu(&mut self, rs1: Register, rs2: Register, imm: i16) -> &mut Self {
        let instr = encode_b_type(opcodes::BRANCH, branch_funct3::BLTU, rs1, rs2, imm);
        self.push(instr);
        self
    }

    /// Generate BGEU (Branch if Greater or Equal Unsigned) instruction
    pub fn bgeu(&mut self, rs1: Register, rs2: Register, imm: i16) -> &mut Self {
        let instr = encode_b_type(opcodes::BRANCH, branch_funct3::BGEU, rs1, rs2, imm);
        self.push(instr);
        self
    }

    /// Generate LD (Load Doubleword) instruction
    pub fn ld(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::LOAD, rd, load_funct3::LD, rs1, imm);
        self.push(instr);
        self
    }

    /// Generate LW (Load Word) instruction
    pub fn lw(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::LOAD, rd, load_funct3::LW, rs1, imm);
        self.push(instr);
        self
    }

    /// Generate LH (Load Halfword) instruction
    pub fn lh(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::LOAD, rd, load_funct3::LH, rs1, imm);
        self.push(instr);
        self
    }

    /// Generate LB (Load Byte) instruction
    pub fn lb(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::LOAD, rd, load_funct3::LB, rs1, imm);
        self.push(instr);
        self
    }

    /// Generate SD (Store Doubleword) instruction
    pub fn sd(&mut self, rs1: Register, rs2: Register, imm: i16) -> &mut Self {
        let instr = encode_s_type(opcodes::STORE, store_funct3::SD, rs1, rs2, imm);
        self.push(instr);
        self
    }

    /// Generate SW (Store Word) instruction
    pub fn sw(&mut self, rs1: Register, rs2: Register, imm: i16) -> &mut Self {
        let instr = encode_s_type(opcodes::STORE, store_funct3::SW, rs1, rs2, imm);
        self.push(instr);
        self
    }

    /// Generate SH (Store Halfword) instruction
    pub fn sh(&mut self, rs1: Register, rs2: Register, imm: i16) -> &mut Self {
        let instr = encode_s_type(opcodes::STORE, store_funct3::SH, rs1, rs2, imm);
        self.push(instr);
        self
    }

    /// Generate SB (Store Byte) instruction
    pub fn sb(&mut self, rs1: Register, rs2: Register, imm: i16) -> &mut Self {
        let instr = encode_s_type(opcodes::STORE, store_funct3::SB, rs1, rs2, imm);
        self.push(instr);
        self
    }

    /// Generate SLL (Shift Left Logical) instruction
    pub fn sll(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, alu_funct3::SLL, rs1, rs2, 0x0);
        self.push(instr);
        self
    }

    /// Generate SRL (Shift Right Logical) instruction
    pub fn srl(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, alu_funct3::SRL_SRA, rs1, rs2, 0x0);
        self.push(instr);
        self
    }

    /// Generate SRA (Shift Right Arithmetic) instruction
    pub fn sra(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, alu_funct3::SRL_SRA, rs1, rs2, 0x20);
        self.push(instr);
        self
    }

    /// Generate SLLI (Shift Left Logical Immediate) instruction
    pub fn slli(&mut self, rd: Register, rs1: Register, shamt: u8) -> &mut Self {
        let instr = encode_i_type(opcodes::OP_IMM, rd, alu_funct3::SLL, rs1, shamt as i16);
        self.push(instr);
        self
    }

    /// Generate SRLI (Shift Right Logical Immediate) instruction
    pub fn srli(&mut self, rd: Register, rs1: Register, shamt: u8) -> &mut Self {
        let instr = encode_i_type(opcodes::OP_IMM, rd, alu_funct3::SRL_SRA, rs1, shamt as i16);
        self.push(instr);
        self
    }

    /// Generate SRAI (Shift Right Arithmetic Immediate) instruction
    pub fn srai(&mut self, rd: Register, rs1: Register, shamt: u8) -> &mut Self {
        let instr = encode_i_type(opcodes::OP_IMM, rd, alu_funct3::SRL_SRA, rs1, (shamt as i16) | 0x400);
        self.push(instr);
        self
    }

    /// Generate LUI (Load Upper Immediate) instruction
    pub fn lui(&mut self, rd: Register, imm: u32) -> &mut Self {
        let instr = encode_u_type(opcodes::LUI, rd, imm);
        self.push(instr);
        self
    }

    /// Generate AUIPC (Add Upper Immediate to PC) instruction
    pub fn auipc(&mut self, rd: Register, imm: u32) -> &mut Self {
        let instr = encode_u_type(opcodes::AUIPC, rd, imm);
        self.push(instr);
        self
    }
}