/// Instruction builder interface for RISC-V assembly generation
use super::instruction::*;
use crate::common::InstructionBuilder;

#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::collections::HashSet;
#[cfg(not(feature = "std"))]
use alloc::collections::BTreeSet as HashSet;

/// Instruction builder for generating RISC-V instructions
pub struct Riscv64InstructionBuilder {
    instructions: Vec<Instruction>,
}

impl Riscv64InstructionBuilder {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    /// Returns a slice of the raw instructions.
    ///
    /// This method exposes the internal instruction buffer directly as a slice.
    /// Prefer using the `instructions()` method from the `InstructionBuilder` trait
    /// for most use cases, as it provides a higher-level abstraction and is part of
    /// the public API. Use `raw_instructions` only if you specifically need access
    /// to the underlying slice for migration or performance reasons.
    pub fn raw_instructions(&self) -> &[Instruction] {
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

impl InstructionBuilder<Instruction> for Riscv64InstructionBuilder {
    fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    fn instructions(&self) -> crate::common::InstructionCollection<Instruction> {
        crate::common::InstructionCollection::from_slice(&self.instructions)
    }

    fn push(&mut self, instr: Instruction) {
        self.instructions.push(instr);
    }

    fn clear(&mut self) {
        self.instructions.clear();
    }
    
    /// Create a JIT-compiled function from the assembled instructions (std-only)
    /// 
    /// This method converts the assembled instructions into executable machine code
    /// that can be called directly as a function. The generic type parameter `F`
    /// specifies the function signature.
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because:
    /// - It allocates executable memory
    /// - It assumes the assembled code follows the correct ABI
    /// - The caller must ensure the function signature matches the actual code
    /// 
    /// # Examples
    /// 
    /// ```rust,no_run
    /// use jit_assembler::riscv::{reg, Riscv64InstructionBuilder};
    /// use jit_assembler::common::InstructionBuilder;
    /// 
    /// let add_func = unsafe {
    ///     Riscv64InstructionBuilder::new()
    ///         .add(reg::A0, reg::A0, reg::A1) // Add first two arguments
    ///         .ret()
    ///         .function::<fn(u64, u64) -> u64>()
    /// }.expect("Failed to create JIT function");
    /// 
    /// // Call the JIT function directly (only works on RISC-V hosts)
    /// let result = add_func.call(10, 20); // Should return 30
    /// ```
    #[cfg(feature = "std")]
    unsafe fn function<F>(&self) -> Result<crate::common::jit::CallableJitFunction<F>, crate::common::jit::JitError> {
        // Convert instructions to bytes using the new Instructions struct
        let code = self.instructions().to_bytes();
        crate::common::jit::CallableJitFunction::<F>::new(&code)
    }

    #[cfg(feature = "std")]
    unsafe fn raw_function(&self) -> Result<crate::common::jit::RawCallableJitFunction, crate::common::jit::JitError> {
        let code = self.instructions().to_bytes();
        crate::common::jit::RawCallableJitFunction::new(&code)
    }
}

impl Riscv64InstructionBuilder {
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

    // Pseudo-instructions for convenience

    /// CSR read (alias for csrrs with rs1=x0)
    /// This is a common alias in RISC-V assembly
    pub fn csrr(&mut self, rd: Register, csr: Csr) -> &mut Self {
        self.csrrs(rd, csr, super::reg::X0)
    }

    /// CSR write (alias for csrrw with rd=x0)
    /// This is a common alias in RISC-V assembly for writing to CSR without reading old value
    pub fn csrw(&mut self, csr: Csr, rs1: Register) -> &mut Self {
        self.csrrw(super::reg::X0, csr, rs1)
    }

    /// CSR set (alias for csrrs with rd=x0)
    /// This is a common alias in RISC-V assembly for setting bits in CSR without reading old value
    pub fn csrs(&mut self, csr: Csr, rs1: Register) -> &mut Self {
        self.csrrs(super::reg::X0, csr, rs1)
    }

    /// CSR clear (alias for csrrc with rd=x0)
    /// This is a common alias in RISC-V assembly for clearing bits in CSR without reading old value
    pub fn csrc(&mut self, csr: Csr, rs1: Register) -> &mut Self {
        self.csrrc(super::reg::X0, csr, rs1)
    }

    /// CSR write immediate (alias for csrrwi with rd=x0)
    /// This is a common alias in RISC-V assembly for writing immediate to CSR without reading old value
    pub fn csrwi(&mut self, csr: Csr, uimm: u8) -> &mut Self {
        self.csrrwi(super::reg::X0, csr, uimm)
    }

    /// CSR set immediate (alias for csrrsi with rd=x0)
    /// This is a common alias in RISC-V assembly for setting bits in CSR with immediate without reading old value
    pub fn csrsi(&mut self, csr: Csr, uimm: u8) -> &mut Self {
        self.csrrsi(super::reg::X0, csr, uimm)
    }

    /// CSR clear immediate (alias for csrrci with rd=x0)
    /// This is a common alias in RISC-V assembly for clearing bits in CSR with immediate without reading old value
    pub fn csrci(&mut self, csr: Csr, uimm: u8) -> &mut Self {
        self.csrrci(super::reg::X0, csr, uimm)
    }

    // ALU instructions

    /// Generate add instruction
    pub fn add(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, alu_funct3::ADD_SUB, rs1, rs2, 0x0);
        self.push(instr);
        self
    }
    /// Generate add immediate instruction
    pub fn addi(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::OP_IMM, rd, alu_funct3::ADD_SUB, rs1, imm);
        self.push(instr);
        self
    }


    /// Generate subtract instruction
    pub fn sub(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, alu_funct3::ADD_SUB, rs1, rs2, 0x20);
        self.push(instr);
        self
    }


    /// Generate subtract immediate instruction
    pub fn subi(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::OP_IMM, rd, alu_funct3::ADD_SUB, rs1, -imm);
        self.push(instr);
        self
    }

    /// Generate XOR instruction
    pub fn xor(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, alu_funct3::XOR, rs1, rs2, 0x0);
        self.push(instr);
        self
    }

    /// Generate XOR immediate instruction
    pub fn xori(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::OP_IMM, rd, alu_funct3::XOR, rs1, imm);
        self.push(instr);
        self
    }

    /// Generate OR instruction
    pub fn or(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, alu_funct3::OR, rs1, rs2, 0x0);
        self.push(instr);
        self
    }

    /// Generate OR immediate instruction
    pub fn ori(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::OP_IMM, rd, alu_funct3::OR, rs1, imm);
        self.push(instr);
        self
    }

    /// Generate Set Less Than instruction
    pub fn slt(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, alu_funct3::SLT, rs1, rs2, 0x0);
        self.push(instr);
        self
    }

    /// Generate Set Less Than immediate instruction
    pub fn slti(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::OP_IMM, rd, alu_funct3::SLT, rs1, imm);
        self.push(instr);
        self
    }

    /// Generate Set Less Than Unsigned instruction
    pub fn sltu(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, alu_funct3::SLTU, rs1, rs2, 0x0);
        self.push(instr);
        self
    }

    /// Generate Set Less Than immediate Unsigned instruction
    pub fn sltiu(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::OP_IMM, rd, alu_funct3::SLTU, rs1, imm);
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

    /// Generate AND instruction
    pub fn and(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, alu_funct3::AND, rs1, rs2, 0x0);
        self.push(instr);
        self
    }

    /// Generate AND immediate instruction
    pub fn andi(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::OP_IMM, rd, alu_funct3::AND, rs1, imm);
        self.push(instr);
        self
    }

    // M Extension (Multiply/Divide) instructions

    /// Generate MUL (Multiply) instruction
    /// Performs signed multiplication and returns the lower 64 bits of the result
    pub fn mul(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, m_funct3::MUL, rs1, rs2, m_funct7::M_EXT);
        self.push(instr);
        self
    }

    /// Generate MULH (Multiply High) instruction
    /// Performs signed × signed multiplication and returns the upper 64 bits of the result
    pub fn mulh(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, m_funct3::MULH, rs1, rs2, m_funct7::M_EXT);
        self.push(instr);
        self
    }

    /// Generate MULHSU (Multiply High Signed × Unsigned) instruction
    /// Performs signed × unsigned multiplication and returns the upper 64 bits of the result
    pub fn mulhsu(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, m_funct3::MULHSU, rs1, rs2, m_funct7::M_EXT);
        self.push(instr);
        self
    }

    /// Generate MULHU (Multiply High Unsigned) instruction
    /// Performs unsigned × unsigned multiplication and returns the upper 64 bits of the result
    pub fn mulhu(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, m_funct3::MULHU, rs1, rs2, m_funct7::M_EXT);
        self.push(instr);
        self
    }

    /// Generate DIV (Divide) instruction
    /// Performs signed division: rs1 ÷ rs2
    pub fn div(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, m_funct3::DIV, rs1, rs2, m_funct7::M_EXT);
        self.push(instr);
        self
    }

    /// Generate DIVU (Divide Unsigned) instruction
    /// Performs unsigned division: rs1 ÷ rs2
    pub fn divu(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, m_funct3::DIVU, rs1, rs2, m_funct7::M_EXT);
        self.push(instr);
        self
    }

    /// Generate REM (Remainder) instruction
    /// Computes signed remainder: rs1 % rs2
    pub fn rem(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, m_funct3::REM, rs1, rs2, m_funct7::M_EXT);
        self.push(instr);
        self
    }

    /// Generate REMU (Remainder Unsigned) instruction
    /// Computes unsigned remainder: rs1 % rs2
    pub fn remu(&mut self, rd: Register, rs1: Register, rs2: Register) -> &mut Self {
        let instr = encode_r_type(opcodes::OP, rd, m_funct3::REMU, rs1, rs2, m_funct7::M_EXT);
        self.push(instr);
        self
    }

    // Load/Store instructions

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

    /// Generate LBU (Load Byte Unsigned) instruction
    pub fn lbu(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::LOAD, rd, load_funct3::LBU, rs1, imm);
        self.push(instr);
        self
    }

    /// Generate LHU (Load Halfword Unsigned) instruction
    pub fn lhu(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::LOAD, rd, load_funct3::LHU, rs1, imm);
        self.push(instr);
        self
    }

    /// Generate LWU (Load Word Unsigned) instruction
    pub fn lwu(&mut self, rd: Register, rs1: Register, imm: i16) -> &mut Self {
        let instr = encode_i_type(opcodes::LOAD, rd, load_funct3::LWU, rs1, imm);
        self.push(instr);
        self
    }

    /// Load immediate value into register (handles large immediates)
    /// This is a common pseudo-instruction in RISC-V assembly
    pub fn li(&mut self, rd: Register, imm: i32) -> &mut Self {
        // LUI loads the upper 20 bits, ADDI adds the lower 12 bits
        let upper = (imm + 0x800) >> 12; // Round up if lower 12 bits are negative
        let lower = imm & 0xfff;
        if upper != 0 {
            self.lui(rd, upper as u32);
        }
        if lower != 0 || upper == 0 {
            self.addi(rd, rd, lower as i16);
        }
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

    // Control flow instructions

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

    // Privileged instructions

    /// Supervisor return instruction
    /// Returns from supervisor mode to user mode or previous privilege level
    /// This is a privileged instruction that can only be executed in supervisor mode or higher
    pub fn sret(&mut self) -> &mut Self {
        let instr = super::instruction::encode_privileged_type(
            super::instruction::opcodes::SYSTEM,
            super::instruction::privileged_funct12::SRET
        );
        self.push(instr);
        self
    }

    /// Machine return instruction
    /// Returns from machine mode to previous privilege level
    /// This is a privileged instruction that can only be executed in machine mode
    pub fn mret(&mut self) -> &mut Self {
        let instr = super::instruction::encode_privileged_type(
            super::instruction::opcodes::SYSTEM,
            super::instruction::privileged_funct12::MRET
        );
        self.push(instr);
        self
    }

    /// Environment call instruction
    /// Generates a system call to the execution environment
    pub fn ecall(&mut self) -> &mut Self {
        let instr = super::instruction::encode_privileged_type(
            super::instruction::opcodes::SYSTEM,
            super::instruction::privileged_funct12::ECALL
        );
        self.push(instr);
        self
    }

    /// Environment break instruction
    /// Generates a breakpoint exception
    pub fn ebreak(&mut self) -> &mut Self {
        let instr = super::instruction::encode_privileged_type(
            super::instruction::opcodes::SYSTEM,
            super::instruction::privileged_funct12::EBREAK
        );
        self.push(instr);
        self
    }

    /// Wait for interrupt instruction
    /// Puts the processor in a low-power state until an interrupt occurs
    /// This is a privileged instruction
    pub fn wfi(&mut self) -> &mut Self {
        let instr = super::instruction::encode_privileged_type(
            super::instruction::opcodes::SYSTEM,
            super::instruction::privileged_funct12::WFI
        );
        self.push(instr);
        self
    }

    /// Return instruction (alias for jalr x0, x1, 0)
    /// This is a common alias in RISC-V assembly for returning from a function
    pub fn ret(&mut self) -> &mut Self {
        self.jalr(super::reg::X0, super::reg::X1, 0)
    }
}