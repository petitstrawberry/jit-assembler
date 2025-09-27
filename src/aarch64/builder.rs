/// Instruction builder interface for AArch64 assembly generation
use super::instruction::*;
use crate::common::InstructionBuilder;

#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Instruction builder for generating AArch64 instructions
pub struct Aarch64InstructionBuilder {
    instructions: Vec<Instruction>,
    #[cfg(feature = "register-tracking")]
    register_usage: crate::common::register_usage::RegisterUsageInfo<Register>,
}

impl Aarch64InstructionBuilder {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            #[cfg(feature = "register-tracking")]
            register_usage: crate::common::register_usage::RegisterUsageInfo::new(),
        }
    }
    
    /// Track a written register (register-tracking feature only)
    #[cfg(feature = "register-tracking")]
    fn track_written_register(&mut self, reg: Register) {
        self.register_usage.add_written_register(reg);
    }
    
    /// Track a read register (register-tracking feature only)
    #[cfg(feature = "register-tracking")]
    fn track_read_register(&mut self, reg: Register) {
        self.register_usage.add_read_register(reg);
    }
    
    /// Track multiple read registers at once (register-tracking feature only)
    #[cfg(feature = "register-tracking")]
    fn track_read_registers(&mut self, regs: &[Register]) {
        for &reg in regs {
            self.register_usage.add_read_register(reg);
        }
    }
    
    /// No-op versions for when register-tracking is disabled
    #[cfg(not(feature = "register-tracking"))]
    fn track_written_register(&mut self, _reg: Register) {
        // No-op
    }
    
    #[cfg(not(feature = "register-tracking"))]
    fn track_read_register(&mut self, _reg: Register) {
        // No-op
    }
    
    #[cfg(not(feature = "register-tracking"))]
    fn track_read_registers(&mut self, _regs: &[Register]) {
        // No-op
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
        #[cfg(feature = "register-tracking")]
        self.register_usage.clear();
        self
    }

    // Arithmetic instructions

    /// Generate ADD instruction (64-bit register)
    /// ADD Xd, Xn, Xm
    pub fn add(&mut self, rd: Register, rn: Register, rm: Register) -> &mut Self {
        self.track_written_register(rd);
        self.track_read_registers(&[rn, rm]);
        let instr = encode_add_sub_reg(1, 0, 0, rm, 0, rn, rd); // sf=1 (64-bit), op=0 (ADD), s=0 (no flags)
        self.push(instr);
        self
    }

    /// Generate ADD immediate instruction (64-bit)
    /// ADD Xd, Xn, #imm
    pub fn addi(&mut self, rd: Register, rn: Register, imm: u16) -> &mut Self {
        self.track_written_register(rd);
        self.track_read_register(rn);
        let instr = encode_add_sub_imm(1, 0, 0, 0, imm, rn, rd); // sf=1, op=0 (ADD), s=0, sh=0
        self.push(instr);
        self
    }

    /// Generate SUB instruction (64-bit register)
    /// SUB Xd, Xn, Xm
    pub fn sub(&mut self, rd: Register, rn: Register, rm: Register) -> &mut Self {
        self.track_written_register(rd);
        self.track_read_registers(&[rn, rm]);
        let instr = encode_add_sub_reg(1, 1, 0, rm, 0, rn, rd); // sf=1, op=1 (SUB), s=0
        self.push(instr);
        self
    }

    /// Generate SUB immediate instruction (64-bit)
    /// SUB Xd, Xn, #imm
    pub fn subi(&mut self, rd: Register, rn: Register, imm: u16) -> &mut Self {
        self.track_written_register(rd);
        self.track_read_register(rn);
        let instr = encode_add_sub_imm(1, 1, 0, 0, imm, rn, rd); // sf=1, op=1 (SUB), s=0, sh=0
        self.push(instr);
        self
    }

    /// Generate MUL instruction (64-bit)
    /// MUL Xd, Xn, Xm (equivalent to MADD Xd, Xn, Xm, XZR)
    pub fn mul(&mut self, rd: Register, rn: Register, rm: Register) -> &mut Self {
        self.track_written_register(rd);
        self.track_read_registers(&[rn, rm]);
        let instr = encode_multiply(1, 0b000, rm, 0, Register::new(31), rn, rd); // sf=1, op31=000, o0=0, ra=XZR
        self.push(instr);
        self
    }

    /// Generate UDIV instruction (64-bit unsigned division)
    /// UDIV Xd, Xn, Xm
    pub fn udiv(&mut self, rd: Register, rn: Register, rm: Register) -> &mut Self {
        self.track_written_register(rd);
        self.track_read_registers(&[rn, rm]);
        let instr = encode_divide(1, 0b000, rm, 1, rn, rd); // sf=1, op31=000, o0=1 for UDIV
        self.push(instr);
        self
    }

    /// Generate SDIV instruction (64-bit signed division)
    /// SDIV Xd, Xn, Xm  
    pub fn sdiv(&mut self, rd: Register, rn: Register, rm: Register) -> &mut Self {
        self.track_written_register(rd);
        self.track_read_registers(&[rn, rm]);
        let instr = encode_divide(1, 0b001, rm, 1, rn, rd); // sf=1, op31=001, o0=1 for SDIV
        self.push(instr);
        self
    }

    /// Generate remainder operation using MSUB after division
    /// This creates a sequence: UDIV tmp, rn, rm; MSUB rd, tmp, rm, rn
    /// Result: rd = rn - (rn / rm) * rm = rn % rm
    pub fn urem(&mut self, rd: Register, rn: Register, rm: Register) -> &mut Self {
        // We need a temporary register - use X17 (IP1) which is caller-saved
        let tmp = reg::X17;
        
        // UDIV tmp, rn, rm
        self.track_read_registers(&[rn, rm]);
        let div_instr = encode_divide(1, 0b000, rm, 1, rn, tmp);
        self.push(div_instr);
        
        // MSUB rd, tmp, rm, rn  (rd = rn - tmp * rm)
        self.track_written_register(rd);
        self.track_read_registers(&[tmp, rm, rn]);
        let msub_instr = encode_multiply(1, 0b000, rm, 1, rn, tmp, rd); // MSUB: op31=000, o0=1
        self.push(msub_instr);
        self
    }

    /// Generate ORR instruction (logical OR, 64-bit)
    /// ORR Xd, Xn, Xm
    pub fn or(&mut self, rd: Register, rn: Register, rm: Register) -> &mut Self {
        self.track_written_register(rd);
        self.track_read_registers(&[rn, rm]);
        let instr = encode_logical_reg(1, 0b01, 0b00, 0, rm, 0, rn, rd); // sf=1, opc=01 (ORR), shift=00, n=0
        self.push(instr);
        self
    }

    /// Generate AND instruction (logical AND, 64-bit)
    /// AND Xd, Xn, Xm
    pub fn and(&mut self, rd: Register, rn: Register, rm: Register) -> &mut Self {
        self.track_written_register(rd);
        self.track_read_registers(&[rn, rm]);
        let instr = encode_logical_reg(1, 0b00, 0b00, 0, rm, 0, rn, rd); // sf=1, opc=00 (AND), shift=00, n=0
        self.push(instr);
        self
    }

    /// Generate EOR instruction (logical XOR, 64-bit)
    /// EOR Xd, Xn, Xm
    pub fn xor(&mut self, rd: Register, rn: Register, rm: Register) -> &mut Self {
        self.track_written_register(rd);
        self.track_read_registers(&[rn, rm]);
        let instr = encode_logical_reg(1, 0b10, 0b00, 0, rm, 0, rn, rd); // sf=1, opc=10 (EOR), shift=00, n=0
        self.push(instr);
        self
    }

    /// Generate MOV instruction (move register to register)
    /// MOV Xd, Xm (implemented as ORR Xd, XZR, Xm)
    pub fn mov(&mut self, rd: Register, rm: Register) -> &mut Self {
        self.track_written_register(rd);
        self.track_read_register(rm);
        let instr = encode_move_reg(1, rm, rd); // sf=1 (64-bit)
        self.push(instr);
        self
    }

    /// Generate RET instruction (return)
    /// RET (returns to X30/LR by default)
    pub fn ret(&mut self) -> &mut Self {
        self.track_read_register(reg::LR);
        let instr = encode_ret(reg::LR); // Return to LR (X30)
        self.push(instr);
        self
    }

    /// Generate RET instruction with specific register
    /// RET Xn
    pub fn ret_reg(&mut self, rn: Register) -> &mut Self {
        self.track_read_register(rn);
        let instr = encode_ret(rn);
        self.push(instr);
        self
    }
}

impl InstructionBuilder<Instruction> for Aarch64InstructionBuilder {
    type Register = Register;
    
    fn new() -> Self {
        Self {
            instructions: Vec::new(),
            #[cfg(feature = "register-tracking")]
            register_usage: crate::common::register_usage::RegisterUsageInfo::new(),
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
        #[cfg(feature = "register-tracking")]
        self.register_usage.clear();
    }
    
    #[cfg(feature = "register-tracking")]
    fn register_usage(&self) -> &crate::common::register_usage::RegisterUsageInfo<Self::Register> {
        &self.register_usage
    }
    
    #[cfg(feature = "register-tracking")]
    fn register_usage_mut(&mut self) -> &mut crate::common::register_usage::RegisterUsageInfo<Self::Register> {
        &mut self.register_usage
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
    /// use jit_assembler::aarch64::{reg, Aarch64InstructionBuilder};
    /// use jit_assembler::common::InstructionBuilder;
    /// 
    /// let add_func = unsafe {
    ///     Aarch64InstructionBuilder::new()
    ///         .add(reg::X0, reg::X0, reg::X1) // Add first two arguments
    ///         .ret()
    ///         .function::<fn(u64, u64) -> u64>()
    /// }.expect("Failed to create JIT function");
    /// 
    /// // Call the JIT function directly (only works on AArch64 hosts)
    /// // let result = add_func(10, 20); // Returns 30
    /// ```
    #[cfg(feature = "std")]
    unsafe fn function<F>(&self) -> Result<crate::common::jit::CallableJitFunction<F>, crate::common::jit::JitError> {
        use crate::common::InstructionCollectionExt;
        
        let bytes = self.instructions().to_bytes();
        crate::common::jit::CallableJitFunction::<F>::new(&bytes)
    }
    
    #[cfg(feature = "std")]
    unsafe fn raw_function(&self) -> Result<crate::common::jit::RawCallableJitFunction, crate::common::jit::JitError> {
        use crate::common::InstructionCollectionExt;
        
        let bytes = self.instructions().to_bytes();
        crate::common::jit::RawCallableJitFunction::new(&bytes)
    }
}