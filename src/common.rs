//! Common types and traits shared across all target architectures.

use core::fmt;

#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A machine instruction that can be encoded to bytes
pub trait Instruction: Copy + Clone + fmt::Debug + fmt::Display {
    /// Get the instruction as a 32-bit or 64-bit value
    fn value(&self) -> u64;
    
    /// Get the instruction as bytes (little-endian)
    fn bytes(&self) -> Vec<u8>;
    
    /// Get the size of this instruction in bytes
    fn size(&self) -> usize;
}

/// A register identifier for a target architecture
pub trait Register: Copy + Clone + fmt::Debug {
    /// Get the register number/identifier
    fn id(&self) -> u32;
}

/// An instruction builder for a specific architecture
pub trait InstructionBuilder<I: Instruction> {
    /// Create a new instruction builder
    fn new() -> Self;
    
    /// Get the generated instructions
    fn instructions(&self) -> &[I];
    
    /// Add an instruction to the builder
    fn push(&mut self, instr: I);
    
    /// Clear all instructions
    fn clear(&mut self);
}

/// Architecture-specific encoding functions
pub trait ArchitectureEncoder<I: Instruction> {
    /// Encode an instruction with specific opcode and operands
    fn encode(&self, opcode: u32, operands: &[u32]) -> I;
}

/// Common result type for instruction building
pub type BuildResult<I> = Result<I, BuildError>;

/// Errors that can occur during instruction building
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildError {
    /// Invalid register identifier
    InvalidRegister(u32),
    /// Invalid immediate value
    InvalidImmediate(i64),
    /// Unsupported instruction
    UnsupportedInstruction,
    /// Invalid operand combination
    InvalidOperands,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::InvalidRegister(id) => write!(f, "Invalid register ID: {}", id),
            BuildError::InvalidImmediate(val) => write!(f, "Invalid immediate value: {}", val),
            BuildError::UnsupportedInstruction => write!(f, "Unsupported instruction"),
            BuildError::InvalidOperands => write!(f, "Invalid operand combination"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BuildError {}