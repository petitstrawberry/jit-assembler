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

/// JIT execution functionality (std-only)
#[cfg(feature = "std")]
pub mod jit {
    use std::marker::PhantomData;
    use jit_allocator2::JitAllocator;

    /// A JIT-compiled function that can be called directly
    pub struct CallableJitFunction<F> {
        _allocator: Box<JitAllocator>,
        exec_ptr: *const u8,
        _phantom: PhantomData<F>,
    }

    impl<F> CallableJitFunction<F> {
        /// Create a new callable JIT function from instruction bytes
        pub fn new(code: &[u8]) -> Result<Self, JitError> {
            let mut allocator = JitAllocator::new(Default::default());
            let (exec_ptr, mut_ptr) = allocator.alloc(code.len()).map_err(JitError::AllocationFailed)?;
            
            unsafe {
                std::ptr::copy_nonoverlapping(code.as_ptr(), mut_ptr, code.len());
            }

            Ok(CallableJitFunction {
                _allocator: allocator,
                exec_ptr,
                _phantom: PhantomData,
            })
        }

        /// Get a function pointer to the JIT-compiled code
        pub unsafe fn as_fn(&self) -> F {
            std::mem::transmute_copy(&self.exec_ptr)
        }
    }

    // Implement direct function call for common function signatures
    impl CallableJitFunction<fn() -> u64> {
        pub fn call(&self) -> u64 {
            let func: fn() -> u64 = unsafe { self.as_fn() };
            func()
        }
    }

    impl CallableJitFunction<fn(u64) -> u64> {
        pub fn call(&self, arg0: u64) -> u64 {
            let func: fn(u64) -> u64 = unsafe { self.as_fn() };
            func(arg0)
        }
    }

    impl CallableJitFunction<fn(u64, u64) -> u64> {
        pub fn call(&self, arg0: u64, arg1: u64) -> u64 {
            let func: fn(u64, u64) -> u64 = unsafe { self.as_fn() };
            func(arg0, arg1)
        }
    }

    impl CallableJitFunction<fn(u64, u64, u64) -> u64> {
        pub fn call(&self, arg0: u64, arg1: u64, arg2: u64) -> u64 {
            let func: fn(u64, u64, u64) -> u64 = unsafe { self.as_fn() };
            func(arg0, arg1, arg2)
        }
    }

    /// Errors that can occur during JIT execution
    #[derive(Debug)]
    pub enum JitError {
        AllocationFailed(jit_allocator2::Error),
    }

    impl std::fmt::Display for JitError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                JitError::AllocationFailed(e) => write!(f, "Failed to allocate JIT memory: {:?}", e),
            }
        }
    }

    impl std::error::Error for JitError {}
}