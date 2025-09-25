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
    fn instructions(&self) -> InstructionCollection<I>;
    
    /// Add an instruction to the builder
    fn push(&mut self, instr: I);
    
    /// Clear all instructions
    fn clear(&mut self);
    
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
    #[cfg(feature = "std")]
    unsafe fn function<F>(&self) -> Result<crate::common::jit::CallableJitFunction<F>, crate::common::jit::JitError>;
}

/// Convenience functions for instruction collections
/// These functions work with any collection of instructions
pub fn instructions_to_bytes<I: Instruction>(instructions: &[I]) -> Vec<u8> {
    let mut result = Vec::new();
    for instr in instructions {
        result.extend_from_slice(&instr.bytes());
    }
    result
}

/// Get the total size in bytes of a collection of instructions
pub fn instructions_total_size<I: Instruction>(instructions: &[I]) -> usize {
    instructions.iter().map(|i| i.size()).sum()
}

/// A collection of instructions with convenient methods for byte manipulation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstructionCollection<I: Instruction> {
    instructions: Vec<I>,
}

impl<I: Instruction> InstructionCollection<I> {
    /// Create a new empty instruction collection
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }
    
    /// Create from a vector of instructions
    pub fn from_vec(instructions: Vec<I>) -> Self {
        Self { instructions }
    }
    
    /// Create from a slice of instructions
    pub fn from_slice(instructions: &[I]) -> Self {
        Self {
            instructions: instructions.to_vec(),
        }
    }
    
    /// Get the instructions as a slice
    pub fn as_slice(&self) -> &[I] {
        &self.instructions
    }
    
    /// Get the instructions as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [I] {
        &mut self.instructions
    }
    
    /// Convert instructions to a single byte vector
    pub fn to_bytes(&self) -> Vec<u8> {
        instructions_to_bytes(&self.instructions)
    }
    
    /// Get the total size in bytes of all instructions
    pub fn total_size(&self) -> usize {
        instructions_total_size(&self.instructions)
    }
    
    /// Get the number of instructions
    pub fn len(&self) -> usize {
        self.instructions.len()
    }
    
    /// Check if the collection is empty
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
    
    /// Add an instruction to the collection
    pub fn push(&mut self, instruction: I) {
        self.instructions.push(instruction);
    }
    
    /// Remove all instructions
    pub fn clear(&mut self) {
        self.instructions.clear();
    }
    
    /// Get an iterator over the instructions
    pub fn iter(&self) -> core::slice::Iter<'_, I> {
        self.instructions.iter()
    }
    
    /// Get a mutable iterator over the instructions
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, I> {
        self.instructions.iter_mut()
    }
    
    /// Convert to a Vec (consumes self)
    pub fn into_vec(self) -> Vec<I> {
        self.instructions
    }
    
    /// Create a Vec by cloning the instructions
    pub fn to_vec(&self) -> Vec<I> {
        self.instructions.clone()
    }
    
    /// Get a reference to the instruction at the given index
    pub fn get(&self, index: usize) -> Option<&I> {
        self.instructions.get(index)
    }
    
    /// Get a mutable reference to the instruction at the given index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut I> {
        self.instructions.get_mut(index)
    }
}

impl<I: Instruction> Default for InstructionCollection<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Instruction> From<Vec<I>> for InstructionCollection<I> {
    fn from(instructions: Vec<I>) -> Self {
        Self::from_vec(instructions)
    }
}

impl<I: Instruction> From<&[I]> for InstructionCollection<I> {
    fn from(instructions: &[I]) -> Self {
        Self::from_slice(instructions)
    }
}

impl<I: Instruction> AsRef<[I]> for InstructionCollection<I> {
    fn as_ref(&self) -> &[I] {
        &self.instructions
    }
}

impl<I: Instruction> AsMut<[I]> for InstructionCollection<I> {
    fn as_mut(&mut self) -> &mut [I] {
        &mut self.instructions
    }
}

impl<I: Instruction> core::ops::Index<usize> for InstructionCollection<I> {
    type Output = I;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.instructions[index]
    }
}

impl<I: Instruction> core::ops::IndexMut<usize> for InstructionCollection<I> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.instructions[index]
    }
}

impl<I: Instruction> IntoIterator for InstructionCollection<I> {
    type Item = I;
    type IntoIter = <Vec<I> as IntoIterator>::IntoIter;
    
    fn into_iter(self) -> Self::IntoIter {
        self.instructions.into_iter()
    }
}

impl<'a, I: Instruction> IntoIterator for &'a InstructionCollection<I> {
    type Item = &'a I;
    type IntoIter = core::slice::Iter<'a, I>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.instructions.iter()
    }
}

impl<'a, I: Instruction> IntoIterator for &'a mut InstructionCollection<I> {
    type Item = &'a mut I;
    type IntoIter = core::slice::IterMut<'a, I>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.instructions.iter_mut()
    }
}

impl<I: Instruction> core::ops::Deref for InstructionCollection<I> {
    type Target = [I];
    
    fn deref(&self) -> &Self::Target {
        &self.instructions
    }
}

impl<I: Instruction> core::ops::DerefMut for InstructionCollection<I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.instructions
    }
}

/// Trait extension for instruction collections
/// This allows you to call `.to_bytes()` and `.total_size()` directly on slices and vectors
pub trait InstructionCollectionExt<I: Instruction> {
    /// Convert instructions to a single byte vector
    fn to_bytes(&self) -> Vec<u8>;
    
    /// Get the total size in bytes of all instructions
    fn total_size(&self) -> usize;
}

impl<I: Instruction> InstructionCollectionExt<I> for [I] {
    fn to_bytes(&self) -> Vec<u8> {
        instructions_to_bytes(self)
    }
    
    fn total_size(&self) -> usize {
        instructions_total_size(self)
    }
}

impl<I: Instruction> InstructionCollectionExt<I> for Vec<I> {
    fn to_bytes(&self) -> Vec<u8> {
        instructions_to_bytes(self)
    }
    
    fn total_size(&self) -> usize {
        instructions_total_size(self)
    }
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

    /// Direct call methods based on function signature - the ultimate solution!
    /// These override the generic call method with signature-specific versions
    
    impl<R> CallableJitFunction<fn() -> R> {
        /// Call with no arguments - natural syntax: func.call()
        pub fn call(&self) -> R {
            let func: fn() -> R = unsafe { self.as_fn() };
            func()
        }
    }

    impl<A1, R> CallableJitFunction<fn(A1) -> R> {
        /// Call with one argument - natural syntax: func.call(arg)
        pub fn call(&self, arg1: A1) -> R {
            let func: fn(A1) -> R = unsafe { self.as_fn() };
            func(arg1)
        }
    }

    impl<A1, A2, R> CallableJitFunction<fn(A1, A2) -> R> {
        /// Call with two arguments - natural syntax: func.call(arg1, arg2)
        pub fn call(&self, arg1: A1, arg2: A2) -> R {
            let func: fn(A1, A2) -> R = unsafe { self.as_fn() };
            func(arg1, arg2)
        }
    }

    impl<A1, A2, A3, R> CallableJitFunction<fn(A1, A2, A3) -> R> {
        /// Call with three arguments - natural syntax: func.call(arg1, arg2, arg3)
        pub fn call(&self, arg1: A1, arg2: A2, arg3: A3) -> R {
            let func: fn(A1, A2, A3) -> R = unsafe { self.as_fn() };
            func(arg1, arg2, arg3)
        }
    }

    impl<A1, A2, A3, A4, R> CallableJitFunction<fn(A1, A2, A3, A4) -> R> {
        /// Call with four arguments - natural syntax: func.call(arg1, arg2, arg3, arg4)
        pub fn call(&self, arg1: A1, arg2: A2, arg3: A3, arg4: A4) -> R {
            let func: fn(A1, A2, A3, A4) -> R = unsafe { self.as_fn() };
            func(arg1, arg2, arg3, arg4)
        }
    }

    impl<A1, A2, A3, A4, A5, R> CallableJitFunction<fn(A1, A2, A3, A4, A5) -> R> {
        /// Call with five arguments
        pub fn call(&self, arg1: A1, arg2: A2, arg3: A3, arg4: A4, arg5: A5) -> R {
            let func: fn(A1, A2, A3, A4, A5) -> R = unsafe { self.as_fn() };
            func(arg1, arg2, arg3, arg4, arg5)
        }
    }

    impl<A1, A2, A3, A4, A5, A6, R> CallableJitFunction<fn(A1, A2, A3, A4, A5, A6) -> R> {
        /// Call with six arguments
        pub fn call(&self, arg1: A1, arg2: A2, arg3: A3, arg4: A4, arg5: A5, arg6: A6) -> R {
            let func: fn(A1, A2, A3, A4, A5, A6) -> R = unsafe { self.as_fn() };
            func(arg1, arg2, arg3, arg4, arg5, arg6)
        }
    }

    impl<A1, A2, A3, A4, A5, A6, A7, R> CallableJitFunction<fn(A1, A2, A3, A4, A5, A6, A7) -> R> {
        /// Call with seven arguments
        pub fn call(&self, arg1: A1, arg2: A2, arg3: A3, arg4: A4, arg5: A5, arg6: A6, arg7: A7) -> R {
            let func: fn(A1, A2, A3, A4, A5, A6, A7) -> R = unsafe { self.as_fn() };
            func(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        }
    }

    // Note: For void return functions, we don't generate them here 
    // as they would need special handling with unit type ()

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