//! Common types and traits shared across all target architectures.

use core::fmt;

/// Register usage tracking functionality
#[cfg(feature = "register-tracking")]
pub mod register_usage;

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
    
/// ABI classification for registers based on preservation requirements.
/// 
/// This simplified classification focuses on whether registers need to be
/// preserved across function calls, which is the most critical information
/// for JIT compilation and register allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbiClass {
    /// Caller-saved registers that don't need to be preserved across calls.
    /// 
    /// These registers can be freely used by JIT-compiled functions without
    /// saving/restoring their values. Examples: argument registers, temp registers.
    CallerSaved,
    
    /// Callee-saved registers that must be preserved across calls.
    /// 
    /// If a JIT-compiled function uses these registers, it must save their
    /// values on entry and restore them before returning. Examples: saved registers.
    CalleeSaved,
    
    /// Special-purpose registers with specific ABI requirements.
    /// 
    /// These registers have specific roles (stack pointer, frame pointer, zero register, etc.)
    /// and require careful handling. Generally should be avoided in JIT code
    /// unless specifically needed for their intended purpose.
    Special,
}

impl fmt::Display for AbiClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AbiClass::CallerSaved => write!(f, "caller-saved"),
            AbiClass::CalleeSaved => write!(f, "callee-saved"),
            AbiClass::Special => write!(f, "special"),
        }
    }
}

/// A register identifier for a target architecture
pub trait Register: Copy + Clone + fmt::Debug + core::hash::Hash + Eq {
    /// Get the register number/identifier
    fn id(&self) -> u32;
    
    /// Get the ABI classification for this register.
    /// 
    /// This method should return the appropriate `AbiClass` based on the
    /// target architecture's calling convention.
    /// 
    /// # Example
    /// 
    /// ```rust,ignore
    /// // For RISC-V
    /// match self {
    ///     Register::T0 | Register::T1 => AbiClass::CallerSaved,
    ///     Register::S0 | Register::S1 => AbiClass::CalleeSaved,
    ///     Register::SP | Register::FP => AbiClass::Other,
    ///     // ...
    /// }
    /// ```
    fn abi_class(&self) -> AbiClass;
    
    /// Check if this register is caller-saved.
    /// 
    /// Convenience method equivalent to `self.abi_class() == AbiClass::CallerSaved`.
    fn is_caller_saved(&self) -> bool {
        self.abi_class() == AbiClass::CallerSaved
    }
    
    /// Check if this register is callee-saved.
    /// 
    /// Convenience method equivalent to `self.abi_class() == AbiClass::CalleeSaved`.
    fn is_callee_saved(&self) -> bool {
        self.abi_class() == AbiClass::CalleeSaved
    }
    
    /// Check if this register is special-purpose.
    /// 
    /// Convenience method equivalent to `self.abi_class() == AbiClass::Special`.
    fn is_special(&self) -> bool {
        self.abi_class() == AbiClass::Special
    }
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
    /// # ABI Compatibility
    /// 
    /// While you specify a Rust function type like `fn() -> u64`, the actual JIT
    /// code uses C ABI internally for stability across Rust versions. This conversion
    /// is handled transparently by the `call()` methods.
    /// 
    /// # Limitations
    /// 
    /// Currently supports function signatures with up to 7 arguments. For functions
    /// with more arguments or complex calling conventions, use manual function pointer
    /// conversion.
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because:
    /// - It allocates executable memory
    /// - It assumes the assembled code follows the correct C ABI
    /// - The caller must ensure the function signature matches the actual code
    /// - The assembled code must be valid for the target architecture
    /// 
    #[cfg(feature = "std")]
    unsafe fn function<F>(&self) -> Result<crate::common::jit::CallableJitFunction<F>, crate::common::jit::JitError>;

    /// Create a raw JIT-compiled function for manual type conversion (std-only)
    /// 
    /// This method is similar to `function()` but returns a type-erased function
    /// that allows manual conversion to any function signature. Use this for
    /// function signatures with more than 7 arguments or custom calling conventions.
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because:
    /// - It allocates executable memory  
    /// - The caller must manually ensure type safety when calling `as_fn()`
    /// - The assembled code must be valid for the target architecture
    /// 
    #[cfg(feature = "std")]
    unsafe fn raw_function(&self) -> Result<crate::common::jit::RawCallableJitFunction, crate::common::jit::JitError>;
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

    /// A raw JIT-compiled function for manual type conversion
    /// 
    /// This is a type-erased version of `CallableJitFunction` that allows
    /// manual function pointer conversion for cases not covered by the
    /// predefined `call()` methods.
    /// 
    /// # Usage
    /// 
    /// ```rust,no_run
    /// # use jit_assembler::common::jit::*;
    /// # let code = &[0u8; 4];
    /// let raw_func = RawCallableJitFunction::new(code)?;
    /// 
    /// // Manual conversion to any function type
    /// let func: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> u64 = 
    ///     unsafe { raw_func.as_fn() };
    /// let result = func(1, 2, 3, 4, 5, 6, 7, 8);
    /// # Ok::<(), JitError>(())
    /// ```
    pub struct RawCallableJitFunction {
        _allocator: Box<JitAllocator>,
        exec_ptr: *const u8,
    }

    impl RawCallableJitFunction {
        /// Create a new raw callable JIT function from instruction bytes
        pub fn new(code: &[u8]) -> Result<Self, JitError> {
            let mut allocator = JitAllocator::new(Default::default());
            let (exec_ptr, mut_ptr) = allocator.alloc(code.len()).map_err(JitError::AllocationFailed)?;
            
            unsafe {
                std::ptr::copy_nonoverlapping(code.as_ptr(), mut_ptr, code.len());
            }

            Ok(RawCallableJitFunction {
                _allocator: allocator,
                exec_ptr,
            })
        }

        /// Convert to a function pointer of the specified type
        /// 
        /// # Safety
        /// 
        /// The caller must ensure that:
        /// - The function signature `F` matches the actual assembled code
        /// - The assembled code follows the expected calling convention (usually C ABI)
        /// - The function pointer is called with valid arguments
        /// 
        /// # Example
        /// 
        /// ```rust,no_run
        /// # use jit_assembler::common::jit::*;
        /// # let code = &[0u8; 4];
        /// let raw_func = RawCallableJitFunction::new(code)?;
        /// 
        /// // For complex signatures not covered by CallableJitFunction
        /// let func: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> u64 = 
        ///     unsafe { raw_func.as_fn() };
        /// let result = func(1, 2, 3, 4, 5, 6, 7, 8);
        /// # Ok::<(), JitError>(())
        /// ```
        pub unsafe fn as_fn<F>(&self) -> F {
            std::mem::transmute_copy(&self.exec_ptr)
        }
    }

    /// A JIT-compiled function that can be called directly
    /// 
    /// This structure wraps executable machine code and provides type-safe
    /// calling methods. While the type parameter `F` represents a Rust function
    /// signature, the actual execution uses C ABI for stability.
    /// 
    /// # Supported Signatures
    /// 
    /// Currently supports function signatures with 0-7 arguments:
    /// - `fn() -> R`
    /// - `fn(A1) -> R`  
    /// - `fn(A1, A2) -> R`
    /// - ... up to `fn(A1, A2, A3, A4, A5, A6, A7) -> R`
    /// 
    /// For unsupported signatures, use `RawCallableJitFunction` instead.
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
    }

    /// Direct call methods based on function signature
    /// 
    /// These methods provide type-safe calling with automatic ABI conversion from
    /// Rust function types to C ABI. Currently supports 0-7 arguments.
    /// 
    /// The conversion from `fn(...)` to `extern "C" fn(...)` is handled internally
    /// for ABI stability across Rust versions.
    
    impl<R> CallableJitFunction<fn() -> R> {
        /// Call with no arguments - natural syntax: func.call()
        pub fn call(&self) -> R {
            let func: extern "C" fn() -> R = unsafe { std::mem::transmute_copy(&self.exec_ptr) };
            func()
        }
    }

    impl<A1, R> CallableJitFunction<fn(A1) -> R> {
        /// Call with one argument - natural syntax: func.call(arg)
        pub fn call(&self, arg1: A1) -> R {
            let func: extern "C" fn(A1) -> R = unsafe { std::mem::transmute_copy(&self.exec_ptr) };
            func(arg1)
        }
    }

    impl<A1, A2, R> CallableJitFunction<fn(A1, A2) -> R> {
        /// Call with two arguments - natural syntax: func.call(arg1, arg2)
        pub fn call(&self, arg1: A1, arg2: A2) -> R {
            let func: extern "C" fn(A1, A2) -> R = unsafe { std::mem::transmute_copy(&self.exec_ptr) };
            func(arg1, arg2)
        }
    }

    impl<A1, A2, A3, R> CallableJitFunction<fn(A1, A2, A3) -> R> {
        /// Call with three arguments - natural syntax: func.call(arg1, arg2, arg3)
        pub fn call(&self, arg1: A1, arg2: A2, arg3: A3) -> R {
            let func: extern "C" fn(A1, A2, A3) -> R = unsafe { std::mem::transmute_copy(&self.exec_ptr) };
            func(arg1, arg2, arg3)
        }
    }

    impl<A1, A2, A3, A4, R> CallableJitFunction<fn(A1, A2, A3, A4) -> R> {
        /// Call with four arguments - natural syntax: func.call(arg1, arg2, arg3, arg4)
        pub fn call(&self, arg1: A1, arg2: A2, arg3: A3, arg4: A4) -> R {
            let func: extern "C" fn(A1, A2, A3, A4) -> R = unsafe { std::mem::transmute_copy(&self.exec_ptr) };
            func(arg1, arg2, arg3, arg4)
        }
    }

    impl<A1, A2, A3, A4, A5, R> CallableJitFunction<fn(A1, A2, A3, A4, A5) -> R> {
        /// Call with five arguments
        pub fn call(&self, arg1: A1, arg2: A2, arg3: A3, arg4: A4, arg5: A5) -> R {
            let func: extern "C" fn(A1, A2, A3, A4, A5) -> R = unsafe { std::mem::transmute_copy(&self.exec_ptr) };
            func(arg1, arg2, arg3, arg4, arg5)
        }
    }

    impl<A1, A2, A3, A4, A5, A6, R> CallableJitFunction<fn(A1, A2, A3, A4, A5, A6) -> R> {
        /// Call with six arguments
        pub fn call(&self, arg1: A1, arg2: A2, arg3: A3, arg4: A4, arg5: A5, arg6: A6) -> R {
            let func: extern "C" fn(A1, A2, A3, A4, A5, A6) -> R = unsafe { std::mem::transmute_copy(&self.exec_ptr) };
            func(arg1, arg2, arg3, arg4, arg5, arg6)
        }
    }

    impl<A1, A2, A3, A4, A5, A6, A7, R> CallableJitFunction<fn(A1, A2, A3, A4, A5, A6, A7) -> R> {
        /// Call with seven arguments
        pub fn call(&self, arg1: A1, arg2: A2, arg3: A3, arg4: A4, arg5: A5, arg6: A6, arg7: A7) -> R {
            let func: extern "C" fn(A1, A2, A3, A4, A5, A6, A7) -> R = unsafe { std::mem::transmute_copy(&self.exec_ptr) };
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