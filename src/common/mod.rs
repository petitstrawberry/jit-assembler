//! Common types and traits shared across all target architectures.

use core::fmt;

/// Register usage tracking functionality
#[cfg(feature = "register-tracking")]
pub mod register_usage;

#[cfg(feature = "register-tracking")]
use register_usage::RegisterUsageInfo;

#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A collection of instructions with associated register usage information.
/// 
/// This struct combines an `InstructionCollection` with `RegisterUsageInfo`,
/// allowing you to merge both instructions and their usage statistics together.
/// This is useful when combining code sequences from multiple builders while
/// preserving register usage information.
/// 
/// # Example
/// 
/// ```rust,ignore
/// // Create three builders for different parts of the code
/// let mut prologue = Riscv64InstructionBuilder::new();
/// prologue.sd(reg::SP, reg::S0, -8);  // Save S0
/// 
/// let mut main = Riscv64InstructionBuilder::new();
/// main.add(reg::S0, reg::A0, reg::A1);  // Main computation
/// 
/// let mut epilogue = Riscv64InstructionBuilder::new();
/// epilogue.ld(reg::S0, reg::SP, -8);  // Restore S0
/// 
/// // Create tracked collections
/// let prologue_tracked = InstructionCollectionWithUsage::new(
///     prologue.instructions(),
///     prologue.register_usage().clone()
/// );
/// let main_tracked = InstructionCollectionWithUsage::new(
///     main.instructions(),
///     main.register_usage().clone()
/// );
/// let epilogue_tracked = InstructionCollectionWithUsage::new(
///     epilogue.instructions(),
///     epilogue.register_usage().clone()
/// );
/// 
/// // Merge them: prologue + main + epilogue
/// let combined = prologue_tracked + main_tracked + epilogue_tracked;
/// 
/// // Now we have the complete function with accurate register usage
/// let instructions = combined.instructions();
/// let usage = combined.register_usage();
/// ```
#[cfg(feature = "register-tracking")]
#[derive(Debug, Clone)]
pub struct InstructionCollectionWithUsage<I: Instruction, R: Register> {
    instructions: InstructionCollection<I>,
    register_usage: RegisterUsageInfo<R>,
}

#[cfg(feature = "register-tracking")]
impl<I: Instruction, R: Register> InstructionCollectionWithUsage<I, R> {
    /// Create a new tracked instruction collection.
    pub fn new(instructions: InstructionCollection<I>, register_usage: RegisterUsageInfo<R>) -> Self {
        Self {
            instructions,
            register_usage,
        }
    }
    
    /// Create from raw parts.
    pub fn from_parts(instructions: InstructionCollection<I>, register_usage: RegisterUsageInfo<R>) -> Self {
        Self::new(instructions, register_usage)
    }
    
    /// Get a reference to the instructions.
    pub fn instructions(&self) -> &InstructionCollection<I> {
        &self.instructions
    }
    
    /// Get a mutable reference to the instructions.
    pub fn instructions_mut(&mut self) -> &mut InstructionCollection<I> {
        &mut self.instructions
    }
    
    /// Get a reference to the register usage information.
    pub fn register_usage(&self) -> &RegisterUsageInfo<R> {
        &self.register_usage
    }
    
    /// Get a mutable reference to the register usage information.
    pub fn register_usage_mut(&mut self) -> &mut RegisterUsageInfo<R> {
        &mut self.register_usage
    }
    
    /// Consume this collection and return the instructions and register usage.
    pub fn into_parts(self) -> (InstructionCollection<I>, RegisterUsageInfo<R>) {
        (self.instructions, self.register_usage)
    }
    
    /// Consume this collection and return just the instructions.
    pub fn into_instructions(self) -> InstructionCollection<I> {
        self.instructions
    }
    
    /// Merge another tracked collection into this one.
    /// 
    /// This appends the instructions and merges the register usage information.
    pub fn append(&mut self, other: InstructionCollectionWithUsage<I, R>) {
        self.instructions.append(other.instructions);
        self.register_usage.merge(&other.register_usage);
    }
    
    /// Extend this collection with instructions and register usage from another.
    /// 
    /// This clones instructions and merges the register usage information.
    pub fn extend_from(&mut self, other: &InstructionCollectionWithUsage<I, R>) {
        self.instructions.extend_from_collection(&other.instructions);
        self.register_usage.merge(&other.register_usage);
    }
    
    /// Concatenate two tracked collections, consuming both.
    pub fn concat(mut self, other: InstructionCollectionWithUsage<I, R>) -> Self {
        self.instructions.append(other.instructions);
        self.register_usage.merge(&other.register_usage);
        self
    }
}

#[cfg(feature = "register-tracking")]
impl<I: Instruction, R: Register> core::ops::Add for InstructionCollectionWithUsage<I, R> {
    type Output = InstructionCollectionWithUsage<I, R>;
    
    /// Concatenate two tracked instruction collections using the `+` operator.
    fn add(self, other: InstructionCollectionWithUsage<I, R>) -> InstructionCollectionWithUsage<I, R> {
        self.concat(other)
    }
}

#[cfg(feature = "register-tracking")]
impl<I: Instruction, R: Register> core::ops::AddAssign for InstructionCollectionWithUsage<I, R> {
    /// Append another tracked instruction collection using the `+=` operator.
    fn add_assign(&mut self, other: InstructionCollectionWithUsage<I, R>) {
        self.append(other);
    }
}

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
    /// The register type used by this architecture
    type Register: Register;
    
    /// Create a new instruction builder
    fn new() -> Self;
    
    /// Get the generated instructions
    fn instructions(&self) -> InstructionCollection<I>;
    
    /// Add an instruction to the builder
    fn push(&mut self, instr: I);
    
    /// Clear all instructions
    fn clear(&mut self);
    
    /// Get register usage information (register-tracking feature only)
    /// 
    /// This method returns information about which registers have been used
    /// by the instructions in this builder, enabling register allocation
    /// analysis and ABI compliance checking.
    #[cfg(feature = "register-tracking")]
    fn register_usage(&self) -> &crate::common::register_usage::RegisterUsageInfo<Self::Register>;
    
    /// Get mutable register usage information (register-tracking feature only)
    /// 
    /// This allows direct manipulation of the usage tracking, which can be
    /// useful for advanced use cases or manual register tracking.
    #[cfg(feature = "register-tracking")]
    fn register_usage_mut(&mut self) -> &mut crate::common::register_usage::RegisterUsageInfo<Self::Register>;
    
    /// Clear register usage information (register-tracking feature only)
    /// 
    /// This resets the usage tracking to an empty state, which can be
    /// useful when reusing a builder for multiple functions.
    #[cfg(feature = "register-tracking")]
    fn clear_register_usage(&mut self) {
        self.register_usage_mut().clear();
    }
    
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
    
    /// Append another instruction collection to this one (consumes other).
    /// 
    /// This moves all instructions from `other` into this collection.
    /// 
    /// # Example
    /// 
    /// ```rust,ignore
    /// let mut collection1 = InstructionCollection::from_slice(&[instr1, instr2]);
    /// let collection2 = InstructionCollection::from_slice(&[instr3, instr4]);
    /// collection1.append(collection2);
    /// // collection1 now contains [instr1, instr2, instr3, instr4]
    /// ```
    pub fn append(&mut self, mut other: InstructionCollection<I>) {
        self.instructions.append(&mut other.instructions);
    }
    
    /// Extend this collection with instructions from another collection.
    /// 
    /// This clones instructions from `other` into this collection.
    /// 
    /// # Example
    /// 
    /// ```rust,ignore
    /// let mut collection1 = InstructionCollection::from_slice(&[instr1, instr2]);
    /// let collection2 = InstructionCollection::from_slice(&[instr3, instr4]);
    /// collection1.extend_from_collection(&collection2);
    /// // collection1 now contains [instr1, instr2, instr3, instr4]
    /// // collection2 is still valid
    /// ```
    pub fn extend_from_collection(&mut self, other: &InstructionCollection<I>) {
        self.instructions.extend_from_slice(&other.instructions);
    }
    
    /// Concatenate two instruction collections, consuming both.
    /// 
    /// Creates a new collection containing all instructions from `self` 
    /// followed by all instructions from `other`.
    /// 
    /// # Example
    /// 
    /// ```rust,ignore
    /// let collection1 = InstructionCollection::from_slice(&[instr1, instr2]);
    /// let collection2 = InstructionCollection::from_slice(&[instr3, instr4]);
    /// let combined = collection1.concat(collection2);
    /// // combined contains [instr1, instr2, instr3, instr4]
    /// ```
    pub fn concat(mut self, mut other: InstructionCollection<I>) -> Self {
        self.instructions.append(&mut other.instructions);
        self
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

impl<I: Instruction> core::ops::Add for InstructionCollection<I> {
    type Output = InstructionCollection<I>;
    
    /// Concatenate two instruction collections using the `+` operator.
    /// 
    /// # Example
    /// 
    /// ```rust,ignore
    /// let collection1 = InstructionCollection::from_slice(&[instr1, instr2]);
    /// let collection2 = InstructionCollection::from_slice(&[instr3, instr4]);
    /// let combined = collection1 + collection2;
    /// // combined contains [instr1, instr2, instr3, instr4]
    /// ```
    fn add(self, other: InstructionCollection<I>) -> InstructionCollection<I> {
        self.concat(other)
    }
}

impl<I: Instruction> core::ops::AddAssign for InstructionCollection<I> {
    /// Append another instruction collection to this one using the `+=` operator.
    /// 
    /// # Example
    /// 
    /// ```rust,ignore
    /// let mut collection1 = InstructionCollection::from_slice(&[instr1, instr2]);
    /// let collection2 = InstructionCollection::from_slice(&[instr3, instr4]);
    /// collection1 += collection2;
    /// // collection1 now contains [instr1, instr2, instr3, instr4]
    /// ```
    fn add_assign(&mut self, other: InstructionCollection<I>) {
        self.append(other);
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

#[cfg(test)]
mod tests {
    use super::*;
    
    // Test instruction type for unit tests
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TestInstruction(u32);
    
    impl fmt::Display for TestInstruction {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TestInstruction(0x{:08x})", self.0)
        }
    }
    
    impl Instruction for TestInstruction {
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
    
    #[test]
    fn test_instruction_collection_append() {
        let mut collection1 = InstructionCollection::from_slice(&[
            TestInstruction(1),
            TestInstruction(2),
        ]);
        let collection2 = InstructionCollection::from_slice(&[
            TestInstruction(3),
            TestInstruction(4),
        ]);
        
        collection1.append(collection2);
        
        assert_eq!(collection1.len(), 4);
        assert_eq!(collection1[0], TestInstruction(1));
        assert_eq!(collection1[1], TestInstruction(2));
        assert_eq!(collection1[2], TestInstruction(3));
        assert_eq!(collection1[3], TestInstruction(4));
    }
    
    #[test]
    fn test_instruction_collection_extend_from_collection() {
        let mut collection1 = InstructionCollection::from_slice(&[
            TestInstruction(1),
            TestInstruction(2),
        ]);
        let collection2 = InstructionCollection::from_slice(&[
            TestInstruction(3),
            TestInstruction(4),
        ]);
        
        collection1.extend_from_collection(&collection2);
        
        // collection1 should have all 4 instructions
        assert_eq!(collection1.len(), 4);
        assert_eq!(collection1[0], TestInstruction(1));
        assert_eq!(collection1[1], TestInstruction(2));
        assert_eq!(collection1[2], TestInstruction(3));
        assert_eq!(collection1[3], TestInstruction(4));
        
        // collection2 should still be valid
        assert_eq!(collection2.len(), 2);
        assert_eq!(collection2[0], TestInstruction(3));
        assert_eq!(collection2[1], TestInstruction(4));
    }
    
    #[test]
    fn test_instruction_collection_concat() {
        let collection1 = InstructionCollection::from_slice(&[
            TestInstruction(1),
            TestInstruction(2),
        ]);
        let collection2 = InstructionCollection::from_slice(&[
            TestInstruction(3),
            TestInstruction(4),
        ]);
        
        let combined = collection1.concat(collection2);
        
        assert_eq!(combined.len(), 4);
        assert_eq!(combined[0], TestInstruction(1));
        assert_eq!(combined[1], TestInstruction(2));
        assert_eq!(combined[2], TestInstruction(3));
        assert_eq!(combined[3], TestInstruction(4));
    }
    
    #[test]
    fn test_instruction_collection_add_operator() {
        let collection1 = InstructionCollection::from_slice(&[
            TestInstruction(1),
            TestInstruction(2),
        ]);
        let collection2 = InstructionCollection::from_slice(&[
            TestInstruction(3),
            TestInstruction(4),
        ]);
        
        let combined = collection1 + collection2;
        
        assert_eq!(combined.len(), 4);
        assert_eq!(combined[0], TestInstruction(1));
        assert_eq!(combined[1], TestInstruction(2));
        assert_eq!(combined[2], TestInstruction(3));
        assert_eq!(combined[3], TestInstruction(4));
    }
    
    #[test]
    fn test_instruction_collection_add_assign_operator() {
        let mut collection1 = InstructionCollection::from_slice(&[
            TestInstruction(1),
            TestInstruction(2),
        ]);
        let collection2 = InstructionCollection::from_slice(&[
            TestInstruction(3),
            TestInstruction(4),
        ]);
        
        collection1 += collection2;
        
        assert_eq!(collection1.len(), 4);
        assert_eq!(collection1[0], TestInstruction(1));
        assert_eq!(collection1[1], TestInstruction(2));
        assert_eq!(collection1[2], TestInstruction(3));
        assert_eq!(collection1[3], TestInstruction(4));
    }
    
    #[test]
    fn test_instruction_collection_multiple_merge() {
        let collection1 = InstructionCollection::from_slice(&[TestInstruction(1)]);
        let collection2 = InstructionCollection::from_slice(&[TestInstruction(2)]);
        let collection3 = InstructionCollection::from_slice(&[TestInstruction(3)]);
        
        // Test chaining with + operator
        let combined = collection1 + collection2 + collection3;
        
        assert_eq!(combined.len(), 3);
        assert_eq!(combined[0], TestInstruction(1));
        assert_eq!(combined[1], TestInstruction(2));
        assert_eq!(combined[2], TestInstruction(3));
    }
    
    #[test]
    fn test_instruction_collection_merge_empty() {
        let mut collection1 = InstructionCollection::from_slice(&[
            TestInstruction(1),
            TestInstruction(2),
        ]);
        let collection2 = InstructionCollection::<TestInstruction>::new();
        
        collection1.append(collection2);
        
        assert_eq!(collection1.len(), 2);
        assert_eq!(collection1[0], TestInstruction(1));
        assert_eq!(collection1[1], TestInstruction(2));
    }
    
    #[test]
    fn test_instruction_collection_merge_into_empty() {
        let mut collection1 = InstructionCollection::<TestInstruction>::new();
        let collection2 = InstructionCollection::from_slice(&[
            TestInstruction(3),
            TestInstruction(4),
        ]);
        
        collection1.append(collection2);
        
        assert_eq!(collection1.len(), 2);
        assert_eq!(collection1[0], TestInstruction(3));
        assert_eq!(collection1[1], TestInstruction(4));
    }
}

#[cfg(all(test, feature = "register-tracking"))]
mod register_tracking_tests {
    use super::*;
    
    // Test instruction and register types for unit tests
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TestInstruction(u32);
    
    impl fmt::Display for TestInstruction {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TestInstruction(0x{:08x})", self.0)
        }
    }
    
    impl Instruction for TestInstruction {
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
    
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum TestRegister {
        T0, T1,    // Caller-saved
        S0, S1,    // Callee-saved
        SP, FP,    // Special
    }
    
    impl Register for TestRegister {
        fn id(&self) -> u32 {
            match self {
                TestRegister::T0 => 0,
                TestRegister::T1 => 1,
                TestRegister::S0 => 2,
                TestRegister::S1 => 3,
                TestRegister::SP => 4,
                TestRegister::FP => 5,
            }
        }
        
        fn abi_class(&self) -> AbiClass {
            match self {
                TestRegister::T0 | TestRegister::T1 => AbiClass::CallerSaved,
                TestRegister::S0 | TestRegister::S1 => AbiClass::CalleeSaved,
                TestRegister::SP | TestRegister::FP => AbiClass::Special,
            }
        }
    }
    
    #[test]
    fn test_instruction_collection_with_usage_new() {
        let instructions = InstructionCollection::from_slice(&[
            TestInstruction(1),
            TestInstruction(2),
        ]);
        let mut usage = register_usage::RegisterUsageInfo::new();
        usage.add_written_register(TestRegister::T0);
        usage.add_read_register(TestRegister::T1);
        
        let tracked = InstructionCollectionWithUsage::new(instructions.clone(), usage.clone());
        
        assert_eq!(tracked.instructions().len(), 2);
        assert_eq!(tracked.register_usage().register_count(), 2);
    }
    
    #[test]
    fn test_instruction_collection_with_usage_append() {
        let instructions1 = InstructionCollection::from_slice(&[TestInstruction(1)]);
        let mut usage1 = register_usage::RegisterUsageInfo::new();
        usage1.add_written_register(TestRegister::T0);
        
        let instructions2 = InstructionCollection::from_slice(&[TestInstruction(2)]);
        let mut usage2 = register_usage::RegisterUsageInfo::new();
        usage2.add_read_register(TestRegister::S0);
        
        let mut tracked1 = InstructionCollectionWithUsage::new(instructions1, usage1);
        let tracked2 = InstructionCollectionWithUsage::new(instructions2, usage2);
        
        tracked1.append(tracked2);
        
        assert_eq!(tracked1.instructions().len(), 2);
        assert_eq!(tracked1.register_usage().register_count(), 2);
        assert!(tracked1.register_usage().contains_written_register(&TestRegister::T0));
        assert!(tracked1.register_usage().contains_read_register(&TestRegister::S0));
    }
    
    #[test]
    fn test_instruction_collection_with_usage_concat() {
        let instructions1 = InstructionCollection::from_slice(&[TestInstruction(1)]);
        let mut usage1 = register_usage::RegisterUsageInfo::new();
        usage1.add_written_register(TestRegister::T0);
        
        let instructions2 = InstructionCollection::from_slice(&[TestInstruction(2)]);
        let mut usage2 = register_usage::RegisterUsageInfo::new();
        usage2.add_read_register(TestRegister::S0);
        
        let tracked1 = InstructionCollectionWithUsage::new(instructions1, usage1);
        let tracked2 = InstructionCollectionWithUsage::new(instructions2, usage2);
        
        let combined = tracked1.concat(tracked2);
        
        assert_eq!(combined.instructions().len(), 2);
        assert_eq!(combined.register_usage().register_count(), 2);
        assert!(combined.register_usage().contains_written_register(&TestRegister::T0));
        assert!(combined.register_usage().contains_read_register(&TestRegister::S0));
    }
    
    #[test]
    fn test_instruction_collection_with_usage_add_operator() {
        let instructions1 = InstructionCollection::from_slice(&[TestInstruction(1)]);
        let mut usage1 = register_usage::RegisterUsageInfo::new();
        usage1.add_written_register(TestRegister::T0);
        
        let instructions2 = InstructionCollection::from_slice(&[TestInstruction(2)]);
        let mut usage2 = register_usage::RegisterUsageInfo::new();
        usage2.add_read_register(TestRegister::S0);
        
        let tracked1 = InstructionCollectionWithUsage::new(instructions1, usage1);
        let tracked2 = InstructionCollectionWithUsage::new(instructions2, usage2);
        
        let combined = tracked1 + tracked2;
        
        assert_eq!(combined.instructions().len(), 2);
        assert_eq!(combined.register_usage().register_count(), 2);
        assert!(combined.register_usage().contains_written_register(&TestRegister::T0));
        assert!(combined.register_usage().contains_read_register(&TestRegister::S0));
    }
    
    #[test]
    fn test_instruction_collection_with_usage_multiple_merge() {
        // Simulate the use case from the issue: prologue + main + epilogue
        
        // Prologue: save registers
        let prologue_instrs = InstructionCollection::from_slice(&[TestInstruction(0x10)]);
        let mut prologue_usage = register_usage::RegisterUsageInfo::new();
        prologue_usage.add_read_register(TestRegister::SP);
        prologue_usage.add_read_register(TestRegister::S0);
        
        // Main: actual computation
        let main_instrs = InstructionCollection::from_slice(&[
            TestInstruction(0x20),
            TestInstruction(0x21),
        ]);
        let mut main_usage = register_usage::RegisterUsageInfo::new();
        main_usage.add_written_register(TestRegister::T0);
        main_usage.add_written_register(TestRegister::S0);
        main_usage.add_read_register(TestRegister::T1);
        
        // Epilogue: restore registers
        let epilogue_instrs = InstructionCollection::from_slice(&[TestInstruction(0x30)]);
        let mut epilogue_usage = register_usage::RegisterUsageInfo::new();
        epilogue_usage.add_read_register(TestRegister::SP);
        epilogue_usage.add_written_register(TestRegister::S0);
        
        let prologue = InstructionCollectionWithUsage::new(prologue_instrs, prologue_usage);
        let main = InstructionCollectionWithUsage::new(main_instrs, main_usage);
        let epilogue = InstructionCollectionWithUsage::new(epilogue_instrs, epilogue_usage);
        
        // Merge: prologue + main + epilogue
        let combined = prologue + main + epilogue;
        
        // Check instructions
        assert_eq!(combined.instructions().len(), 4);
        assert_eq!(combined.instructions()[0], TestInstruction(0x10));
        assert_eq!(combined.instructions()[1], TestInstruction(0x20));
        assert_eq!(combined.instructions()[2], TestInstruction(0x21));
        assert_eq!(combined.instructions()[3], TestInstruction(0x30));
        
        // Check register usage is properly merged
        let usage = combined.register_usage();
        assert_eq!(usage.register_count(), 4);  // T0, T1, S0, SP
        assert!(usage.contains_register(&TestRegister::T0));
        assert!(usage.contains_register(&TestRegister::T1));
        assert!(usage.contains_register(&TestRegister::S0));
        assert!(usage.contains_register(&TestRegister::SP));
        
        // Check that S0 appears as both written and read
        assert!(usage.contains_written_register(&TestRegister::S0));
        assert!(usage.contains_read_register(&TestRegister::S0));
    }
    
    #[test]
    fn test_instruction_collection_with_usage_into_parts() {
        let instructions = InstructionCollection::from_slice(&[TestInstruction(1)]);
        let mut usage = register_usage::RegisterUsageInfo::new();
        usage.add_written_register(TestRegister::T0);
        
        let tracked = InstructionCollectionWithUsage::new(instructions.clone(), usage.clone());
        let (instrs, reg_usage) = tracked.into_parts();
        
        assert_eq!(instrs.len(), 1);
        assert_eq!(reg_usage.register_count(), 1);
    }
    
    #[test]
    fn test_instruction_collection_with_usage_extend_from() {
        let instructions1 = InstructionCollection::from_slice(&[TestInstruction(1)]);
        let mut usage1 = register_usage::RegisterUsageInfo::new();
        usage1.add_written_register(TestRegister::T0);
        
        let instructions2 = InstructionCollection::from_slice(&[TestInstruction(2)]);
        let mut usage2 = register_usage::RegisterUsageInfo::new();
        usage2.add_read_register(TestRegister::S0);
        
        let mut tracked1 = InstructionCollectionWithUsage::new(instructions1, usage1);
        let tracked2 = InstructionCollectionWithUsage::new(instructions2, usage2);
        
        tracked1.extend_from(&tracked2);
        
        // tracked1 should be extended
        assert_eq!(tracked1.instructions().len(), 2);
        assert_eq!(tracked1.register_usage().register_count(), 2);
        
        // tracked2 should still exist
        assert_eq!(tracked2.instructions().len(), 1);
        assert_eq!(tracked2.register_usage().register_count(), 1);
    }
}