//! Register usage tracking functionality for JIT assembly.
//! 
//! This module provides utilities to track which registers are used by 
//! assembled instructions, enabling better register allocation and ABI 
//! compliance analysis.

use core::fmt;
use crate::common::{AbiClass, Register};

#[cfg(feature = "std")]
use std::collections::HashSet;
#[cfg(not(feature = "std"))]
use hashbrown::HashSet;

#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Information about register usage in a collection of instructions.
/// 
/// This structure tracks which registers are used and provides analysis
/// methods for ABI compliance and register allocation decisions.
/// 
/// # Type Parameter
/// 
/// - `R`: The register type, must implement `Register`
/// 
/// # Example
/// 
/// ```rust,ignore
/// let mut usage_info = RegisterUsageInfo::new();
/// usage_info.add_register(Register::T0);
/// usage_info.add_register(Register::S1);
/// 
/// println!("Used caller-saved: {:?}", usage_info.caller_saved_registers());
/// println!("Used callee-saved: {:?}", usage_info.callee_saved_registers());
/// println!("Needs stack frame: {}", usage_info.needs_stack_frame());
/// ```
#[derive(Debug, Clone)]
pub struct RegisterUsageInfo<R: Register> {
    /// Set of all registers used by the instructions
    used_registers: HashSet<R>,
}

impl<R: Register> RegisterUsageInfo<R> {
    /// Create a new empty register usage tracker.
    pub fn new() -> Self {
        Self {
            used_registers: HashSet::new(),
        }
    }
    
    /// Add a register to the usage tracking.
    /// 
    /// This method should be called for each register that appears in
    /// an instruction (as source or destination operand).
    pub fn add_register(&mut self, register: R) {
        self.used_registers.insert(register);
    }
    
    /// Get all used registers as a slice.
    /// 
    /// The order of registers in the returned vector is not guaranteed.
    pub fn used_registers(&self) -> Vec<R> {
        self.used_registers.iter().copied().collect()
    }
    
    /// Get all caller-saved registers that are used.
    /// 
    /// These registers don't need special preservation handling in JIT code.
    pub fn caller_saved_registers(&self) -> Vec<R> {
        self.used_registers
            .iter()
            .filter(|&reg| reg.is_caller_saved())
            .copied()
            .collect()
    }
    
    /// Get all callee-saved registers that are used.
    /// 
    /// These registers must be saved on function entry and restored on exit
    /// if used by JIT-compiled code.
    pub fn callee_saved_registers(&self) -> Vec<R> {
        self.used_registers
            .iter()
            .filter(|&reg| reg.is_callee_saved())
            .copied()
            .collect()
    }
    
    /// Get all special-purpose registers that are used.
    /// 
    /// These registers require careful handling and may indicate
    /// advanced usage patterns.
    pub fn special_registers(&self) -> Vec<R> {
        self.used_registers
            .iter()
            .filter(|&reg| reg.is_special())
            .copied()
            .collect()
    }
    
    /// Get the total number of unique registers used.
    pub fn register_count(&self) -> usize {
        self.used_registers.len()
    }
    
    /// Check if any registers are used.
    pub fn has_used_registers(&self) -> bool {
        !self.used_registers.is_empty()
    }
    
    /// Check if any callee-saved registers are used.
    /// 
    /// If this returns `true`, the JIT-compiled function needs to implement
    /// a proper function prologue/epilogue to save and restore these registers.
    pub fn needs_stack_frame(&self) -> bool {
        self.used_registers.iter().any(|reg| reg.is_callee_saved())
    }
    
    /// Get a count of registers by ABI class.
    /// 
    /// Returns a tuple of (caller_saved_count, callee_saved_count, special_count).
    pub fn count_by_abi_class(&self) -> (usize, usize, usize) {
        let mut caller_saved = 0;
        let mut callee_saved = 0;
        let mut special = 0;
        
        for register in &self.used_registers {
            match register.abi_class() {
                AbiClass::CallerSaved => caller_saved += 1,
                AbiClass::CalleeSaved => callee_saved += 1,
                AbiClass::Special => special += 1,
            }
        }
        
        (caller_saved, callee_saved, special)
    }
    
    /// Clear all register usage information.
    pub fn clear(&mut self) {
        self.used_registers.clear();
    }
    
    /// Check if a specific register is used.
    pub fn contains_register(&self, register: &R) -> bool {
        self.used_registers.contains(register)
    }
    
    /// Merge another register usage info into this one.
    /// 
    /// This is useful for combining usage information from multiple
    /// instruction sequences.
    pub fn merge(&mut self, other: &RegisterUsageInfo<R>) {
        for register in &other.used_registers {
            self.used_registers.insert(*register);
        }
    }
    
    /// Create a new register usage info by merging two existing ones.
    pub fn merged(mut self, other: &RegisterUsageInfo<R>) -> Self {
        self.merge(other);
        self
    }
}

impl<R: Register> Default for RegisterUsageInfo<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Register> fmt::Display for RegisterUsageInfo<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (caller_saved, callee_saved, special) = self.count_by_abi_class();
        
        write!(f, "RegisterUsage(total: {}, caller-saved: {}, callee-saved: {}, special: {})", 
               self.register_count(), caller_saved, callee_saved, special)?;
        
        if self.needs_stack_frame() {
            write!(f, " [needs stack frame]")?;
        }
        
        Ok(())
    }
}

/// Trait for tracking register usage in instruction builders.
/// 
/// This trait provides methods for instruction builders to track which
/// registers are used during code generation, enabling register usage
/// analysis and optimization.
pub trait RegisterUsageTracker<R: Register> {
    /// Get the current register usage information.
    /// 
    /// This returns a snapshot of all registers that have been used
    /// by instructions added to the builder.
    fn register_usage(&self) -> &RegisterUsageInfo<R>;
    
    /// Get a mutable reference to the register usage information.
    /// 
    /// This allows direct manipulation of the usage tracking, which
    /// can be useful for advanced use cases.
    fn register_usage_mut(&mut self) -> &mut RegisterUsageInfo<R>;
    
    /// Clear all register usage information.
    /// 
    /// This resets the usage tracking to an empty state, which can be
    /// useful when reusing a builder for multiple functions.
    fn clear_register_usage(&mut self) {
        self.register_usage_mut().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Test register type for unit tests
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum TestRegister {
        T0, T1,    // Caller-saved
        S0, S1,    // Callee-saved  
        SP, FP,    // Special
    }
    

    
    impl crate::common::Register for TestRegister {
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
        
        fn abi_class(&self) -> crate::common::AbiClass {
            match self {
                TestRegister::T0 | TestRegister::T1 => crate::common::AbiClass::CallerSaved,
                TestRegister::S0 | TestRegister::S1 => crate::common::AbiClass::CalleeSaved,
                TestRegister::SP | TestRegister::FP => crate::common::AbiClass::Special,
            }
        }
    }
    
    #[test]
    fn test_abi_class_display() {
        assert_eq!(crate::common::AbiClass::CallerSaved.to_string(), "caller-saved");
        assert_eq!(crate::common::AbiClass::CalleeSaved.to_string(), "callee-saved");
        assert_eq!(crate::common::AbiClass::Special.to_string(), "special");
    }
    
    #[test]
    fn test_register_trait() {
        use crate::common::{Register, AbiClass};
        
        assert!(TestRegister::T0.is_caller_saved());
        assert!(!TestRegister::T0.is_callee_saved());
        assert_eq!(TestRegister::T0.abi_class(), AbiClass::CallerSaved);
        
        assert!(!TestRegister::S0.is_caller_saved());
        assert!(TestRegister::S0.is_callee_saved());
        assert_eq!(TestRegister::S0.abi_class(), AbiClass::CalleeSaved);
        
        assert!(!TestRegister::SP.is_caller_saved());
        assert!(!TestRegister::SP.is_callee_saved());
        assert_eq!(TestRegister::SP.abi_class(), AbiClass::Special);
        assert!(TestRegister::SP.is_special());
    }
    
    #[test]
    fn test_register_usage_info() {
        let mut info = RegisterUsageInfo::new();
        assert_eq!(info.register_count(), 0);
        assert!(!info.has_used_registers());
        assert!(!info.needs_stack_frame());
        
        info.add_register(TestRegister::T0);
        info.add_register(TestRegister::T1);
        info.add_register(TestRegister::S0);
        
        assert_eq!(info.register_count(), 3);
        assert!(info.has_used_registers());
        assert!(info.needs_stack_frame());
        
        let caller_saved = info.caller_saved_registers();
        assert_eq!(caller_saved.len(), 2);
        assert!(caller_saved.contains(&TestRegister::T0));
        assert!(caller_saved.contains(&TestRegister::T1));
        
        let callee_saved = info.callee_saved_registers();
        assert_eq!(callee_saved.len(), 1);
        assert!(callee_saved.contains(&TestRegister::S0));
        
        let (caller_count, callee_count, special_count) = info.count_by_abi_class();
        assert_eq!(caller_count, 2);
        assert_eq!(callee_count, 1);
        assert_eq!(special_count, 0);
    }
    
    #[test]
    fn test_register_usage_merge() {
        let mut info1 = RegisterUsageInfo::new();
        info1.add_register(TestRegister::T0);
        info1.add_register(TestRegister::S0);
        
        let mut info2 = RegisterUsageInfo::new();
        info2.add_register(TestRegister::T1);
        info2.add_register(TestRegister::S0);  // Duplicate
        
        info1.merge(&info2);
        
        assert_eq!(info1.register_count(), 3);
        assert!(info1.contains_register(&TestRegister::T0));
        assert!(info1.contains_register(&TestRegister::T1));
        assert!(info1.contains_register(&TestRegister::S0));
    }
    
    #[test]
    fn test_register_usage_display() {
        let mut info = RegisterUsageInfo::new();
        info.add_register(TestRegister::T0);
        info.add_register(TestRegister::S0);
        info.add_register(TestRegister::SP);
        
        let display = info.to_string();
        assert!(display.contains("total: 3"));
        assert!(display.contains("caller-saved: 1"));
        assert!(display.contains("callee-saved: 1"));
        assert!(display.contains("special: 1"));
        assert!(display.contains("needs stack frame"));
    }
}