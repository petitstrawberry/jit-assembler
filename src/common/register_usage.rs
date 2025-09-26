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
/// usage_info.add_written_register(Register::T0);
/// usage_info.add_read_register(Register::T1);
/// 
/// println!("Written (def): {:?}", usage_info.written_registers());
/// println!("Read (use): {:?}", usage_info.read_registers());
/// println!("Needs stack frame: {}", usage_info.needs_stack_frame());
/// ```
#[derive(Debug, Clone)]
pub struct RegisterUsageInfo<R: Register> {
    /// Set of registers that are written to (def)
    written_registers: HashSet<R>,
    /// Set of registers that are read from (use)
    read_registers: HashSet<R>,
}

impl<R: Register> RegisterUsageInfo<R> {
    /// Create a new empty register usage tracker.
    pub fn new() -> Self {
        Self {
            written_registers: HashSet::new(),
            read_registers: HashSet::new(),
        }
    }
    
    /// Add a register that is written to (destination register).
    /// 
    /// This should be called for rd registers that receive new values.
    pub fn add_written_register(&mut self, register: R) {
        self.written_registers.insert(register);
    }
    
    /// Add a register that is read from (source register).
    /// 
    /// This should be called for rs1, rs2 registers that provide input values.
    pub fn add_read_register(&mut self, register: R) {
        self.read_registers.insert(register);
    }
    
    /// Get all written registers (def).
    /// 
    /// These are registers that receive new values and may need to be
    /// preserved if they are callee-saved.
    pub fn written_registers(&self) -> Vec<R> {
        self.written_registers.iter().copied().collect()
    }
    
    /// Get all read registers (use).
    /// 
    /// These are registers that provide input values.
    pub fn read_registers(&self) -> Vec<R> {
        self.read_registers.iter().copied().collect()
    }
    
    /// Get all used registers (def ∪ use).
    /// 
    /// This returns the union of written and read registers.
    /// The order of registers in the returned vector is not guaranteed.
    pub fn used_registers(&self) -> Vec<R> {
        let mut all_registers: HashSet<R> = self.written_registers.clone();
        all_registers.extend(&self.read_registers);
        all_registers.iter().copied().collect()
    }
    
    /// Get all caller-saved registers that are written to.
    /// 
    /// These registers don't need special preservation handling in JIT code.
    pub fn caller_saved_written(&self) -> Vec<R> {
        self.written_registers
            .iter()
            .filter(|&reg| reg.is_caller_saved())
            .copied()
            .collect()
    }
    
    /// Get all caller-saved registers that are used (written or read).
    /// 
    /// These registers don't need special preservation handling in JIT code.
    pub fn caller_saved_registers(&self) -> Vec<R> {
        let mut all_registers: HashSet<R> = self.written_registers.clone();
        all_registers.extend(&self.read_registers);
        all_registers
            .iter()
            .filter(|&reg| reg.is_caller_saved())
            .copied()
            .collect()
    }
    
    /// Get all callee-saved registers that are written to.
    /// 
    /// These registers must be saved on function entry and restored on exit
    /// if used by JIT-compiled code. This is the primary list for determining
    /// which registers need preservation.
    pub fn callee_saved_written(&self) -> Vec<R> {
        self.written_registers
            .iter()
            .filter(|&reg| reg.is_callee_saved())
            .copied()
            .collect()
    }
    
    /// Get all callee-saved registers that are used (written or read).
    /// 
    /// For register preservation, prefer callee_saved_written() which only
    /// includes registers that actually need to be saved.
    pub fn callee_saved_registers(&self) -> Vec<R> {
        let mut all_registers: HashSet<R> = self.written_registers.clone();
        all_registers.extend(&self.read_registers);
        all_registers
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
        let mut all_registers: HashSet<R> = self.written_registers.clone();
        all_registers.extend(&self.read_registers);
        all_registers
            .iter()
            .filter(|&reg| reg.is_special())
            .copied()
            .collect()
    }
    
    /// Get the total number of unique registers used.
    pub fn register_count(&self) -> usize {
        let mut all_registers: HashSet<R> = self.written_registers.clone();
        all_registers.extend(&self.read_registers);
        all_registers.len()
    }
    
    /// Check if any registers are used.
    pub fn has_used_registers(&self) -> bool {
        !self.written_registers.is_empty() || !self.read_registers.is_empty()
    }
    
    /// Check if any callee-saved registers are written to.
    /// 
    /// If this returns `true`, the JIT-compiled function needs to implement
    /// a proper function prologue/epilogue to save and restore these registers.
    /// Only written registers need to be preserved, not read-only ones.
    pub fn needs_stack_frame(&self) -> bool {
        self.written_registers.iter().any(|reg| reg.is_callee_saved())
    }
    
    /// Get a count of registers by ABI class.
    /// 
    /// Returns a tuple of (caller_saved_count, callee_saved_count, special_count).
    pub fn count_by_abi_class(&self) -> (usize, usize, usize) {
        let mut caller_saved = 0;
        let mut callee_saved = 0;
        let mut special = 0;
        
        let mut all_registers: HashSet<R> = self.written_registers.clone();
        all_registers.extend(&self.read_registers);
        
        for register in &all_registers {
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
        self.written_registers.clear();
        self.read_registers.clear();
    }
    
    /// Check if a specific register is used (written or read).
    pub fn contains_register(&self, register: &R) -> bool {
        self.written_registers.contains(register) || self.read_registers.contains(register)
    }
    
    /// Check if a specific register is written to.
    pub fn contains_written_register(&self, register: &R) -> bool {
        self.written_registers.contains(register)
    }
    
    /// Check if a specific register is read from.
    pub fn contains_read_register(&self, register: &R) -> bool {
        self.read_registers.contains(register)
    }
    
    /// Merge another register usage info into this one.
    /// 
    /// This is useful for combining usage information from multiple
    /// instruction sequences.
    pub fn merge(&mut self, other: &RegisterUsageInfo<R>) {
        for register in &other.written_registers {
            self.written_registers.insert(*register);
        }
        for register in &other.read_registers {
            self.read_registers.insert(*register);
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
        
        info.add_written_register(TestRegister::T0);
        info.add_written_register(TestRegister::T1);
        info.add_written_register(TestRegister::S0);

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
        info1.add_written_register(TestRegister::T0);
        info1.add_read_register(TestRegister::S0);
        
        let mut info2 = RegisterUsageInfo::new();
        info2.add_read_register(TestRegister::T1);
        info2.add_read_register(TestRegister::S0);  // Duplicate
        
        info1.merge(&info2);
        
        assert_eq!(info1.register_count(), 3);
        assert!(info1.contains_register(&TestRegister::T0));
        assert!(info1.contains_register(&TestRegister::T1));
        assert!(info1.contains_register(&TestRegister::S0));
    }
    
    #[test]
    fn test_register_usage_display() {
        let mut info = RegisterUsageInfo::new();
        info.add_written_register(TestRegister::T0);
        info.add_written_register(TestRegister::S0);
        info.add_read_register(TestRegister::SP);
        
        let display = info.to_string();
        assert!(display.contains("total: 3"));
        assert!(display.contains("caller-saved: 1"));
        assert!(display.contains("callee-saved: 1"));
        assert!(display.contains("special: 1"));
        assert!(display.contains("needs stack frame"));
    }
    
    #[test]
    fn test_written_vs_read_registers() {
        let mut info = RegisterUsageInfo::new();
        
        // Add some written registers
        info.add_written_register(TestRegister::T0);
        info.add_written_register(TestRegister::S0);
        
        // Add some read registers
        info.add_read_register(TestRegister::T1);
        info.add_read_register(TestRegister::SP);
        
        // Test written registers
        let written = info.written_registers();
        assert_eq!(written.len(), 2);
        assert!(written.contains(&TestRegister::T0));
        assert!(written.contains(&TestRegister::S0));
        
        // Test read registers
        let read = info.read_registers();
        assert_eq!(read.len(), 2);
        assert!(read.contains(&TestRegister::T1));
        assert!(read.contains(&TestRegister::SP));
        
        // Test combined used registers
        let used = info.used_registers();
        assert_eq!(used.len(), 4);
        assert!(used.contains(&TestRegister::T0));
        assert!(used.contains(&TestRegister::S0));
        assert!(used.contains(&TestRegister::T1));
        assert!(used.contains(&TestRegister::SP));
    }
    
    #[test]
    fn test_abi_class_specific_methods() {
        let mut info = RegisterUsageInfo::new();
        
        // Mix of written and read registers
        info.add_written_register(TestRegister::T0);  // caller-saved written
        info.add_read_register(TestRegister::T1);      // caller-saved read
        info.add_written_register(TestRegister::S0);  // callee-saved written
        info.add_read_register(TestRegister::S1);      // callee-saved read
        info.add_read_register(TestRegister::SP);      // special read
        
        // Test caller-saved written (only T0)
        let caller_written = info.caller_saved_written();
        assert_eq!(caller_written.len(), 1);
        assert!(caller_written.contains(&TestRegister::T0));
        
        // Test callee-saved written (only S0)
        let callee_written = info.callee_saved_written();
        assert_eq!(callee_written.len(), 1);
        assert!(callee_written.contains(&TestRegister::S0));
        
        // Test all caller-saved (T0 and T1)
        let caller_all = info.caller_saved_registers();
        assert_eq!(caller_all.len(), 2);
        assert!(caller_all.contains(&TestRegister::T0));
        assert!(caller_all.contains(&TestRegister::T1));
        
        // Test all callee-saved (S0 and S1)
        let callee_all = info.callee_saved_registers();
        assert_eq!(callee_all.len(), 2);
        assert!(callee_all.contains(&TestRegister::S0));
        assert!(callee_all.contains(&TestRegister::S1));
        
        // Stack frame needed because S0 is written
        assert!(info.needs_stack_frame());
    }
    
    #[test]
    fn test_contains_methods() {
        let mut info = RegisterUsageInfo::new();
        info.add_written_register(TestRegister::T0);
        info.add_read_register(TestRegister::S0);
        
        // Test general contains
        assert!(info.contains_register(&TestRegister::T0));
        assert!(info.contains_register(&TestRegister::S0));
        assert!(!info.contains_register(&TestRegister::T1));
        
        // Test specific contains
        assert!(info.contains_written_register(&TestRegister::T0));
        assert!(!info.contains_written_register(&TestRegister::S0));
        
        assert!(info.contains_read_register(&TestRegister::S0));
        assert!(!info.contains_read_register(&TestRegister::T0));
    }
    
    #[test]
    fn test_stack_frame_requirements() {
        let mut info = RegisterUsageInfo::new();
        
        // Only caller-saved registers - no stack frame needed
        info.add_written_register(TestRegister::T0);
        info.add_written_register(TestRegister::T1);
        assert!(!info.needs_stack_frame());
        
        // Add callee-saved read - still no stack frame needed
        info.add_read_register(TestRegister::S0);
        assert!(!info.needs_stack_frame());
        
        // Add callee-saved write - now stack frame needed
        info.add_written_register(TestRegister::S1);
        assert!(info.needs_stack_frame());
    }
    
    #[test]
    fn test_register_overlap() {
        let mut info = RegisterUsageInfo::new();
        
        // Same register used as both written and read
        info.add_written_register(TestRegister::T0);
        info.add_read_register(TestRegister::T0);
        
        // Should appear in both lists
        assert!(info.contains_written_register(&TestRegister::T0));
        assert!(info.contains_read_register(&TestRegister::T0));
        
        // But only once in combined list
        let used = info.used_registers();
        assert_eq!(used.len(), 1);
        assert!(used.contains(&TestRegister::T0));
        
        let written = info.written_registers();
        let read = info.read_registers();
        assert_eq!(written.len(), 1);
        assert_eq!(read.len(), 1);
    }
    
    #[test]
    fn test_empty_register_usage() {
        let info = RegisterUsageInfo::<TestRegister>::new();
        
        assert_eq!(info.register_count(), 0);
        assert!(!info.has_used_registers());
        assert!(!info.needs_stack_frame());
        
        assert!(info.written_registers().is_empty());
        assert!(info.read_registers().is_empty());
        assert!(info.used_registers().is_empty());
        assert!(info.caller_saved_registers().is_empty());
        assert!(info.callee_saved_registers().is_empty());
        assert!(info.special_registers().is_empty());
        
        let (caller, callee, special) = info.count_by_abi_class();
        assert_eq!(caller, 0);
        assert_eq!(callee, 0);
        assert_eq!(special, 0);
    }
}