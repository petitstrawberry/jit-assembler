//! Register usage tracking for RISC-V instruction generation
//! 
//! This module provides functionality to track register usage during instruction
//! generation, including ABI classification (caller-saved, callee-saved, special).

use super::instruction::Register;

// Conditional imports for HashSet
#[cfg(all(feature = "register-tracking", feature = "std"))]
use std::collections::HashSet;

#[cfg(all(feature = "register-tracking", not(feature = "std")))]
use hashbrown::HashSet;

/// ABI classification for RISC-V registers
#[cfg(feature = "register-tracking")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiClass {
    /// Caller-saved registers (T0-T6, A0-A7, RA)
    CallerSaved,
    /// Callee-saved registers (S0-S11, SP)
    CalleeSaved,
    /// Special registers (ZERO, GP, TP)
    Special,
}

#[cfg(feature = "register-tracking")]
impl Register {
    /// Get the ABI classification for this register
    pub const fn abi_class(&self) -> AbiClass {
        match self.0 {
            0 => AbiClass::Special,        // ZERO
            1 => AbiClass::CallerSaved,    // RA  
            2 => AbiClass::CalleeSaved,    // SP
            3 => AbiClass::Special,        // GP
            4 => AbiClass::Special,        // TP
            5..=7 => AbiClass::CallerSaved,    // T0-T2
            8..=9 => AbiClass::CalleeSaved,    // S0-S1/FP
            10..=17 => AbiClass::CallerSaved,  // A0-A7
            18..=27 => AbiClass::CalleeSaved,  // S2-S11
            28..=31 => AbiClass::CallerSaved,  // T3-T6
            _ => AbiClass::Special,
        }
    }

    /// Check if this register is the ZERO register
    pub const fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

/// Information about register usage during instruction generation
#[cfg(feature = "register-tracking")]
#[derive(Debug, Clone)]
pub struct RegisterUsageInfo {
    /// All registers that have been used
    pub used_registers: HashSet<Register>,
    
    /// Caller-saved registers that have been used (T0-T6, A0-A7, RA)
    pub caller_saved: HashSet<Register>,
    
    /// Callee-saved registers that have been used (S0-S11, SP)
    pub callee_saved: HashSet<Register>,
    
    /// Special registers that have been used (ZERO, GP, TP)
    pub special: HashSet<Register>,
    
    /// Currently active registers (for tracking concurrent usage)
    pub(crate) current_active: HashSet<Register>,
    
    /// Maximum number of registers used concurrently
    pub max_concurrent_usage: usize,
}

#[cfg(feature = "register-tracking")]
impl RegisterUsageInfo {
    /// Create a new empty register usage tracker
    pub fn new() -> Self {
        Self {
            used_registers: HashSet::new(),
            caller_saved: HashSet::new(),
            callee_saved: HashSet::new(),
            special: HashSet::new(),
            current_active: HashSet::new(),
            max_concurrent_usage: 0,
        }
    }

    /// Reset all tracking information
    pub fn reset(&mut self) {
        self.used_registers.clear();
        self.caller_saved.clear();
        self.callee_saved.clear();
        self.special.clear();
        self.current_active.clear();
        self.max_concurrent_usage = 0;
    }

    /// Track the usage of a set of registers
    pub fn track_registers(&mut self, registers: &[Register]) {
        for &reg in registers {
            if !reg.is_zero() { // Don't count ZERO register usage
                self.used_registers.insert(reg);
                self.current_active.insert(reg);
                
                // Classify by ABI
                match reg.abi_class() {
                    AbiClass::CallerSaved => { 
                        self.caller_saved.insert(reg); 
                    }
                    AbiClass::CalleeSaved => { 
                        self.callee_saved.insert(reg); 
                    }
                    AbiClass::Special => {
                        self.special.insert(reg);
                    }
                }
            }
        }
        
        // Update maximum concurrent usage
        self.max_concurrent_usage = self.max_concurrent_usage.max(self.current_active.len());
    }

    /// Get registers that need to be preserved across function calls (callee-saved)
    pub fn needs_preservation(&self) -> &HashSet<Register> {
        &self.callee_saved
    }
    
    /// Get registers that can be freely used (caller-saved)
    pub fn freely_usable(&self) -> &HashSet<Register> {
        &self.caller_saved
    }
    
    /// Get special registers that have been used
    pub fn special_registers(&self) -> &HashSet<Register> {
        &self.special
    }
    
    /// Get the total number of registers used
    pub fn total_count(&self) -> usize {
        self.used_registers.len()
    }

    /// Check if any callee-saved registers are used (indicates need for preservation)
    pub fn uses_callee_saved(&self) -> bool {
        !self.callee_saved.is_empty()
    }

    /// Get a summary string of register usage
    pub fn summary(&self) -> String {
        #[cfg(feature = "std")]
        {
            format!(
                "Register Usage: {} total, {} caller-saved, {} callee-saved, {} special, {} max concurrent",
                self.total_count(),
                self.caller_saved.len(),
                self.callee_saved.len(),
                self.special.len(),
                self.max_concurrent_usage
            )
        }
        #[cfg(not(feature = "std"))]
        {
            // For no_std environments, provide a simpler summary
            "Register usage tracking active"
        }
    }
}

#[cfg(feature = "register-tracking")]
impl Default for RegisterUsageInfo {
    fn default() -> Self {
        Self::new()
    }
}

// Provide stub types when register-tracking feature is disabled
#[cfg(not(feature = "register-tracking"))]
pub struct RegisterUsageInfo;

#[cfg(not(feature = "register-tracking"))]
impl RegisterUsageInfo {
    pub fn new() -> Self { Self }
    pub fn reset(&mut self) {}
    pub fn track_registers(&mut self, _registers: &[Register]) {}
    pub fn total_count(&self) -> usize { 0 }
    pub fn uses_callee_saved(&self) -> bool { false }
    pub fn summary(&self) -> &'static str { "Register tracking disabled" }
}

#[cfg(not(feature = "register-tracking"))]
impl Default for RegisterUsageInfo {
    fn default() -> Self {
        Self::new()
    }
}