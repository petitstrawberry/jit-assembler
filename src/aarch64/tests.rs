use crate::aarch64::{reg, Instruction, Aarch64InstructionBuilder};

#[cfg(feature = "std")]
use std::vec;
#[cfg(not(feature = "std"))]
use alloc::vec;

#[cfg(feature = "std")]
use std::process::Command;
#[cfg(feature = "std")]
use std::fs;
#[cfg(feature = "std")]
use crate::common::InstructionBuilder;

/// Helper function to assemble AArch64 assembly and extract binary data
#[cfg(feature = "std")]
fn assemble_aarch64(assembly: &str) -> Vec<u8> {
    use std::io::Write;
    
    // Create temporary assembly file
    let asm_content = format!(".text\n{}", assembly);
    let mut asm_file = tempfile::NamedTempFile::new().expect("Failed to create temp file");
    asm_file.write_all(asm_content.as_bytes()).expect("Failed to write assembly");
    asm_file.flush().expect("Failed to flush temp file");
    
    // Assemble with GNU assembler
    let obj_file = tempfile::NamedTempFile::new().expect("Failed to create temp obj file");
    let output = Command::new("aarch64-linux-gnu-as")
        .arg("-o")
        .arg(obj_file.path())
        .arg(asm_file.path())
        .output();
    
    match output {
        Ok(result) => {
            if !result.status.success() {
                panic!("GNU assembler failed: {}", String::from_utf8_lossy(&result.stderr));
            }
        }
        Err(e) => {
            // Skip test if GNU assembler is not available
            println!("Warning: GNU assembler (aarch64-linux-gnu-as) not available, skipping comparison test: {}", e);
            return vec![];
        }
    }
    
    // Extract binary data from object file using objcopy
    let bin_file = tempfile::NamedTempFile::new().expect("Failed to create temp bin file");
    let objcopy_result = Command::new("aarch64-linux-gnu-objcopy")
        .arg("-O")
        .arg("binary")
        .arg("--only-section=.text")
        .arg(obj_file.path())
        .arg(bin_file.path())
        .output()
        .expect("Failed to run objcopy");
    
    if !objcopy_result.status.success() {
        panic!("objcopy failed: {}", String::from_utf8_lossy(&objcopy_result.stderr));
    }
    
    // Read binary data
    fs::read(bin_file.path()).expect("Failed to read binary file")
}

/// Compare JIT assembler output with GNU assembler output for a single instruction
#[cfg(feature = "std")]
fn compare_instruction(jit_instr: Instruction, gnu_assembly: &str) {
    let jit_bytes = jit_instr.bytes();
    let gnu_bytes = assemble_aarch64(gnu_assembly);
    
    // Skip comparison if GNU assembler is not available
    if gnu_bytes.is_empty() {
        return;
    }
    
    // The GNU assembler output might have extra padding, so we only compare the first 4 bytes
    assert_eq!(jit_bytes.len(), 4, "JIT instruction should be 4 bytes");
    assert!(gnu_bytes.len() >= 4, "GNU assembler output should be at least 4 bytes");
    
    assert_eq!(
        jit_bytes, 
        &gnu_bytes[0..4],
        "JIT assembler output does not match GNU assembler output\nJIT: {:02x?}\nGNU: {:02x?}\nAssembly: {}",
        jit_bytes,
        &gnu_bytes[0..4],
        gnu_assembly
    );
}

#[test]
fn test_register_values() {
    // Test that register values are correct
    assert_eq!(reg::X0.value(), 0);
    assert_eq!(reg::X1.value(), 1);
    assert_eq!(reg::X10.value(), 10);
    assert_eq!(reg::X29.value(), 29); // FP
    assert_eq!(reg::X30.value(), 30); // LR
    assert_eq!(reg::SP.value(), 31);
    assert_eq!(reg::XZR.value(), 31);
    
    // Test aliases
    assert_eq!(reg::FP.value(), 29);
    assert_eq!(reg::LR.value(), 30);
}

#[test]
fn test_basic_arithmetic_instructions() {
    let mut builder = Aarch64InstructionBuilder::new();
    
    builder.add(reg::X0, reg::X1, reg::X2);
    builder.addi(reg::X3, reg::X4, 100);
    builder.sub(reg::X5, reg::X6, reg::X7);
    builder.subi(reg::X8, reg::X9, 50);
    builder.mul(reg::X10, reg::X11, reg::X12);
    builder.udiv(reg::X13, reg::X14, reg::X15);
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 6);
}

#[test]
fn test_logical_instructions() {
    let mut builder = Aarch64InstructionBuilder::new();
    
    builder.and(reg::X0, reg::X1, reg::X2);
    builder.or(reg::X3, reg::X4, reg::X5);
    builder.xor(reg::X6, reg::X7, reg::X8);
    builder.mov(reg::X9, reg::X10);
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 4);
}

#[test]
fn test_control_flow_instructions() {
    let mut builder = Aarch64InstructionBuilder::new();
    
    builder.ret();
    builder.ret_reg(reg::X1);
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 2);
}

#[test]
fn test_instruction_encoding() {
    // Test a simple ADD instruction: ADD X0, X1, X2
    let mut builder = Aarch64InstructionBuilder::new();
    builder.add(reg::X0, reg::X1, reg::X2);
    let instructions = builder.instructions();
    
    assert_eq!(instructions.len(), 1);
    
    // Check that the instruction has expected bits
    let instr = instructions[0];
    let value = instr.value();
    
    // For ADD X0, X1, X2:
    // - SF (bit 31) = 1 (64-bit)
    // - op (bit 30) = 0 (ADD)
    // - S (bit 29) = 0 (don't set flags)
    // - Fixed bits 28-24 = 01011
    // - Rm (bits 20-16) = X2 = 2
    // - Rn (bits 9-5) = X1 = 1  
    // - Rd (bits 4-0) = X0 = 0
    
    assert_eq!(value >> 31, 1, "SF bit should be 1 for 64-bit operation");
    assert_eq!((value >> 30) & 1, 0, "op bit should be 0 for ADD");
    assert_eq!((value >> 29) & 1, 0, "S bit should be 0");
    assert_eq!((value >> 24) & 0x1f, 0b01011, "Fixed bits should be 01011");
    assert_eq!((value >> 16) & 0x1f, 2, "Rm should be X2 (2)");
    assert_eq!((value >> 5) & 0x1f, 1, "Rn should be X1 (1)");
    assert_eq!(value & 0x1f, 0, "Rd should be X0 (0)");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_basic_arithmetic() {
    // Test ADD X0, X1, X2
    let mut builder = Aarch64InstructionBuilder::new();
    builder.add(reg::X0, reg::X1, reg::X2);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "add x0, x1, x2\n");
    
    // Test SUB X3, X4, X5
    let mut builder = Aarch64InstructionBuilder::new();
    builder.sub(reg::X3, reg::X4, reg::X5);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "sub x3, x4, x5\n");
    
    // Test ADD immediate: ADD X0, X1, #100
    let mut builder = Aarch64InstructionBuilder::new();
    builder.addi(reg::X0, reg::X1, 100);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "add x0, x1, #100\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_logical() {
    // Test AND X0, X1, X2
    let mut builder = Aarch64InstructionBuilder::new();
    builder.and(reg::X0, reg::X1, reg::X2);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "and x0, x1, x2\n");
    
    // Test ORR X3, X4, X5
    let mut builder = Aarch64InstructionBuilder::new();
    builder.or(reg::X3, reg::X4, reg::X5);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "orr x3, x4, x5\n");
    
    // Test EOR X6, X7, X8
    let mut builder = Aarch64InstructionBuilder::new();
    builder.xor(reg::X6, reg::X7, reg::X8);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "eor x6, x7, x8\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_multiply_divide() {
    // Test MUL X0, X1, X2
    let mut builder = Aarch64InstructionBuilder::new();
    builder.mul(reg::X0, reg::X1, reg::X2);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "mul x0, x1, x2\n");
    
    // Test UDIV X3, X4, X5
    let mut builder = Aarch64InstructionBuilder::new();
    builder.udiv(reg::X3, reg::X4, reg::X5);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "udiv x3, x4, x5\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_control_flow() {
    // Test RET
    let mut builder = Aarch64InstructionBuilder::new();
    builder.ret();
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "ret\n");
}

#[cfg(feature = "std")]
#[test]
fn test_jit_add_function() {
    // Create a function that adds two numbers (X0 + X1 -> X0)
    let jit_func = unsafe {
        Aarch64InstructionBuilder::new()
            .add(reg::X0, reg::X0, reg::X1)  // Add X0 + X1, result in X0
            .ret()                           // Return
            .function::<fn(u64, u64) -> u64>()
    };

    assert!(jit_func.is_ok(), "JIT add function creation should succeed");
}

#[cfg(feature = "register-tracking")]
mod register_tracking_tests {
    use super::*;
    
    #[test]
    fn test_basic_arithmetic_tracking() {
        let mut builder = Aarch64InstructionBuilder::new();
        builder.add(reg::X0, reg::X1, reg::X2);
        
        let usage = builder.register_usage();
        
        // Check written registers
        let written = usage.written_registers();
        assert_eq!(written.len(), 1);
        assert!(usage.contains_written_register(&reg::X0));
        
        // Check read registers
        let read = usage.read_registers();
        assert_eq!(read.len(), 2);
        assert!(usage.contains_read_register(&reg::X1));
        assert!(usage.contains_read_register(&reg::X2));
        
        // Check total usage
        assert_eq!(usage.register_count(), 3);
        assert!(usage.has_used_registers());
    }
    
    #[test]
    fn test_immediate_tracking() {
        let mut builder = Aarch64InstructionBuilder::new();
        builder.addi(reg::X0, reg::X1, 100);
        
        let usage = builder.register_usage();
        
        // Only X1 should be read, X0 should be written
        assert!(usage.contains_written_register(&reg::X0));
        assert!(usage.contains_read_register(&reg::X1));
        assert_eq!(usage.register_count(), 2);
    }
    
    #[test]
    fn test_move_tracking() {
        let mut builder = Aarch64InstructionBuilder::new();
        builder.mov(reg::X0, reg::X1);
        
        let usage = builder.register_usage();
        
        // X1 should be read, X0 should be written
        assert!(usage.contains_written_register(&reg::X0));
        assert!(usage.contains_read_register(&reg::X1));
        assert_eq!(usage.register_count(), 2);
    }
}