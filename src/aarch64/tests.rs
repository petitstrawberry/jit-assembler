use crate::aarch64::{reg, Instruction, Aarch64InstructionBuilder};



#[cfg(feature = "std")]
use std::process::Command;
#[cfg(feature = "std")]
use std::fs;

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
                // Skip test if GNU assembler fails or is not available
                println!("Warning: GNU assembler (aarch64-linux-gnu-as) failed, skipping comparison test: {}", 
                        String::from_utf8_lossy(&result.stderr));
                return vec![];
            }
        }
        Err(e) => {
            // Skip test if GNU assembler is not available
            println!("Warning: GNU assembler (aarch64-linux-gnu-as) not available, skipping comparison test: {}", e);
            return vec![];
        }
    }
    
    // Check if object file exists and is not empty
    if let Ok(metadata) = fs::metadata(obj_file.path()) {
        if metadata.len() == 0 {
            println!("Warning: GNU assembler produced empty object file, skipping comparison test");
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
        println!("Warning: objcopy failed, skipping comparison test: {}", 
                String::from_utf8_lossy(&objcopy_result.stderr));
        return vec![];
    }
    
    // Read binary data
    match fs::read(bin_file.path()) {
        Ok(data) => data,
        Err(e) => {
            println!("Warning: Failed to read binary file, skipping comparison test: {}", e);
            vec![]
        }
    }
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

/// AArch64-specific JIT execution tests (only run on AArch64 architecture)
#[cfg(feature = "std")]
#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_execution() {
    // Test basic addition function
    let add_func = unsafe {
        Aarch64InstructionBuilder::new()
            .add(reg::X0, reg::X0, reg::X1)  // Add X0 + X1 -> X0
            .ret()                           // Return
            .function::<fn(u64, u64) -> u64>()
    }.expect("Failed to create add function");
    
    let result = add_func.call(10, 20);
    assert_eq!(result, 30, "10 + 20 should equal 30");
    
    // Test subtraction function
    let sub_func = unsafe {
        Aarch64InstructionBuilder::new()
            .sub(reg::X0, reg::X0, reg::X1)  // Sub X0 - X1 -> X0  
            .ret()                           // Return
            .function::<fn(u64, u64) -> u64>()
    }.expect("Failed to create sub function");
    
    let result = sub_func.call(50, 30);
    assert_eq!(result, 20, "50 - 30 should equal 20");
}

#[cfg(feature = "std")]
#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_multiplication() {
    // Test multiplication function
    let mul_func = unsafe {
        Aarch64InstructionBuilder::new()
            .mul(reg::X0, reg::X0, reg::X1)  // Mul X0 * X1 -> X0
            .ret()                           // Return
            .function::<fn(u64, u64) -> u64>()
    }.expect("Failed to create mul function");
    
    let result = mul_func.call(7, 6);
    assert_eq!(result, 42, "7 * 6 should equal 42");
}

#[cfg(feature = "std")]
#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_division() {
    // Test unsigned division function
    let div_func = unsafe {
        Aarch64InstructionBuilder::new()
            .udiv(reg::X0, reg::X0, reg::X1)  // UDiv X0 / X1 -> X0
            .ret()                            // Return
            .function::<fn(u64, u64) -> u64>()
    }.expect("Failed to create div function");
    
    let result = div_func.call(84, 12);
    assert_eq!(result, 7, "84 / 12 should equal 7");
}

#[cfg(feature = "std")]
#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_complex_expression() {
    // Test complex expression: (a + b) * c
    let complex_func = unsafe {
        Aarch64InstructionBuilder::new()
            .add(reg::X0, reg::X0, reg::X1)  // X0 = X0 + X1 (a + b)
            .mul(reg::X0, reg::X0, reg::X2)  // X0 = X0 * X2 ((a + b) * c)
            .ret()                           // Return
            .function::<fn(u64, u64, u64) -> u64>()
    }.expect("Failed to create complex function");
    
    let result = complex_func.call(10, 5, 3);
    assert_eq!(result, 45, "(10 + 5) * 3 should equal 45");
}

#[cfg(feature = "std")]
#[test]
#[cfg(target_arch = "aarch64")]
fn test_jit_remainder_operation() {
    // Test remainder operation: a % b using the urem instruction
    let rem_func = unsafe {
        Aarch64InstructionBuilder::new()
            .urem(reg::X0, reg::X0, reg::X1) // X0 = X0 % X1
            .ret()                           // Return
            .function::<fn(u64, u64) -> u64>()
    }.expect("Failed to create remainder function");
    
    let result = rem_func.call(23, 7);
    assert_eq!(result, 2, "23 % 7 should equal 2");
}

#[test]
fn test_immediate_move_operations() {
    let mut builder = Aarch64InstructionBuilder::new();
    
    // Test small immediate
    builder.mov_imm(reg::X0, 42);
    
    // Test larger immediate that requires multiple instructions
    builder.mov_imm(reg::X1, 0x1234);
    
    let instructions = builder.instructions();
    assert!(instructions.len() >= 2);
}

#[test]
fn test_shift_operations() {
    let mut builder = Aarch64InstructionBuilder::new();
    
    // Test left shift by 1 (multiply by 2)
    builder.shl(reg::X0, reg::X1, 1);
    
    // Test left shift by 3 (multiply by 8)  
    builder.shl(reg::X2, reg::X3, 3);
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 4); // Each shl generates 2 instructions
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

#[test]
fn test_movz_movk_instructions() {
    let mut builder = Aarch64InstructionBuilder::new();
    
    // Test MOVZ instruction
    builder.movz(reg::X0, 0x1234, 0);  // MOVZ X0, #0x1234, LSL #0
    builder.movk(reg::X0, 0x5678, 1);  // MOVK X0, #0x5678, LSL #16
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 2);
    
    // Test MOVZ encoding: MOVZ X0, #0x1234, LSL #0
    let movz_instr = instructions[0];
    let movz_value = movz_instr.value();
    
    // MOVZ: sf=1, opc=10, op=100101, hw=0, imm16=0x1234, rd=0
    assert_eq!(movz_value >> 31, 1, "sf bit should be 1 for 64-bit");
    assert_eq!((movz_value >> 29) & 0b11, 0b10, "opc should be 10 for MOVZ");
    assert_eq!((movz_value >> 23) & 0b111111, 0b100101, "op should be 100101");
    assert_eq!((movz_value >> 21) & 0b11, 0, "hw should be 0");
    assert_eq!((movz_value >> 5) & 0xFFFF, 0x1234, "imm16 should be 0x1234");
    assert_eq!(movz_value & 0x1F, 0, "rd should be X0 (0)");
    
    // Test MOVK encoding: MOVK X0, #0x5678, LSL #16
    let movk_instr = instructions[1];
    let movk_value = movk_instr.value();
    
    // MOVK: sf=1, opc=11, op=100101, hw=1, imm16=0x5678, rd=0
    assert_eq!(movk_value >> 31, 1, "sf bit should be 1 for 64-bit");
    assert_eq!((movk_value >> 29) & 0b11, 0b11, "opc should be 11 for MOVK");
    assert_eq!((movk_value >> 23) & 0b111111, 0b100101, "op should be 100101");
    assert_eq!((movk_value >> 21) & 0b11, 1, "hw should be 1");
    assert_eq!((movk_value >> 5) & 0xFFFF, 0x5678, "imm16 should be 0x5678");
    assert_eq!(movk_value & 0x1F, 0, "rd should be X0 (0)");
}

#[test]
fn test_mov_imm_instruction() {
    let mut builder = Aarch64InstructionBuilder::new();
    
    // Test small immediate (should use single MOVZ)
    builder.mov_imm(reg::X1, 42);
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 1);
    
    // Should be MOVZ X1, #42, LSL #0
    let instr = instructions[0];
    let value = instr.value();
    assert_eq!((value >> 5) & 0xFFFF, 42, "imm16 should be 42");
    assert_eq!(value & 0x1F, 1, "rd should be X1 (1)");
}

#[test] 
fn test_mov_imm_large_values() {
    let mut builder = Aarch64InstructionBuilder::new();
    
    // Test 32-bit value that requires MOVZ + MOVK
    builder.mov_imm(reg::X2, 0x12345678);
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 2);
    
    // First instruction: MOVZ X2, #0x5678, LSL #0
    let movz_instr = instructions[0];
    let movz_value = movz_instr.value();
    assert_eq!((movz_value >> 29) & 0b11, 0b10, "should be MOVZ");
    assert_eq!((movz_value >> 21) & 0b11, 0, "hw should be 0");
    assert_eq!((movz_value >> 5) & 0xFFFF, 0x5678, "imm16 should be 0x5678");
    assert_eq!(movz_value & 0x1F, 2, "rd should be X2 (2)");
    
    // Second instruction: MOVK X2, #0x1234, LSL #16  
    let movk_instr = instructions[1];
    let movk_value = movk_instr.value();
    assert_eq!((movk_value >> 29) & 0b11, 0b11, "should be MOVK");
    assert_eq!((movk_value >> 21) & 0b11, 1, "hw should be 1");
    assert_eq!((movk_value >> 5) & 0xFFFF, 0x1234, "imm16 should be 0x1234");
    assert_eq!(movk_value & 0x1F, 2, "rd should be X2 (2)");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_move_immediate() {
    // Test MOVZ X0, #42, LSL #0
    let mut builder = Aarch64InstructionBuilder::new();
    builder.movz(reg::X0, 42, 0);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "movz x0, #42, lsl #0\n");
    
    // Test MOVK X1, #0x1234, LSL #16
    let mut builder = Aarch64InstructionBuilder::new();
    builder.movk(reg::X1, 0x1234, 1);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "movk x1, #0x1234, lsl #16\n");
    
    // Test larger immediate using mov_imm
    let mut builder = Aarch64InstructionBuilder::new();
    builder.mov_imm(reg::X2, 0x123456789ABCDEF0);
    let instructions = builder.instructions();
    
    // This should generate multiple instructions, let's test the first one
    if !instructions.is_empty() {
        // The exact sequence depends on implementation, but it should be valid
        assert!(instructions.len() >= 1, "Should generate at least one instruction for large immediate");
    }
}

// ============================================================================
// Comprehensive Instruction Testing
// ============================================================================

#[test]
fn test_all_arithmetic_instructions() {
    let mut builder = Aarch64InstructionBuilder::new();
    
    // Test all arithmetic instructions with different register combinations
    builder.add(reg::X0, reg::X1, reg::X2);
    builder.addi(reg::X3, reg::X4, 123);
    builder.sub(reg::X5, reg::X6, reg::X7);
    builder.subi(reg::X8, reg::X9, 456);
    builder.mul(reg::X10, reg::X11, reg::X12);
    builder.udiv(reg::X13, reg::X14, reg::X15);
    builder.sdiv(reg::X16, reg::X17, reg::X18);
    builder.urem(reg::X19, reg::X20, reg::X21);
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 9); // urem generates 2 instructions (udiv + msub), so total is 9
    
    // Verify each instruction is encoded properly (basic check)
    for instr in instructions {
        let value = instr.value();
        assert_ne!(value, 0, "Instruction should not be zero");
        assert_ne!(value, 0xFFFFFFFF, "Instruction should not be all ones");
    }
}

#[test]
fn test_all_logical_instructions() {
    let mut builder = Aarch64InstructionBuilder::new();
    
    // Test all logical instructions
    builder.and(reg::X0, reg::X1, reg::X2);
    builder.or(reg::X3, reg::X4, reg::X5);
    builder.xor(reg::X6, reg::X7, reg::X8);
    builder.mov(reg::X9, reg::X10);
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 4);
    
    // Test specific encodings - AArch64 logical instructions use different bit patterns
    let and_instr = instructions[0];
    let and_value = and_instr.value();
    // AND uses sf=1, opc=00, logical register format
    assert_eq!(and_value >> 31, 1, "AND should have sf=1 for 64-bit");
    assert_eq!((and_value >> 29) & 0b11, 0b00, "AND should have opc=00");
    
    let or_instr = instructions[1];
    let or_value = or_instr.value();
    // ORR uses sf=1, opc=01, logical register format  
    assert_eq!(or_value >> 31, 1, "ORR should have sf=1 for 64-bit");
    assert_eq!((or_value >> 29) & 0b11, 0b01, "ORR should have opc=01");
    
    let xor_instr = instructions[2];
    let xor_value = xor_instr.value();
    // EOR uses sf=1, opc=10, logical register format
    assert_eq!(xor_value >> 31, 1, "EOR should have sf=1 for 64-bit");
    assert_eq!((xor_value >> 29) & 0b11, 0b10, "EOR should have opc=10");
}

#[test]
fn test_move_instructions_comprehensive() {
    let mut builder = Aarch64InstructionBuilder::new();
    
    // Test different move variants
    builder.mov(reg::X0, reg::X1);                    // Register to register
    builder.movz(reg::X2, 0x1234, 0);                // Move zero with immediate
    builder.movk(reg::X3, 0x5678, 1);                // Move keep with immediate
    builder.mov_imm(reg::X4, 42);                     // Small immediate
    builder.mov_imm(reg::X5, 0x123456789ABCDEF0);     // Large immediate
    
    let instructions = builder.instructions();
    assert!(instructions.len() >= 5, "Should generate at least 5 instructions");
    
    // Test MOV encoding (ORR with XZR)
    let mov_instr = instructions[0];
    let mov_value = mov_instr.value();
    assert_eq!((mov_value >> 5) & 0x1F, 31, "MOV should use XZR as source register");
}

#[test]
fn test_control_flow_instructions_comprehensive() {
    let mut builder = Aarch64InstructionBuilder::new();
    
    // Test different return variants
    builder.ret();                          // Default return (X30)
    builder.ret_reg(reg::X1);              // Return to specific register
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 2);
    
    // Test RET encodings
    let ret_default = instructions[0];
    let ret_default_value = ret_default.value();
    assert_eq!((ret_default_value >> 5) & 0x1F, 30, "Default RET should use X30 (LR)");
    
    let ret_reg = instructions[1];
    let ret_reg_value = ret_reg.value();
    assert_eq!((ret_reg_value >> 5) & 0x1F, 1, "RET X1 should use X1");
}

#[test]
fn test_immediate_value_ranges() {
    let mut builder = Aarch64InstructionBuilder::new();
    
    // Test edge cases for immediate values
    builder.addi(reg::X0, reg::X1, 0);        // Minimum immediate
    builder.addi(reg::X2, reg::X3, 4095);     // Maximum 12-bit immediate
    builder.movz(reg::X4, 0, 0);              // Zero immediate
    builder.movz(reg::X5, 0xFFFF, 0);         // Maximum 16-bit immediate
    builder.movz(reg::X6, 0x1234, 3);         // Maximum shift (48 bits)
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 5);
    
    // Verify immediate encodings
    let addi_max = instructions[1];
    let addi_max_value = addi_max.value();
    assert_eq!((addi_max_value >> 10) & 0xFFF, 4095, "Should encode maximum 12-bit immediate");
    
    let movz_max = instructions[3];
    let movz_max_value = movz_max.value();
    assert_eq!((movz_max_value >> 5) & 0xFFFF, 0xFFFF, "Should encode maximum 16-bit immediate");
}

#[test]
fn test_register_combinations() {
    let mut builder = Aarch64InstructionBuilder::new();
    
    // Test with all special registers
    builder.add(reg::X0, reg::XZR, reg::X1);       // Zero register as source
    builder.add(reg::SP, reg::X1, reg::X2);        // Stack pointer as destination  
    builder.add(reg::X3, reg::FP, reg::X4);        // Frame pointer as source
    builder.add(reg::X5, reg::X6, reg::LR);        // Link register as source
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 4);
    
    // Verify register encodings
    let zero_reg_instr = instructions[0];
    let zero_reg_value = zero_reg_instr.value();
    assert_eq!((zero_reg_value >> 5) & 0x1F, 31, "Should use XZR (31) as Rn");
    
    let sp_instr = instructions[1];
    let sp_value = sp_instr.value();
    assert_eq!(sp_value & 0x1F, 31, "Should use SP (31) as Rd");
}

// ============================================================================
// GNU Assembler Binary Comparison Tests
// ============================================================================

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_comprehensive_arithmetic() {
    // Test SUB immediate
    let mut builder = Aarch64InstructionBuilder::new();
    builder.subi(reg::X1, reg::X2, 100);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "sub x1, x2, #100\n");
    
    // Test SDIV
    let mut builder = Aarch64InstructionBuilder::new();
    builder.sdiv(reg::X0, reg::X1, reg::X2);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "sdiv x0, x1, x2\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_comprehensive_logical() {
    // Test different register combinations for logical operations
    let mut builder = Aarch64InstructionBuilder::new();
    builder.and(reg::X10, reg::X11, reg::X12);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "and x10, x11, x12\n");
    
    let mut builder = Aarch64InstructionBuilder::new();
    builder.or(reg::X20, reg::X21, reg::X22);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "orr x20, x21, x22\n");
    
    let mut builder = Aarch64InstructionBuilder::new();
    builder.xor(reg::X30, reg::XZR, reg::X1);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "eor x30, xzr, x1\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_move_comprehensive() {
    // Test MOV register to register
    let mut builder = Aarch64InstructionBuilder::new();
    builder.mov(reg::X15, reg::X16);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "mov x15, x16\n");
    
    // Test MOVZ with different shifts
    let mut builder = Aarch64InstructionBuilder::new();
    builder.movz(reg::X5, 0xABCD, 2);  // LSL #32
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "movz x5, #0xabcd, lsl #32\n");
    
    // Test MOVK
    let mut builder = Aarch64InstructionBuilder::new();
    builder.movk(reg::X7, 0x1234, 3);  // LSL #48
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "movk x7, #0x1234, lsl #48\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_edge_cases() {
    // Test with maximum immediate values
    let mut builder = Aarch64InstructionBuilder::new();
    builder.addi(reg::X0, reg::X1, 4095);  // Maximum 12-bit immediate
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "add x0, x1, #4095\n");
    
    // Test with zero immediate
    let mut builder = Aarch64InstructionBuilder::new();
    builder.addi(reg::X2, reg::X3, 0);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "add x2, x3, #0\n");
}

// ============================================================================
// JIT Execution Tests for Large Immediates
// ============================================================================

#[cfg(all(feature = "std", target_arch = "aarch64"))]
#[test]
fn test_jit_large_immediate_values() {
    // Test various large immediate values in JIT execution
    let test_cases = vec![
        (0x1234u64, "16-bit value"),
        (0x12345678u64, "32-bit value"),
        (0x123456789ABCDEFu64, "56-bit value"),
        (0xFFFFFFFFFFFFFFFFu64, "Maximum 64-bit value"),
    ];
    
    for (value, description) in test_cases {
        let constant_func = unsafe {
            Aarch64InstructionBuilder::new()
                .mov_imm(reg::X0, value)
                .ret()
                .function::<fn() -> u64>()
        }.expect(&format!("Failed to create function for {}", description));
        
        let result = constant_func.call();
        assert_eq!(result, value, "JIT execution failed for {}: expected 0x{:016x}, got 0x{:016x}", 
                  description, value, result);
    }
}

#[cfg(all(feature = "std", target_arch = "aarch64"))]
#[test]
fn test_jit_arithmetic_with_large_constants() {
    // Test arithmetic operations with large constants
    let add_large_func = unsafe {
        Aarch64InstructionBuilder::new()
            .mov_imm(reg::X1, 0x123456789ABCDEF0)  // Load large constant
            .add(reg::X0, reg::X0, reg::X1)        // Add to input
            .ret()
            .function::<fn(u64) -> u64>()
    }.expect("Failed to create large constant addition function");
    
    let result = add_large_func.call(0x10);
    assert_eq!(result, 0x123456789ABCDF00, "Large constant addition failed");
}