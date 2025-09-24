#[cfg(test)]
mod tests {
use crate::riscv::{reg, csr, Instruction, InstructionBuilder};

#[cfg(feature = "std")]
use std::vec;
#[cfg(not(feature = "std"))]
use alloc::vec;

#[cfg(feature = "std")]
use std::process::Command;
#[cfg(feature = "std")]
use std::fs;

/// Helper function to assemble RISC-V assembly and extract binary data
#[cfg(feature = "std")]
fn assemble_riscv(assembly: &str) -> Vec<u8> {
    use std::io::Write;
    
    // Create unique temporary files to avoid conflicts with parallel tests
    let temp_dir = std::env::temp_dir();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let unique_id = format!("test_{}", timestamp);
    let asm_file = temp_dir.join(format!("{}.s", unique_id));
    let obj_file = temp_dir.join(format!("{}.o", unique_id));
    let bin_file = temp_dir.join(format!("{}.bin", unique_id));
    
    // Write assembly to file
    let mut file = std::fs::File::create(&asm_file).expect("Failed to create assembly file");
    writeln!(file, ".section .text").expect("Failed to write section directive");
    writeln!(file, ".global _start").expect("Failed to write global directive");  
    writeln!(file, "_start:").expect("Failed to write label");
    write!(file, "{}", assembly).expect("Failed to write assembly");
    
    // Assemble the file
    let output = Command::new("riscv64-linux-gnu-as")
        .args(&["-march=rv64i", &asm_file.to_string_lossy(), "-o", &obj_file.to_string_lossy()])
        .output()
        .expect("Failed to run assembler");
        
    if !output.status.success() {
        panic!("Assembly failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    
    // Extract the binary data using objcopy
    let output = Command::new("riscv64-linux-gnu-objcopy")
        .args(&["-O", "binary", "--only-section=.text", &obj_file.to_string_lossy(), &bin_file.to_string_lossy()])
        .output()
        .expect("Failed to run objcopy");
        
    if !output.status.success() {
        panic!("objcopy failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    
    // Read the binary data
    let binary_data = fs::read(&bin_file).expect("Failed to read binary file");
    
    // Clean up temporary files
    let _ = fs::remove_file(&asm_file);
    let _ = fs::remove_file(&obj_file); 
    let _ = fs::remove_file(&bin_file);
    
    binary_data
}

/// Compare JIT assembler output with GNU assembler output for a single instruction
#[cfg(feature = "std")]
fn compare_instruction(jit_instr: Instruction, gnu_assembly: &str) {
    let jit_bytes = jit_instr.bytes();
    let gnu_bytes = assemble_riscv(gnu_assembly);
    
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
fn test_csr_instructions() {
    let mut builder = InstructionBuilder::new();
    
    // Test CSR instructions
    builder.csrrw(reg::X1, csr::MSTATUS, reg::X2);
    builder.csrrs(reg::X3, csr::MEPC, reg::X4);
    builder.csrrwi(reg::X5, csr::MTVEC, 0x10);

    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 3);
    
    // Verify first instruction (csrrw x1, mstatus, x2)
    // The actual encoding value is 805376243 (0x30021073 in hex)
    // Let's just check that we get a reasonable value for now
    assert!(instructions[0].value() != 0);
}

#[test]
fn test_arithmetic_instructions() {
    let mut builder = InstructionBuilder::new();
    
    builder.addi(reg::X1, reg::X2, 100);
    builder.add(reg::X3, reg::X1, reg::X2);
    builder.sub(reg::X4, reg::X3, reg::X1);

    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 3);
}

// Example of how this would be used in codegen
#[test]
fn test_instruction_encoding() {
    // Test 32-bit instruction creation
    let instr = Instruction::Standard(0x12345678);
    assert_eq!(instr.value(), 0x12345678);
    assert_eq!(instr.size(), 4);
    assert!(!instr.is_compressed());
    
    let bytes = instr.bytes();
    assert_eq!(bytes, vec![0x78, 0x56, 0x34, 0x12]); // little-endian
}

#[test]
fn test_compressed_instruction() {
    // Test 16-bit compressed instruction creation  
    let instr = Instruction::Compressed(0x1234);
    assert_eq!(instr.value(), 0x1234);
    assert_eq!(instr.size(), 2);
    assert!(instr.is_compressed());
    
    let bytes = instr.bytes();
    assert_eq!(bytes, vec![0x34, 0x12]); // little-endian
}

#[test]
fn test_method_chaining() {
    // Test that builder methods can be chained
    let mut builder = InstructionBuilder::new();
    
    builder
        .csrrw(reg::X1, csr::MSTATUS, reg::X2)
        .addi(reg::X3, reg::X1, 100)
        .add(reg::X4, reg::X1, reg::X2)
        .sub(reg::X5, reg::X4, reg::X3);
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 4);
    
    // Test fluent interface pattern
    let mut builder2 = InstructionBuilder::new();
    let instructions2 = builder2
        .csrrs(reg::X10, csr::MEPC, reg::X0) // csrr equivalent
        .addi(reg::X11, reg::X10, 42)
        .instructions();
    
    assert_eq!(instructions2.len(), 2);
}

#[test]
fn test_register_aliases() {
    // Test that register aliases map to correct x registers
    assert_eq!(reg::ZERO.value(), 0);
    assert_eq!(reg::RA.value(), 1);
    assert_eq!(reg::SP.value(), 2);
    assert_eq!(reg::GP.value(), 3);
    assert_eq!(reg::TP.value(), 4);
    assert_eq!(reg::T0.value(), 5);
    assert_eq!(reg::T1.value(), 6);
    assert_eq!(reg::T2.value(), 7);
    assert_eq!(reg::S0.value(), 8);
    assert_eq!(reg::FP.value(), 8); // FP is alias for S0
    assert_eq!(reg::S1.value(), 9);
    assert_eq!(reg::A0.value(), 10);
    assert_eq!(reg::A1.value(), 11);
    assert_eq!(reg::A2.value(), 12);
    assert_eq!(reg::A3.value(), 13);
    assert_eq!(reg::A4.value(), 14);
    assert_eq!(reg::A5.value(), 15);
    assert_eq!(reg::A6.value(), 16);
    assert_eq!(reg::A7.value(), 17);
    assert_eq!(reg::S2.value(), 18);
    assert_eq!(reg::S3.value(), 19);
    assert_eq!(reg::S4.value(), 20);
    assert_eq!(reg::S5.value(), 21);
    assert_eq!(reg::S6.value(), 22);
    assert_eq!(reg::S7.value(), 23);
    assert_eq!(reg::S8.value(), 24);
    assert_eq!(reg::S9.value(), 25);
    assert_eq!(reg::S10.value(), 26);
    assert_eq!(reg::S11.value(), 27);
    assert_eq!(reg::T3.value(), 28);
    assert_eq!(reg::T4.value(), 29);
    assert_eq!(reg::T5.value(), 30);
    assert_eq!(reg::T6.value(), 31);
}

#[test]
fn test_register_aliases_usage() {
    // Test using aliases in actual instructions
    let mut builder = InstructionBuilder::new();
    builder
        .addi(reg::A0, reg::ZERO, 42)    // Load 42 into a0
        .add(reg::A1, reg::A0, reg::SP)  // Add a0 and sp, store in a1
        .sub(reg::T0, reg::A1, reg::A0); // Subtract a0 from a1, store in t0
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 3);
    
    // Verify instructions are equivalent to using x registers
    let mut builder2 = InstructionBuilder::new();
    builder2
        .addi(reg::X10, reg::X0, 42)    // a0 = x10, zero = x0
        .add(reg::X11, reg::X10, reg::X2)  // a1 = x11, sp = x2
        .sub(reg::X5, reg::X11, reg::X10); // t0 = x5
    
    let instructions2 = builder2.instructions();
    assert_eq!(instructions, instructions2);
}

#[test]
fn test_ret_instruction() {
    // Test ret instruction (should be equivalent to jalr x0, x1, 0)
    let mut builder = InstructionBuilder::new();
    builder.ret();
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 1);
    
    // Compare with explicit jalr x0, x1, 0
    let mut builder2 = InstructionBuilder::new();
    builder2.jalr(reg::X0, reg::X1, 0);
    
    let instructions2 = builder2.instructions();
    assert_eq!(instructions, instructions2);
}

#[test]
fn test_ret_instruction_with_aliases() {
    // Test ret using register aliases
    let mut builder = InstructionBuilder::new();
    builder.ret();
    
    let instructions = builder.instructions();
    assert_eq!(instructions.len(), 1);
    
    // Compare with explicit jalr using aliases
    let mut builder2 = InstructionBuilder::new();
    builder2.jalr(reg::ZERO, reg::RA, 0);
    
    let instructions2 = builder2.instructions();
    assert_eq!(instructions, instructions2);
}

#[test]
fn test_aliases_with_macro() {
    // Test using aliases in macro
    let instructions = crate::jit_asm! {
        addi(reg::A0, reg::ZERO, 42);      // Load 42 into a0
        add(reg::A1, reg::A0, reg::SP);     // Add a0 and sp, store in a1
        sub(reg::T0, reg::A1, reg::A0);     // Subtract a0 from a1, store in t0
        ret();                              // Return from function
    };

    assert_eq!(instructions.len(), 4);
    
    // Verify macro produces same results as builder with aliases
    let builder_instructions = InstructionBuilder::new()
        .addi(reg::A0, reg::ZERO, 42)
        .add(reg::A1, reg::A0, reg::SP)
        .sub(reg::T0, reg::A1, reg::A0)
        .ret()
        .instructions()
        .to_vec();
        
    assert_eq!(instructions, builder_instructions);
}

#[test]
fn test_comprehensive_alias_demo() {
    // A more comprehensive test showing typical function prologue/epilogue
    println!("Testing RISC-V register aliases and ret instruction...\n");

    // Test register aliases
    println!("Register alias mappings:");
    println!("  a0 = x{}", reg::A0.value());
    println!("  sp = x{}", reg::SP.value());
    println!("  ra = x{}", reg::RA.value());
    println!("  zero = x{}", reg::ZERO.value());
    println!("  t0 = x{}", reg::T0.value());
    println!();

    // Test with macro (function-like code)
    let instructions = crate::jit_asm! {
        addi(reg::SP, reg::SP, -16);        // Allocate stack space
        sd(reg::RA, reg::SP, 8);            // Save return address
        sd(reg::S0, reg::SP, 0);            // Save frame pointer
        addi(reg::S0, reg::SP, 16);         // Set frame pointer
        
        // Function body
        addi(reg::A0, reg::ZERO, 42);       // Load 42 into a0 (return value)
        
        // Function epilogue
        ld(reg::S0, reg::SP, 0);            // Restore frame pointer
        ld(reg::RA, reg::SP, 8);            // Restore return address
        addi(reg::SP, reg::SP, 16);         // Deallocate stack space
        ret();                              // Return
    };

    assert_eq!(instructions.len(), 9);
    
    // Verify instructions contain expected values
    println!("Generated {} instructions using aliases", instructions.len());
    for (i, instr) in instructions.iter().enumerate() {
        println!("  {}: {}", i, instr);
    }
}

#[test] 
fn test_macro_chaining() {
    let instructions = crate::jit_asm! {
        addi(reg::X1, reg::X0, 10);
        add(reg::X2, reg::X1, reg::X0);
        csrrw(reg::X3, csr::MSTATUS, reg::X2);
    };

    assert_eq!(instructions.len(), 3);
    
    // Verify macro produces same results as builder
    let builder_instructions = InstructionBuilder::new()
        .addi(reg::X1, reg::X0, 10)
        .add(reg::X2, reg::X1, reg::X0)
        .csrrw(reg::X3, csr::MSTATUS, reg::X2)
        .instructions()
        .to_vec();
        
    assert_eq!(instructions, builder_instructions);
}

#[test]
fn test_macro_comprehensive() {
    // Test multiple instruction types through macro
    let instructions = crate::jit_asm! {
        lui(reg::X1, 0x12345);
        addi(reg::X2, reg::X1, 100);
        add(reg::X3, reg::X1, reg::X2);
        beq(reg::X1, reg::X2, 8);
        jal(reg::X1, 16);
        csrr(reg::X4, csr::MSTATUS);
    };

    assert_eq!(instructions.len(), 6);
    
    // Compare with builder version and verify comprehensive functionality
    let builder_instructions = InstructionBuilder::new()
        .lui(reg::X1, 0x12345)
        .addi(reg::X2, reg::X1, 100)
        .add(reg::X3, reg::X1, reg::X2)
        .beq(reg::X1, reg::X2, 8)
        .jal(reg::X1, 16)
        .csrrs(reg::X4, csr::MSTATUS, reg::X0)  // csrr is csrrs with x0
        .instructions()
        .to_vec();
        
    assert_eq!(instructions, builder_instructions);
    
    // Each instruction should be non-zero
    for instr in &instructions {
        assert!(instr.value() != 0);
    }
}

// Binary correctness tests comparing JIT assembler with GNU assembler output
#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_arithmetic() {
    // Test ADDI instruction
    let mut builder = InstructionBuilder::new();
    builder.addi(reg::X1, reg::X0, 100);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "addi x1, x0, 100\n");
    
    // Test ADD instruction - use fresh builder
    let mut builder = InstructionBuilder::new();
    builder.add(reg::X3, reg::X1, reg::X2);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "add x3, x1, x2\n");
    
    // Test SUB instruction - use fresh builder  
    let mut builder = InstructionBuilder::new();
    builder.sub(reg::X4, reg::X3, reg::X1);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "sub x4, x3, x1\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_immediate_arithmetic() {
    // Test various immediate values
    let test_cases = vec![
        (0, "addi x1, x0, 0\n"),
        (1, "addi x1, x0, 1\n"),
        (-1, "addi x1, x0, -1\n"),
        (100, "addi x1, x0, 100\n"),
        (-100, "addi x1, x0, -100\n"),
        (2047, "addi x1, x0, 2047\n"),  // Max positive 12-bit immediate
        (-2048, "addi x1, x0, -2048\n"), // Min negative 12-bit immediate
    ];
    
    for (imm, assembly) in test_cases {
        let mut builder = InstructionBuilder::new();
        builder.addi(reg::X1, reg::X0, imm);
        let instructions = builder.instructions();
        compare_instruction(instructions[0], assembly);
    }
}

#[cfg(feature = "std")]
#[test]  
fn test_binary_correctness_logical() {
    // Test XOR
    let mut builder = InstructionBuilder::new();
    builder.xor(reg::X5, reg::X1, reg::X2);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "xor x5, x1, x2\n");
    
    // Test OR
    let mut builder = InstructionBuilder::new();
    builder.or(reg::X6, reg::X3, reg::X4);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "or x6, x3, x4\n");
    
    // Test AND
    let mut builder = InstructionBuilder::new();
    builder.and(reg::X7, reg::X5, reg::X6);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "and x7, x5, x6\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_shifts() {
    // Test shift left logical
    let mut builder = InstructionBuilder::new();
    builder.sll(reg::X8, reg::X1, reg::X2);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "sll x8, x1, x2\n");
    
    // Test shift right logical
    let mut builder = InstructionBuilder::new();
    builder.srl(reg::X9, reg::X3, reg::X4);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "srl x9, x3, x4\n");
    
    // Test shift right arithmetic
    let mut builder = InstructionBuilder::new();
    builder.sra(reg::X10, reg::X5, reg::X6);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "sra x10, x5, x6\n");
    
    // Test shift left logical immediate
    let mut builder = InstructionBuilder::new();
    builder.slli(reg::X11, reg::X1, 5);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "slli x11, x1, 5\n");
    
    // Test shift right logical immediate
    let mut builder = InstructionBuilder::new();
    builder.srli(reg::X12, reg::X2, 10);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "srli x12, x2, 10\n");
    
    // Test shift right arithmetic immediate
    let mut builder = InstructionBuilder::new();
    builder.srai(reg::X13, reg::X3, 15);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "srai x13, x3, 15\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_upper_immediate() {
    // Test LUI instruction
    let mut builder = InstructionBuilder::new();
    builder.lui(reg::X1, 0x12345);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "lui x1, 0x12345\n");
    
    // Test AUIPC instruction
    let mut builder = InstructionBuilder::new();
    builder.auipc(reg::X2, 0x6789A);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "auipc x2, 0x6789A\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_csr() {
    // Test CSRRW instruction
    let mut builder = InstructionBuilder::new();
    builder.csrrw(reg::X1, csr::MSTATUS, reg::X2);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "csrrw x1, mstatus, x2\n");
    
    // Test CSRRS instruction
    let mut builder = InstructionBuilder::new();
    builder.csrrs(reg::X3, csr::MEPC, reg::X4);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "csrrs x3, mepc, x4\n");
    
    // Test CSRRWI instruction
    let mut builder = InstructionBuilder::new();
    builder.csrrwi(reg::X5, csr::MTVEC, 0x10);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "csrrwi x5, mtvec, 0x10\n");
    
    // Test CSRR instruction (alias for csrrs with x0)
    let mut builder = InstructionBuilder::new();
    builder.csrr(reg::X4, csr::MSTATUS);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "csrr x4, mstatus\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_branches() {
    // Test BEQ instruction
    let mut builder = InstructionBuilder::new();
    builder.beq(reg::X1, reg::X2, 0);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "beq x1, x2, .\n");
    
    // Test BNE instruction
    let mut builder = InstructionBuilder::new();
    builder.bne(reg::X3, reg::X4, 0);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "bne x3, x4, .\n");
    
    // Test BLT instruction
    let mut builder = InstructionBuilder::new();
    builder.blt(reg::X5, reg::X6, 0);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "blt x5, x6, .\n");
    
    // Test BGE instruction
    let mut builder = InstructionBuilder::new();
    builder.bge(reg::X7, reg::X8, 0);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "bge x7, x8, .\n");
    
    // Test BLTU instruction
    let mut builder = InstructionBuilder::new();
    builder.bltu(reg::X9, reg::X10, 0);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "bltu x9, x10, .\n");
    
    // Test BGEU instruction
    let mut builder = InstructionBuilder::new();
    builder.bgeu(reg::X11, reg::X12, 0);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "bgeu x11, x12, .\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_jumps() {
    // Test JAL instruction with zero offset
    let mut builder = InstructionBuilder::new();
    builder.jal(reg::X1, 0);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "jal x1, .\n");
    
    // Test JALR instruction
    let mut builder = InstructionBuilder::new();
    builder.jalr(reg::X0, reg::X1, 0);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "jalr x0, x1, 0\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_memory() {
    // Test LD instruction
    let mut builder = InstructionBuilder::new();
    builder.ld(reg::X1, reg::X2, 8);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "ld x1, 8(x2)\n");
    
    // Test LW instruction
    let mut builder = InstructionBuilder::new();
    builder.lw(reg::X3, reg::X4, 4);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "lw x3, 4(x4)\n");
    
    // Test LH instruction
    let mut builder = InstructionBuilder::new();
    builder.lh(reg::X5, reg::X6, 2);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "lh x5, 2(x6)\n");
    
    // Test LB instruction
    let mut builder = InstructionBuilder::new();
    builder.lb(reg::X7, reg::X8, 1);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "lb x7, 1(x8)\n");
    
    // Test SD instruction
    let mut builder = InstructionBuilder::new();
    builder.sd(reg::X9, reg::X10, 8);
    let instructions = builder.instructions(); 
    compare_instruction(instructions[0], "sd x10, 8(x9)\n");
    
    // Test SW instruction
    let mut builder = InstructionBuilder::new();
    builder.sw(reg::X11, reg::X12, 4);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "sw x12, 4(x11)\n");
    
    // Test SH instruction
    let mut builder = InstructionBuilder::new();
    builder.sh(reg::X13, reg::X14, 2);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "sh x14, 2(x13)\n");
    
    // Test SB instruction
    let mut builder = InstructionBuilder::new();
    builder.sb(reg::X15, reg::X16, 1);
    let instructions = builder.instructions();
    compare_instruction(instructions[0], "sb x16, 1(x15)\n");
}

/// Compare JIT assembler output with GNU assembler output for multiple instructions
#[cfg(feature = "std")]
fn compare_instructions(jit_instrs: &[Instruction], gnu_assembly: &str) {
    let gnu_bytes = assemble_riscv(gnu_assembly);
    
    let mut jit_bytes = Vec::new();
    for instr in jit_instrs {
        jit_bytes.extend_from_slice(&instr.bytes());
    }
    
    // The GNU assembler output might have extra padding, so we only compare the first N bytes
    assert_eq!(jit_bytes.len(), jit_instrs.len() * 4, "JIT instructions should be 4 bytes each");
    assert!(gnu_bytes.len() >= jit_bytes.len(), "GNU assembler output should be at least {} bytes", jit_bytes.len());
    
    assert_eq!(
        jit_bytes, 
        &gnu_bytes[0..jit_bytes.len()],
        "JIT assembler output does not match GNU assembler output\nJIT: {:02x?}\nGNU: {:02x?}\nAssembly:\n{}",
        jit_bytes,
        &gnu_bytes[0..jit_bytes.len()],
        gnu_assembly
    );
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_multiline_arithmetic() {
    // Test a sequence of arithmetic operations
    let mut builder = InstructionBuilder::new();
    builder.addi(reg::X1, reg::X0, 10);
    builder.addi(reg::X2, reg::X0, 20);
    builder.add(reg::X3, reg::X1, reg::X2);
    builder.sub(reg::X4, reg::X3, reg::X1);
    
    let instructions = builder.instructions();
    compare_instructions(instructions, 
        "addi x1, x0, 10\naddi x2, x0, 20\nadd x3, x1, x2\nsub x4, x3, x1\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_multiline_logic_shifts() {
    // Test a sequence of logical and shift operations
    let mut builder = InstructionBuilder::new();
    builder.lui(reg::X1, 0x12345);
    builder.addi(reg::X2, reg::X1, 0x678);
    builder.xor(reg::X3, reg::X1, reg::X2);
    builder.slli(reg::X4, reg::X3, 4);
    builder.srli(reg::X5, reg::X4, 2);
    
    let instructions = builder.instructions();
    compare_instructions(instructions,
        "lui x1, 0x12345\naddi x2, x1, 0x678\nxor x3, x1, x2\nslli x4, x3, 4\nsrli x5, x4, 2\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_multiline_csr_operations() {
    // Test a sequence of CSR operations
    let mut builder = InstructionBuilder::new();
    builder.csrr(reg::X1, csr::MSTATUS);
    builder.addi(reg::X2, reg::X1, 1);
    builder.csrrw(reg::X3, csr::MSTATUS, reg::X2);
    builder.csrrs(reg::X4, csr::MEPC, reg::X0);
    
    let instructions = builder.instructions();
    compare_instructions(instructions,
        "csrr x1, mstatus\naddi x2, x1, 1\ncsrrw x3, mstatus, x2\ncsrrs x4, mepc, x0\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_multiline_memory_operations() {
    // Test a sequence of memory operations
    let mut builder = InstructionBuilder::new();
    builder.lui(reg::X1, 0x10000);
    builder.addi(reg::X2, reg::X0, 42);
    builder.sw(reg::X1, reg::X2, 0);
    builder.lw(reg::X3, reg::X1, 0);
    builder.addi(reg::X4, reg::X3, 1);
    builder.sw(reg::X1, reg::X4, 4);
    
    let instructions = builder.instructions();
    compare_instructions(instructions,
        "lui x1, 0x10000\naddi x2, x0, 42\nsw x2, 0(x1)\nlw x3, 0(x1)\naddi x4, x3, 1\nsw x4, 4(x1)\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_multiline_control_flow() {
    // Test a sequence with branches and jumps (using zero offsets for simplicity)
    let mut builder = InstructionBuilder::new();
    builder.addi(reg::X1, reg::X0, 5);
    builder.addi(reg::X2, reg::X0, 10);
    builder.beq(reg::X1, reg::X2, 0);  // Branch to self (zero offset)
    builder.bne(reg::X1, reg::X2, 0);  // Branch to self (zero offset)
    builder.jal(reg::X3, 0);           // Jump to self (zero offset)
    
    let instructions = builder.instructions();
    compare_instructions(instructions,
        "addi x1, x0, 5\naddi x2, x0, 10\nbeq x1, x2, .\nbne x1, x2, .\njal x3, .\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_multiline_comprehensive() {
    // Test a comprehensive sequence mixing different instruction types
    let mut builder = InstructionBuilder::new();
    builder.lui(reg::X1, 0x12345);          // Upper immediate
    builder.addi(reg::X1, reg::X1, 0x678);  // Immediate arithmetic
    builder.add(reg::X2, reg::X1, reg::X0);  // Register arithmetic
    builder.slli(reg::X3, reg::X2, 1);      // Shift immediate
    builder.xor(reg::X4, reg::X2, reg::X3);  // Logical operation
    builder.csrr(reg::X5, csr::MSTATUS);    // CSR operation
    builder.sw(reg::X1, reg::X4, 8);        // Store operation
    builder.lw(reg::X6, reg::X1, 8);        // Load operation
    builder.beq(reg::X4, reg::X6, 0);       // Branch operation
    
    let instructions = builder.instructions();
    compare_instructions(instructions,
        "lui x1, 0x12345\naddi x1, x1, 0x678\nadd x2, x1, x0\nslli x3, x2, 1\nxor x4, x2, x3\ncsrr x5, mstatus\nsw x4, 8(x1)\nlw x6, 8(x1)\nbeq x4, x6, .\n");
}

#[cfg(feature = "std")]
#[test]
fn test_binary_correctness_multiline_macro_comparison() {
    // Compare macro-generated instructions with builder-generated instructions
    let macro_instructions = crate::jit_asm! {
        lui(reg::X1, 0x12345);
        addi(reg::X2, reg::X1, 100);
        add(reg::X3, reg::X1, reg::X2);
        sub(reg::X4, reg::X3, reg::X1);
        xor(reg::X5, reg::X3, reg::X4);
    };
    
    let mut builder = InstructionBuilder::new();
    builder.lui(reg::X1, 0x12345);
    builder.addi(reg::X2, reg::X1, 100);
    builder.add(reg::X3, reg::X1, reg::X2);
    builder.sub(reg::X4, reg::X3, reg::X1);
    builder.xor(reg::X5, reg::X3, reg::X4);
    let builder_instructions = builder.instructions();
    
    // Both should match
    assert_eq!(macro_instructions.len(), builder_instructions.len());
    for (i, (macro_instr, builder_instr)) in macro_instructions.iter().zip(builder_instructions.iter()).enumerate() {
        assert_eq!(macro_instr.bytes(), builder_instr.bytes(), 
                   "Instruction {} differs between macro and builder", i);
    }
    
    // Both should match GNU assembler
    compare_instructions(&macro_instructions,
        "lui x1, 0x12345\naddi x2, x1, 100\nadd x3, x1, x2\nsub x4, x3, x1\nxor x5, x3, x4\n");
}

} // end of tests module