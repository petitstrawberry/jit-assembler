//! Example demonstrating InstructionCollection merge functionality
//! 
//! This example shows how to combine multiple instruction sequences,
//! such as function prologue, main body, and epilogue, in arbitrary order.
//! When the register-tracking feature is enabled, register usage information
//! is also properly merged.

use jit_assembler::riscv64::{Riscv64InstructionBuilder, reg};
use jit_assembler::common::InstructionBuilder;

#[cfg(feature = "register-tracking")]
use jit_assembler::common::InstructionCollectionWithUsage;

fn main() {
    println!("=== Instruction Collection Merge Demo ===\n");
    
    // Create three separate builders for different parts of the code
    
    // Builder 1: Main computation
    println!("1. Building main computation...");
    let mut main_builder = Riscv64InstructionBuilder::new();
    main_builder
        .add(reg::A0, reg::A1, reg::A2)  // a0 = a1 + a2
        .mul(reg::A0, reg::A0, reg::A3)  // a0 = a0 * a3
        .addi(reg::A0, reg::A0, 100);    // a0 = a0 + 100
    
    // Builder 2: Function prologue (register save)
    // This is typically built AFTER the main code to know which registers to save
    println!("2. Building prologue (register save)...");
    let mut prologue_builder = Riscv64InstructionBuilder::new();
    prologue_builder
        .addi(reg::SP, reg::SP, -16)     // Allocate stack space
        .sd(reg::SP, reg::RA, 8)         // Save return address
        .sd(reg::SP, reg::S0, 0);        // Save frame pointer
    
    // Builder 3: Function epilogue (register restore)
    println!("3. Building epilogue (register restore)...");
    let mut epilogue_builder = Riscv64InstructionBuilder::new();
    epilogue_builder
        .ld(reg::RA, reg::SP, 8)         // Restore return address
        .ld(reg::S0, reg::SP, 0)         // Restore frame pointer
        .addi(reg::SP, reg::SP, 16)      // Deallocate stack space
        .ret();                          // Return
    
    // Get instruction collections from each builder
    let prologue = prologue_builder.instructions();
    let main_code = main_builder.instructions();
    let epilogue = epilogue_builder.instructions();
    
    println!("\nInstruction counts:");
    println!("  Prologue: {} instructions", prologue.len());
    println!("  Main:     {} instructions", main_code.len());
    println!("  Epilogue: {} instructions", epilogue.len());
    
    // Combine them in the desired order: prologue + main + epilogue
    println!("\n4. Combining instruction collections...");
    let combined = prologue + main_code + epilogue;
    
    println!("  Combined: {} instructions", combined.len());
    
    // Display the combined instructions
    println!("\nCombined instructions:");
    for (i, instr) in combined.iter().enumerate() {
        println!("  [{:2}] {}", i, instr);
    }
    
    // Convert to bytes for execution
    let bytes = combined.to_bytes();
    println!("\nGenerated {} bytes of machine code", bytes.len());
    
    // Demonstrate with register tracking (when feature is enabled)
    #[cfg(feature = "register-tracking")]
    demonstrate_with_register_tracking();
}

#[cfg(feature = "register-tracking")]
fn demonstrate_with_register_tracking() {
    println!("\n\n=== With Register Tracking ===\n");
    
    // Create the same three builders
    let mut main_builder = Riscv64InstructionBuilder::new();
    main_builder
        .add(reg::A0, reg::A1, reg::A2)
        .mul(reg::A0, reg::A0, reg::A3)
        .addi(reg::A0, reg::A0, 100);
    
    let mut prologue_builder = Riscv64InstructionBuilder::new();
    prologue_builder
        .addi(reg::SP, reg::SP, -16)
        .sd(reg::SP, reg::RA, 8)
        .sd(reg::SP, reg::S0, 0);
    
    let mut epilogue_builder = Riscv64InstructionBuilder::new();
    epilogue_builder
        .ld(reg::RA, reg::SP, 8)
        .ld(reg::S0, reg::SP, 0)
        .addi(reg::SP, reg::SP, 16)
        .ret();
    
    // Create tracked collections that include register usage info
    let prologue = InstructionCollectionWithUsage::new(
        prologue_builder.instructions(),
        prologue_builder.register_usage().clone()
    );
    let main_code = InstructionCollectionWithUsage::new(
        main_builder.instructions(),
        main_builder.register_usage().clone()
    );
    let epilogue = InstructionCollectionWithUsage::new(
        epilogue_builder.instructions(),
        epilogue_builder.register_usage().clone()
    );
    
    println!("Register usage before merge:");
    println!("  Prologue: {}", prologue.register_usage());
    println!("  Main:     {}", main_code.register_usage());
    println!("  Epilogue: {}", epilogue.register_usage());
    
    // Merge with register usage tracking
    let combined = prologue + main_code + epilogue;
    
    println!("\nRegister usage after merge:");
    println!("  Combined: {}", combined.register_usage());
    
    let usage = combined.register_usage();
    println!("\nDetailed register analysis:");
    println!("  Total registers used: {}", usage.register_count());
    println!("  Caller-saved: {:?}", usage.caller_saved_registers());
    println!("  Callee-saved: {:?}", usage.callee_saved_registers());
    println!("  Special: {:?}", usage.special_registers());
    println!("  Needs stack frame: {}", usage.needs_stack_frame());
    
    // You can still access the instructions
    println!("\nFinal instruction count: {}", combined.instructions().len());
}
