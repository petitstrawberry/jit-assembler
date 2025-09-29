use jit_assembler::riscv64::{Riscv64InstructionBuilder, reg};
use jit_assembler::common::InstructionBuilder;

#[cfg(feature = "register-tracking")]
fn main() {
    let mut builder = Riscv64InstructionBuilder::new();
    
    // Add some instructions
    builder
        .add(reg::T0, reg::T1, reg::T2)     // T0, T1, T2を使用
        .addi(reg::T3, reg::SP, 16)         // T3, SPを使用
        .mul(reg::A0, reg::A1, reg::A2)     // A0, A1, A2を使用
        .ld(reg::S0, reg::T0, 8)            // S0, T0を使用
        .sd(reg::SP, reg::S1, -16);         // SP, S1を使用
    
    let usage = builder.register_usage();
    
    println!("=== Register Usage Analysis ===");
    println!("Total registers used: {}", usage.register_count());
    println!("Caller-saved registers: {:?}", usage.caller_saved_registers());
    println!("Callee-saved registers: {:?}", usage.callee_saved_registers());
    println!("Special registers: {:?}", usage.special_registers());
    println!("Needs stack frame: {}", usage.needs_stack_frame());
    println!("Usage info: {}", usage);
}

#[cfg(not(feature = "register-tracking"))]
fn main() {
    println!("Register tracking feature is not enabled. Run with --features register-tracking");
}