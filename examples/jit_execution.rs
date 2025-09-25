//! JIT Execution Example
//! 
//! This example demonstrates how to use the JIT execution functionality
//! to create and execute functions at runtime.
//! 
//! Note: This example only works on RISC-V hosts or in emulation.
//! On other architectures, the functions will be created successfully
//! but calling them will likely crash.

use jit_assembler::riscv::{reg, csr, Riscv64InstructionBuilder};
use jit_assembler::common::InstructionBuilder;

fn main() {
    println!("JIT Assembler - JIT Execution Example");
    println!("=====================================");

    // Example 1: Constant function
    println!("\n1. Creating a function that returns 42...");
    let constant_func = unsafe {
        Riscv64InstructionBuilder::new()
            .addi(reg::A0, reg::ZERO, 42)  // Load 42 into a0 (return value)
            .ret()                         // Return
            .function::<fn() -> u64>()
    };

    match constant_func {
        Ok(func) => {
            println!("   Function created successfully!");
            // Only call on RISC-V
            if cfg!(target_arch = "riscv64") {
                let result = func.call();
                println!("   Result: {}", result);
            } else {
                println!("   (Skipping execution - not on RISC-V host)");
            }
        }
        Err(e) => println!("   Failed to create function: {}", e),
    }

    // Example 2: Addition function
    println!("\n2. Creating a function that adds two numbers...");
    let add_func = unsafe {
        Riscv64InstructionBuilder::new()
            .add(reg::A0, reg::A0, reg::A1)  // Add a0 + a1, result in a0
            .ret()                           // Return
            .function::<fn(u64, u64) -> u64>()
    };

    match add_func {
        Ok(func) => {
            println!("   Function created successfully!");
            if cfg!(target_arch = "riscv64") {
                let result = func.call(10, 20);
                println!("   10 + 20 = {}", result);
            } else {
                println!("   (Skipping execution - not on RISC-V host)");
            }
        }
        Err(e) => println!("   Failed to create function: {}", e),
    }

    // Example 3: More complex function with immediate values
    println!("\n3. Creating a function that computes (x + 100) * 2...");
    let complex_func = unsafe {
        Riscv64InstructionBuilder::new()
            .addi(reg::A0, reg::A0, 100)    // x + 100
            .slli(reg::A0, reg::A0, 1)      // << 1 (multiply by 2)
            .ret()                          // Return
            .function::<fn(u64) -> u64>()
    };

    match complex_func {
        Ok(func) => {
            println!("   Function created successfully!");
            if cfg!(target_arch = "riscv64") {
                let result = func.call(5);
                println!("   (5 + 100) * 2 = {}", result);
            } else {
                println!("   (Skipping execution - not on RISC-V host)");
            }
        }
        Err(e) => println!("   Failed to create function: {}", e),
    }

    // Example 4: CSR access function (more advanced)
    println!("\n4. Creating a function that reads MEPC CSR...");
    let csr_func = unsafe {
        Riscv64InstructionBuilder::new()
            .csrr(reg::A0, csr::MEPC)       // Read MEPC into a0
            .ret()                          // Return
            .function::<fn() -> u64>()
    };

    match csr_func {
        Ok(_func) => {
            println!("   Function created successfully!");
            if cfg!(target_arch = "riscv64") {
                // Note: This might require privileged mode
                println!("   (CSR access requires appropriate privileges)");
            } else {
                println!("   (Skipping execution - not on RISC-V host)");
            }
        }
        Err(e) => println!("   Failed to create function: {}", e),
    }

    println!("\nAll examples completed!");
    println!("Note: To actually execute these functions, run this example on a RISC-V system.");
}