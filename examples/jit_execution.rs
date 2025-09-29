//! JIT Execution Example
//! 
//! This example demonstrates how to use the JIT execution functionality
//! to create and execute functions at runtime.
//! 
//! This example supports both RISC-V and AArch64 architectures depending
//! on the enabled features.

use jit_assembler::common::InstructionBuilder;

#[cfg(feature = "riscv64")]
use jit_assembler::riscv64::{reg as riscv_reg, csr, Riscv64InstructionBuilder};

#[cfg(feature = "aarch64")]
use jit_assembler::aarch64::{reg as aarch64_reg, Aarch64InstructionBuilder};

fn main() {
    println!("JIT Assembler - JIT Execution Example");
    println!("=====================================");

    #[cfg(feature = "riscv64")]
    riscv_examples();

    #[cfg(feature = "aarch64")]
    aarch64_examples();

    #[cfg(not(any(feature = "riscv64", feature = "aarch64")))]
    println!("No architecture features enabled. Enable 'riscv' or 'aarch64' features to see examples.");
}

#[cfg(feature = "riscv64")]
fn riscv_examples() {
    println!("\n=== RISC-V Examples ===");

    // Example 1: Constant function
    println!("\n1. Creating a function that returns 42...");
    let constant_func = unsafe {
        Riscv64InstructionBuilder::new()
            .addi(riscv_reg::A0, riscv_reg::ZERO, 42)  // Load 42 into a0 (return value)
            .ret()                                     // Return
            .function::<fn() -> u64>()
    };

    match constant_func {
        Ok(func) => {
            println!("   Function created successfully!");
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
            .add(riscv_reg::A0, riscv_reg::A0, riscv_reg::A1)  // Add a0 + a1, result in a0
            .ret()                                              // Return
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
            .addi(riscv_reg::A0, riscv_reg::A0, 100)    // x + 100
            .slli(riscv_reg::A0, riscv_reg::A0, 1)       // << 1 (multiply by 2)
            .ret()                                       // Return
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
            .csrr(riscv_reg::A0, csr::MEPC)              // Read MEPC into a0
            .ret()                                       // Return
            .function::<fn() -> u64>()
    };

    match csr_func {
        Ok(_func) => {
            println!("   Function created successfully!");
            if cfg!(target_arch = "riscv64") {
                println!("   (CSR access requires appropriate privileges)");
            } else {
                println!("   (Skipping execution - not on RISC-V host)");
            }
        }
        Err(e) => println!("   Failed to create function: {}", e),
    }
}

#[cfg(feature = "aarch64")]
fn aarch64_examples() {
    println!("\n=== AArch64 Examples ===");

    // Example 1: Constant function
    println!("\n1. Creating a function that returns 42...");
    let constant_func = unsafe {
        Aarch64InstructionBuilder::new()
            .mov_imm(aarch64_reg::X0, 42)                  // Load 42 into X0 (return value)
            .ret()                                         // Return
            .function::<fn() -> u64>()
    };

    match constant_func {
        Ok(func) => {
            println!("   Function created successfully!");
            if cfg!(target_arch = "aarch64") {
                let result = func.call();
                println!("   Result: {}", result);
            } else {
                println!("   (Skipping execution - not on AArch64 host)");
            }
        }
        Err(e) => println!("   Failed to create function: {}", e),
    }

    // Example 2: Addition function
    println!("\n2. Creating a function that adds two numbers...");
    let add_func = unsafe {
        Aarch64InstructionBuilder::new()
            .add(aarch64_reg::X0, aarch64_reg::X0, aarch64_reg::X1)  // Add X0 + X1, result in X0
            .ret()                                                    // Return
            .function::<fn(u64, u64) -> u64>()
    };

    match add_func {
        Ok(func) => {
            println!("   Function created successfully!");
            if cfg!(target_arch = "aarch64") {
                let result = func.call(10, 20);
                println!("   10 + 20 = {}", result);
            } else {
                println!("   (Skipping execution - not on AArch64 host)");
            }
        }
        Err(e) => println!("   Failed to create function: {}", e),
    }

    // Example 3: More complex function with multiplication
    println!("\n3. Creating a function that computes (x + 100) * 2...");
    let complex_func = unsafe {
        Aarch64InstructionBuilder::new()
            .addi(aarch64_reg::X0, aarch64_reg::X0, 100)             // x + 100
            .mov_imm(aarch64_reg::X1, 2)                             // Load 2 into X1
            .mul(aarch64_reg::X0, aarch64_reg::X0, aarch64_reg::X1)  // Multiply by 2
            .ret()                                                    // Return
            .function::<fn(u64) -> u64>()
    };

    match complex_func {
        Ok(func) => {
            println!("   Function created successfully!");
            if cfg!(target_arch = "aarch64") {
                let result = func.call(5);
                println!("   (5 + 100) * 2 = {}", result);
            } else {
                println!("   (Skipping execution - not on AArch64 host)");
            }
        }
        Err(e) => println!("   Failed to create function: {}", e),
    }

    println!("\nAArch64 examples completed!");
    println!("Note: To actually execute these functions, run this example on an AArch64 system.");
}