# jit-assembler

A multi-architecture JIT assembler library for runtime code generation that works on any host architecture.

## Features

- **Multi-architecture support**: Generate machine code for different target architectures
- **Host-independent**: Runs on any host architecture (x86_64, ARM64, etc.) to generate target code
- **No-std compatible**: Works in both `std` and `no_std` environments
- **Type-safe**: Leverages Rust's type system for safe instruction generation
- **Dual API**: Both macro-based DSL and builder pattern for different use cases
- **IDE-friendly**: Full autocomplete and type checking support
- **JIT execution**: Direct execution of assembled code as functions (std-only)
- **Register usage tracking**: Analyze register usage patterns for optimization and ABI compliance (`register-tracking` feature)

## Supported Architectures

- **RISC-V 64-bit** (`riscv` feature, enabled by default)
- **AArch64** (`aarch64` feature, enabled by default) - Basic arithmetic and logical operations
- **x86-64** (`x86_64` feature) - Coming soon

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
jit-assembler = "0.1"
```

### Basic Usage

```rust
use jit_assembler::riscv::{reg, csr, Riscv64InstructionBuilder};
use jit_assembler::riscv64_asm;

// Macro style (concise and assembly-like)
let instructions = riscv64_asm! {
    csrrw(reg::RA, csr::MSTATUS, reg::SP);   // CSR read-write using aliases
    csrr(reg::T0, csr::MSTATUS);             // CSR read (alias)
    addi(reg::A0, reg::ZERO, 100);           // Add immediate using aliases
    add(reg::A1, reg::A0, reg::SP);          // Register add with aliases
    beq(reg::A0, reg::A1, 8);                // Branch if equal
    jal(reg::RA, 0x1000);                    // Jump and link
    ret();                                   // Return (alias for jalr x0, x1, 0)
};

// Method chaining style (recommended for programmatic use)
let mut builder = Riscv64InstructionBuilder::new();
let instructions2 = builder
    .csrrw(reg::RA, csr::MSTATUS, reg::SP)   // CSR read-write using aliases
    .addi(reg::A0, reg::ZERO, 100)           // Add immediate with aliases
    .add(reg::A1, reg::A0, reg::SP)          // Register add using aliases
    .beq(reg::A0, reg::A1, 8)                // Branch if equal
    .jal(reg::RA, 0x1000)                    // Jump and link
    .ret()                                   // Return instruction
    .instructions();

// Traditional style
let mut builder3 = Riscv64InstructionBuilder::new();
builder3.csrrw(reg::RA, csr::MSTATUS, reg::SP);
builder3.addi(reg::A0, reg::ZERO, 100);
builder3.ret();
let instructions3 = builder3.instructions();

// Convert instructions to bytes easily
let bytes = instructions.to_bytes();     // All instructions as one byte vector
let size = instructions.total_size();    // Total size in bytes
let count = instructions.len();          // Number of instructions

// Iterate over instructions
for (i, instr) in instructions.iter().enumerate() {
    println!("Instruction {}: {} -> {:?}", i, instr, instr.bytes());
}

// Or access by index
let first_instr = instructions[0];
```

### No-std Usage

For `no_std` environments, disable the default features:

```toml
[dependencies]
jit-assembler = { version = "0.1", default-features = false, features = ["riscv"] }
# Or for AArch64 only:
# jit-assembler = { version = "0.1", default-features = false, features = ["aarch64"] }
# Or for both architectures without std:
# jit-assembler = { version = "0.1", default-features = false, features = ["riscv", "aarch64"] }
```

## Architecture Support

### RISC-V

The RISC-V backend supports:

- **Base integer instruction set (RV64I)**:
  - **Arithmetic**: `add`, `sub`, `addi`, `xor`, `or`, `and`, `slt`, `sltu`
  - **Immediate arithmetic**: `andi`, `ori`, `xori`, `slti`, `sltiu`
  - **Shifts**: `sll`, `srl`, `sra`, `slli`, `srli`, `srai`
  - **Upper immediates**: `lui`, `auipc`
- **M extension (Integer Multiplication and Division)**:
  - **Multiply**: `mul`, `mulh`, `mulhsu`, `mulhu`
  - **Divide**: `div`, `divu`, `rem`, `remu`
- **Memory operations**:
  - **Loads (signed)**: `ld`, `lw`, `lh`, `lb`
  - **Loads (unsigned)**: `lbu`, `lhu`, `lwu`
  - **Stores**: `sd`, `sw`, `sh`, `sb`
- **Control flow**: `jal`, `jalr`, `beq`, `bne`, `blt`, `bge`, `bltu`, `bgeu`
- **CSR instructions**: `csrrw`, `csrrs`, `csrrc`, `csrrwi`, `csrrsi`, `csrrci`
- **CSR pseudo-instructions**: `csrr` (read), `csrw` (write), `csrs` (set), `csrc` (clear), `csrwi`, `csrsi`, `csrci`
- **Privileged instructions**: `sret`, `mret`, `ecall`, `ebreak`, `wfi`
- **Pseudo-instructions**: `ret`, `li`
- **Register usage tracking**: Full tracking support for all instruction types (`register-tracking` feature)

### AArch64

The AArch64 backend supports:

- **Basic arithmetic operations**:
  - **Register operations**: `add`, `sub`, `mul`, `udiv`, `sdiv`
  - **Immediate operations**: `addi`, `subi`
  - **Remainder operations**: `urem` (unsigned remainder)
- **Logical operations**:
  - **Register operations**: `and`, `or`, `xor` (EOR)
  - **Move operations**: `mov`
- **Control flow**:
  - **Return**: `ret`, `ret_reg`
- **Extended operations**:
  - **Immediate moves**: `mov_imm` (for larger constants)
  - **Shift operations**: `shl` (left shift using multiply)
- **Register conventions**: Following AAPCS64 (ARM ABI)
  - **Argument/return registers**: X0-X7
  - **Caller-saved temporaries**: X8-X18
  - **Callee-saved registers**: X19-X28
  - **Special registers**: X29 (FP), X30 (LR), X31 (SP/XZR)
- **Register usage tracking**: Full tracking support (`register-tracking` feature)
- **JIT compilation**: Direct function compilation and execution

### Future Architectures

Support for additional architectures is planned:

- x86-64: Intel/AMD 64-bit instruction set

## Examples

### JIT Compiler Integration

```rust
use jit_assembler::riscv::{reg, csr, Riscv64InstructionBuilder};
use jit_assembler::riscv64_asm;

// Simple function generator with macro
fn generate_add_function(a: i16, b: i16) -> Vec<u8> {
    let instructions = riscv64_asm! {
        addi(reg::A0, reg::ZERO, a);       // Load first operand into a0
        addi(reg::A1, reg::ZERO, b);       // Load second operand into a1
        add(reg::A0, reg::A0, reg::A1);    // Add them, result in a0
        ret();                             // Return
    };
    
    // Convert to bytes for execution
    instructions.to_bytes()
}

// Builder pattern for complex logic
fn generate_csr_routine() -> Vec<u8> {
    let mut builder = Riscv64InstructionBuilder::new();
    
    builder
        .csrr(reg::T0, csr::MSTATUS)         // Read current status into t0
        .addi(reg::T1, reg::T0, 1)           // Modify value in t1
        .csrrw(reg::A0, csr::MSTATUS, reg::T1); // Write back, old value in a0
    
    // Convert to executable code
    builder.instructions().to_bytes()
}
```

### AArch64 Usage

```rust
use jit_assembler::aarch64::{reg, Aarch64InstructionBuilder};
use jit_assembler::common::InstructionBuilder;

// Create an AArch64 function that adds two numbers
fn generate_aarch64_add_function() -> Vec<u8> {
    let mut builder = Aarch64InstructionBuilder::new();
    
    builder
        .add(reg::X0, reg::X0, reg::X1)  // Add first two arguments (X0 + X1 -> X0)
        .ret();                          // Return
    
    builder.instructions().to_bytes()
}

// More complex AArch64 example with immediate values
fn generate_aarch64_calculation() -> Vec<u8> {
    let mut builder = Aarch64InstructionBuilder::new();
    
    builder
        .mov_imm(reg::X1, 42)            // Load immediate 42 into X1
        .mul(reg::X0, reg::X0, reg::X1)  // Multiply X0 by 42
        .addi(reg::X0, reg::X0, 100)     // Add 100 to result
        .ret();                          // Return
    
    builder.instructions().to_bytes()
}
```

### JIT Execution (std-only)

Create and execute functions directly at runtime:

```rust
use jit_assembler::riscv::{reg, Riscv64InstructionBuilder};
use jit_assembler::common::InstructionBuilder;

// Create a JIT function that adds two numbers
let add_func = unsafe {
    Riscv64InstructionBuilder::new()
        .add(reg::A0, reg::A0, reg::A1) // Add first two arguments
        .ret()                          // Return result
        .function::<fn(u64, u64) -> u64>()
}.expect("Failed to create JIT function");

// Call the JIT function naturally - just like a regular function!
let result = add_func.call(10, 20);
assert_eq!(result, 30);

// Create a function that returns a constant
let constant_func = unsafe {
    Riscv64InstructionBuilder::new()
        .addi(reg::A0, reg::ZERO, 42)  // Load 42 into return register
        .ret()                         // Return
        .function::<fn() -> u64>()
}.expect("Failed to create JIT function");

let result = constant_func.call();
assert_eq!(result, 42);
```

**Note**: JIT execution requires the target architecture to match the host architecture. RISC-V code will only execute correctly on RISC-V systems.

**Features**:
- Type-safe function signatures
- Automatic memory management with `jit-allocator2`
- Natural function call syntax: `func.call()`, `func.call(arg)`, `func.call(arg1, arg2)`, etc. - just like regular functions!
- Cross-platform executable memory allocation

## Register Usage Tracking

The `register-tracking` feature enables comprehensive analysis of register usage patterns in your JIT-compiled code, helping with optimization and ABI compliance.

### Enable Register Tracking

Add the feature to your `Cargo.toml`:

```toml
[dependencies]
jit-assembler = { version = "0.1", features = ["register-tracking"] }
```

### Usage Example

```rust
use jit_assembler::riscv::{reg, Riscv64InstructionBuilder};
use jit_assembler::common::InstructionBuilder;

let mut builder = Riscv64InstructionBuilder::new();

// Build a function that uses various registers
builder
    .add(reg::T0, reg::T1, reg::T2)     // T0 written, T1+T2 read
    .addi(reg::T3, reg::SP, 16)         // T3 written, SP read
    .mul(reg::A0, reg::A1, reg::A2)     // A0 written, A1+A2 read
    .ld(reg::S0, reg::T0, 8)            // S0 written, T0 read
    .sd(reg::SP, reg::S1, -16);         // SP+S1 read

// Analyze register usage
let usage = builder.register_usage();

println!("=== Register Usage Analysis ===");
println!("Total registers used: {}", usage.register_count());
println!("Written registers: {:?}", usage.written_registers());
println!("Read registers: {:?}", usage.read_registers());

// ABI compliance analysis
println!("Caller-saved (written): {:?}", usage.caller_saved_written());
println!("Callee-saved (written): {:?}", usage.callee_saved_written());
println!("Needs stack frame: {}", usage.needs_stack_frame());

// Detailed breakdown
let (caller, callee, special) = usage.count_by_abi_class();
println!("ABI breakdown - Caller: {}, Callee: {}, Special: {}", 
         caller, callee, special);
```

### Key Features

- **Separate tracking**: Distinguishes between written (def) and read (use) registers
- **ABI classification**: Automatically categorizes registers as caller-saved, callee-saved, or special-purpose
- **Stack frame analysis**: Determines if function prologue/epilogue is needed based on callee-saved register usage
- **Comprehensive coverage**: Tracks all RISC-V instruction types (R, I, S, B, U, J, CSR)
- **No-std compatible**: Uses `hashbrown` for no-std environments

### Register ABI Classification (RISC-V)

- **Caller-saved**: T0-T6, A0-A7, RA - Can be freely used without preservation
- **Callee-saved**: S0-S11, SP - Must be saved/restored if modified
- **Special**: X0 (zero), GP, TP - Require careful handling

This information is invaluable for:
- **Register allocation**: Choose optimal registers for variables
- **ABI compliance**: Ensure proper calling convention adherence
- **Performance optimization**: Minimize unnecessary register saves/restores
- **Code analysis**: Understand register pressure and usage patterns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.