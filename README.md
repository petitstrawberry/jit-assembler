# jit-assembler

A multi-architecture JIT assembler library for runtime code generation that works on any host architecture.

## Features

- **Multi-architecture support**: Generate machine code for different target architectures
- **Host-independent**: Runs on any host architecture (x86_64, ARM64, etc.) to generate target code
- **No-std compatible**: Works in both `std` and `no_std` environments
- **Type-safe**: Leverages Rust's type system for safe instruction generation
- **Dual API**: Both macro-based DSL and builder pattern for different use cases
- **IDE-friendly**: Full autocomplete and type checking support

## Supported Architectures

- **RISC-V 64-bit** (`riscv` feature, enabled by default)
- **x86-64** (`x86_64` feature) - Coming soon
- **ARM64** (`arm64` feature) - Coming soon

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
jit-assembler = "0.1"
```

### Basic Usage

```rust
use jit_assembler::riscv::{reg, csr, InstructionBuilder};
use jit_assembler::jit_asm;

// Macro style (concise and assembly-like)
let instructions = jit_asm! {
    csrrw(reg::X1, csr::MSTATUS, reg::X2);  // CSR read-write  
    csrr(reg::X4, csr::MSTATUS);            // CSR read (alias)
    addi(reg::X3, reg::X1, 100);            // Add immediate
    add(reg::X4, reg::X1, reg::X2);         // Register add
    beq(reg::X1, reg::X2, 8);               // Branch if equal
    jal(reg::X1, 0x1000);                   // Jump and link
};

// Method chaining style (recommended for programmatic use)
let mut builder = InstructionBuilder::new();
let instructions2 = builder
    .csrrw(reg::X1, csr::MSTATUS, reg::X2)  // CSR read-write
    .addi(reg::X3, reg::X1, 100)            // Add immediate
    .add(reg::X4, reg::X1, reg::X2)         // Register add
    .beq(reg::X1, reg::X2, 8)               // Branch if equal
    .jal(reg::X1, 0x1000)                   // Jump and link
    .instructions();

// Traditional style
let mut builder3 = InstructionBuilder::new();
builder3.csrrw(reg::X1, csr::MSTATUS, reg::X2);
builder3.addi(reg::X3, reg::X1, 100);
let instructions3 = builder3.instructions();

// Convert to bytes for execution
for instr in instructions {
    let bytes = instr.bytes();
    println!("Instruction: {} -> {:?}", instr, bytes);
}
```

### No-std Usage

For `no_std` environments, disable the default features:

```toml
[dependencies]
jit-assembler = { version = "0.1", default-features = false, features = ["riscv"] }
```

## Architecture Support

### RISC-V

The RISC-V backend supports:

- **CSR instructions**: `csrrw`, `csrrs`, `csrrc`, `csrrwi`, `csrrsi`, `csrrci`, `csrr` (read alias)
- **Arithmetic**: `add`, `sub`, `addi`, `xor`, `or`, `and`
- **Control flow**: `jal`, `jalr`, `beq`, `bne`, `blt`, `bge`, `bltu`, `bgeu`
- **Memory**: `ld`, `lw`, `lh`, `lb`, `sd`, `sw`, `sh`, `sb`
- **Shifts**: `sll`, `srl`, `sra`, `slli`, `srli`, `srai`
- **Upper immediates**: `lui`, `auipc`

### Future Architectures

Support for additional architectures is planned:

- x86-64: Intel/AMD 64-bit instruction set
- ARM64: AArch64 instruction set

## Examples

### JIT Compiler Integration

```rust
use jit_assembler::riscv::{reg, csr, InstructionBuilder};
use jit_assembler::jit_asm;

// Simple function generator with macro
fn generate_add_function(a: i16, b: i16) -> Vec<u8> {
    let instructions = jit_asm! {
        addi(reg::X1, reg::X0, a);    // Load first operand
        addi(reg::X2, reg::X0, b);    // Load second operand
        add(reg::X3, reg::X1, reg::X2); // Add them
        jalr(reg::X0, reg::X1, 0);    // Return
    };
    
    // Convert to bytes for execution
    let mut code = Vec::new();
    for instr in instructions {
        code.extend_from_slice(&instr.bytes());
    }
    code
}

// Builder pattern for complex logic
fn generate_csr_routine() -> Vec<u8> {
    let mut builder = InstructionBuilder::new();
    
    builder
        .csrr(reg::X1, csr::MSTATUS)     // Read current status
        .addi(reg::X2, reg::X1, 1)       // Modify value
        .csrrw(reg::X3, csr::MSTATUS, reg::X2); // Write back
    
    // Convert to executable code
    let mut code = Vec::new();
    for instr in builder.instructions() {
        code.extend_from_slice(&instr.bytes());
    }
    code
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.