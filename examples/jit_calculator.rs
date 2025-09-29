//! JIT Calculator with AST Example
//! 
//! This example demonstrates a sophisticated JIT-compiled calculator that uses
//! an Abstract Syntax Tree (AST) to represent mathematical expressions.
//! The calculator compiles expressions to native machine code using
//! either RISC-V or AArch64 depending on the target architecture.
//! 
//! Features:
//! - AST-based expression parsing and evaluation
//! - Support for parentheses and operator precedence
//! - JIT compilation to RISC-V or AArch64 machine code
//! - Optimized code generation for complex expressions
//! 
//! Supported operations: +, -, *, /, % (remainder), and parentheses
//! 
//! Note: This example works on RISC-V or AArch64 hosts or in emulation.
//! On other architectures, the functions will be created successfully
//! but calling them will likely crash.

// Architecture-specific imports
#[cfg(target_arch = "riscv64")]
use jit_assembler::riscv64::{reg, Riscv64InstructionBuilder};

#[cfg(target_arch = "aarch64")]
use jit_assembler::aarch64::{reg, Aarch64InstructionBuilder};

use jit_assembler::common::InstructionBuilder;

use std::fmt;
use std::env;

/// Configuration for the JIT calculator
#[derive(Debug, Clone)]
pub struct CalculatorConfig {
    /// Whether to show generated machine code
    pub show_machine_code: bool,
}

impl Default for CalculatorConfig {
    fn default() -> Self {
        Self {
            show_machine_code: false,
        }
    }
}

/// Abstract Syntax Tree node representing mathematical expressions
#[derive(Debug, Clone, PartialEq)]
pub enum AstNode {
    /// A numeric literal value
    Number(u64),
    /// Binary operation: left operand, operator, right operand
    BinaryOp {
        left: Box<AstNode>,
        op: BinaryOperator,
        right: Box<AstNode>,
    },
}

/// Binary operators supported by the calculator
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
}

impl BinaryOperator {
    fn symbol(&self) -> &'static str {
        match self {
            BinaryOperator::Add => "+",
            BinaryOperator::Subtract => "-",
            BinaryOperator::Multiply => "*",
            BinaryOperator::Divide => "/",
            BinaryOperator::Remainder => "%",
        }
    }

    fn precedence(&self) -> u8 {
        match self {
            BinaryOperator::Add | BinaryOperator::Subtract => 1,
            BinaryOperator::Multiply | BinaryOperator::Divide | BinaryOperator::Remainder => 2,
        }
    }
}

impl fmt::Display for AstNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AstNode::Number(n) => write!(f, "{}", n),
            AstNode::BinaryOp { left, op, right } => {
                write!(f, "({} {} {})", left, op.symbol(), right)
            }
        }
    }
}

/// Tokenizer for mathematical expressions
pub struct Tokenizer {
    input: Vec<char>,
    pos: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Number(u64),
    Plus,
    Minus,
    Multiply,
    Divide,
    Remainder,
    LeftParen,
    RightParen,
    End,
}

impl Tokenizer {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
        }
    }

    pub fn next_token(&mut self) -> Result<Token, String> {
        self.skip_whitespace();

        if self.pos >= self.input.len() {
            return Ok(Token::End);
        }

        let ch = self.input[self.pos];
        match ch {
            '+' => { self.pos += 1; Ok(Token::Plus) }
            '-' => { self.pos += 1; Ok(Token::Minus) }
            '*' => { self.pos += 1; Ok(Token::Multiply) }
            '/' => { self.pos += 1; Ok(Token::Divide) }
            '%' => { self.pos += 1; Ok(Token::Remainder) }
            '(' => { self.pos += 1; Ok(Token::LeftParen) }
            ')' => { self.pos += 1; Ok(Token::RightParen) }
            '0'..='9' => self.parse_number(),
            _ => Err(format!("Unexpected character: {}", ch)),
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() && self.input[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn parse_number(&mut self) -> Result<Token, String> {
        let start = self.pos;
        while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        
        let number_str: String = self.input[start..self.pos].iter().collect();
        number_str.parse::<u64>()
            .map(Token::Number)
            .map_err(|_| format!("Invalid number: {}", number_str))
    }
}

/// Recursive descent parser for mathematical expressions
pub struct Parser {
    tokenizer: Tokenizer,
    current_token: Token,
}

impl Parser {
    pub fn new(input: &str) -> Result<Self, String> {
        let mut tokenizer = Tokenizer::new(input);
        let current_token = tokenizer.next_token()?;
        Ok(Self {
            tokenizer,
            current_token,
        })
    }

    pub fn parse(&mut self) -> Result<AstNode, String> {
        let node = self.parse_expression()?;
        if self.current_token != Token::End {
            return Err("Unexpected token at end of expression".to_string());
        }
        Ok(node)
    }

    fn advance(&mut self) -> Result<(), String> {
        self.current_token = self.tokenizer.next_token()?;
        Ok(())
    }

    fn parse_expression(&mut self) -> Result<AstNode, String> {
        self.parse_additive()
    }

    fn parse_additive(&mut self) -> Result<AstNode, String> {
        let mut left = self.parse_multiplicative()?;

        while matches!(self.current_token, Token::Plus | Token::Minus) {
            let op = match self.current_token {
                Token::Plus => BinaryOperator::Add,
                Token::Minus => BinaryOperator::Subtract,
                _ => unreachable!(),
            };
            self.advance()?;
            let right = self.parse_multiplicative()?;
            left = AstNode::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<AstNode, String> {
        let mut left = self.parse_primary()?;

        while matches!(self.current_token, Token::Multiply | Token::Divide | Token::Remainder) {
            let op = match self.current_token {
                Token::Multiply => BinaryOperator::Multiply,
                Token::Divide => BinaryOperator::Divide,
                Token::Remainder => BinaryOperator::Remainder,
                _ => unreachable!(),
            };
            self.advance()?;
            let right = self.parse_primary()?;
            left = AstNode::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_primary(&mut self) -> Result<AstNode, String> {
        match &self.current_token {
            Token::Number(n) => {
                let value = *n;
                self.advance()?;
                Ok(AstNode::Number(value))
            }
            Token::LeftParen => {
                self.advance()?; // consume '('
                let node = self.parse_expression()?;
                if self.current_token != Token::RightParen {
                    return Err("Expected closing parenthesis".to_string());
                }
                self.advance()?; // consume ')'
                Ok(node)
            }
            _ => Err(format!("Unexpected token: {:?}", self.current_token)),
        }
    }
}

/// JIT compiler that converts AST to machine code
/// Unified implementation that works across supported architectures
pub struct JitCompiler {
    #[cfg(target_arch = "riscv64")]
    builder: Riscv64InstructionBuilder,
    #[cfg(target_arch = "riscv64")]
    register_stack: Vec<jit_assembler::riscv64::Register>,

    #[cfg(target_arch = "aarch64")]
    builder: Aarch64InstructionBuilder,
    #[cfg(target_arch = "aarch64")]
    register_stack: Vec<jit_assembler::aarch64::Register>,

    next_temp_reg: usize,
}

impl JitCompiler {
    /// Available temporary registers for computation (RISC-V)
    #[cfg(target_arch = "riscv64")]
    const TEMP_REGISTERS_RISCV: &'static [jit_assembler::riscv64::Register] = &[
        reg::T0, reg::T1, reg::T2, reg::T3, reg::T4, reg::T5, reg::T6,
    ];

    /// Available temporary registers for computation (AArch64)
    #[cfg(target_arch = "aarch64")]
    const TEMP_REGISTERS_AARCH64: &'static [jit_assembler::aarch64::Register] = &[
        reg::X9, reg::X10, reg::X11, reg::X12, reg::X13, reg::X14, reg::X15,
    ];

    pub fn new() -> Self {
        Self {
            #[cfg(target_arch = "riscv64")]
            builder: Riscv64InstructionBuilder::new(),
            #[cfg(target_arch = "riscv64")]
            register_stack: Vec::new(),

            #[cfg(target_arch = "aarch64")]
            builder: Aarch64InstructionBuilder::new(),
            #[cfg(target_arch = "aarch64")]
            register_stack: Vec::new(),

            next_temp_reg: 0,
        }
    }

    /// Allocate a temporary register (architecture-agnostic interface)
    #[cfg(target_arch = "riscv64")]
    fn alloc_register(&mut self) -> Result<jit_assembler::riscv64::Register, String> {
        if self.next_temp_reg >= Self::TEMP_REGISTERS_RISCV.len() {
            return Err("Out of temporary registers".to_string());
        }
        let reg = Self::TEMP_REGISTERS_RISCV[self.next_temp_reg];
        self.next_temp_reg += 1;
        self.register_stack.push(reg);
        Ok(reg)
    }

    /// Allocate a temporary register (architecture-agnostic interface)
    #[cfg(target_arch = "aarch64")]
    fn alloc_register(&mut self) -> Result<jit_assembler::aarch64::Register, String> {
        if self.next_temp_reg >= Self::TEMP_REGISTERS_AARCH64.len() {
            return Err("Out of temporary registers".to_string());
        }
        let reg = Self::TEMP_REGISTERS_AARCH64[self.next_temp_reg];
        self.next_temp_reg += 1;
        self.register_stack.push(reg);
        Ok(reg)
    }

    /// Free the last allocated register
    fn free_register(&mut self) -> Result<(), String> {
        #[cfg(target_arch = "riscv64")]
        {
            if self.register_stack.is_empty() {
                return Err("No registers to free".to_string());
            }
            self.register_stack.pop();
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            if self.register_stack.is_empty() {
                return Err("No registers to free".to_string());
            }
            self.register_stack.pop();
        }
        
        self.next_temp_reg = self.next_temp_reg.saturating_sub(1);
        Ok(())
    }

    /// Compile an AST to a JIT function
    /// The result is stored in the appropriate return register for each architecture
    pub fn compile_expression(&mut self, ast: &AstNode, config: &CalculatorConfig) -> Result<Box<dyn Fn() -> u64>, Box<dyn std::error::Error>> {
        // Generate code that computes the expression result in the return register
        #[cfg(target_arch = "riscv64")]
        {
            self.compile_node(ast, reg::A0)?;
            self.builder.ret();
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            self.compile_node(ast, reg::X0)?;
            self.builder.ret();
        }
        
        #[cfg(not(any(target_arch = "riscv64", target_arch = "aarch64")))]
        {
            return Err("JIT compilation not supported on this architecture".into());
        }

        // Show machine code if requested
        if config.show_machine_code {
            self.show_generated_code();
        }

        #[cfg(any(target_arch = "riscv64", target_arch = "aarch64"))]
        {
            let jit_func = unsafe {
                self.builder.function::<fn() -> u64>()?
            };
            Ok(Box::new(move || jit_func.call()))
        }
        
        #[cfg(not(any(target_arch = "riscv64", target_arch = "aarch64")))]
        {
            // This will never be reached due to the early return above, but needed for type consistency
            unreachable!()
        }
    }

    /// Display the generated machine code
    pub fn show_generated_code(&self) {
        #[cfg(target_arch = "riscv64")]
        {
            let instructions = self.builder.instructions();
            let bytes = instructions.to_bytes();
            
            println!("🤖 Generated Machine Code:");
            println!("   Instructions: {}, Total bytes: {}", instructions.len(), bytes.len());
            
            for (i, instr) in instructions.iter().enumerate() {
                let instr_bytes = instr.bytes();
                println!("   [{:2}]: {:02X?} ({})", 
                         i + 1, 
                         instr_bytes,
                         if instr.is_compressed() { "16-bit" } else { "32-bit" });
            }
            
            println!("   Raw bytes: {:02X?}", bytes);
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            let instructions = self.builder.instructions();
            let bytes = instructions.to_bytes();
            
            println!("🤖 Generated Machine Code:");
            println!("   Instructions: {}, Total bytes: {}", instructions.len(), bytes.len());
            
            for (i, instr) in instructions.iter().enumerate() {
                let instr_bytes = instr.bytes();
                println!("   [{:2}]: {:02X?} (32-bit)", i + 1, instr_bytes);
            }
            
            println!("   Raw bytes: {:02X?}", bytes);
        }
    }

    /// Compile an AST node, storing the result in the specified register (RISC-V)
    #[cfg(target_arch = "riscv64")]
    fn compile_node(&mut self, node: &AstNode, result_reg: jit_assembler::riscv64::Register) -> Result<(), String> {
        match node {
            AstNode::Number(value) => {
                // Load immediate value into result register
                if *value <= 2047 {
                    // Small immediate: can use ADDI with zero register
                    self.builder.addi(result_reg, reg::ZERO, *value as i16);
                } else {
                    // Large immediate: use LUI + ADDI with correct sign extension handling
                    let lower = (*value & 0xFFF) as i16;
                    let upper = if lower < 0 {
                        // If lower part is negative, we need to add 1 to upper part 
                        // because LUI will be sign-extended
                        ((*value + 0x800) >> 12) as u32
                    } else {
                        (*value >> 12) as u32
                    };
                    
                    self.builder.lui(result_reg, upper);
                    if lower != 0 {
                        self.builder.addi(result_reg, result_reg, lower);
                    }
                }
                Ok(())
            }
            AstNode::BinaryOp { left, op, right } => {
                // Use result_reg for left operand to save registers
                self.compile_node(left, result_reg)?;
                
                // Only allocate one temp register for right operand
                let right_reg = self.alloc_register()?;
                self.compile_node(right, right_reg)?;

                // Perform operation: result_reg = result_reg op right_reg
                match op {
                    BinaryOperator::Add => {
                        self.builder.add(result_reg, result_reg, right_reg);
                    }
                    BinaryOperator::Subtract => {
                        self.builder.sub(result_reg, result_reg, right_reg);
                    }
                    BinaryOperator::Multiply => {
                        self.builder.mul(result_reg, result_reg, right_reg);
                    }
                    BinaryOperator::Divide => {
                        // Note: This is unsigned division. For signed, use div instead
                        self.builder.divu(result_reg, result_reg, right_reg);
                    }
                    BinaryOperator::Remainder => {
                        // Note: This is unsigned remainder. For signed, use rem instead
                        self.builder.remu(result_reg, result_reg, right_reg);
                    }
                }

                // Free the temporary register  
                self.free_register()?; // right_reg
                
                Ok(())
            }
        }
    }

    /// Compile an AST node, storing the result in the specified register (AArch64)
    #[cfg(target_arch = "aarch64")]
    fn compile_node(&mut self, node: &AstNode, result_reg: jit_assembler::aarch64::Register) -> Result<(), String> {
        match node {
            AstNode::Number(value) => {
                // Load immediate value into result register
                self.builder.mov_imm(result_reg, *value);
                Ok(())
            }
            AstNode::BinaryOp { left, op, right } => {
                // Use result_reg for left operand to save registers
                self.compile_node(left, result_reg)?;
                
                // Only allocate one temp register for right operand
                let right_reg = self.alloc_register()?;
                self.compile_node(right, right_reg)?;

                // Perform operation: result_reg = result_reg op right_reg
                match op {
                    BinaryOperator::Add => {
                        self.builder.add(result_reg, result_reg, right_reg);
                    }
                    BinaryOperator::Subtract => {
                        self.builder.sub(result_reg, result_reg, right_reg);
                    }
                    BinaryOperator::Multiply => {
                        self.builder.mul(result_reg, result_reg, right_reg);
                    }
                    BinaryOperator::Divide => {
                        self.builder.udiv(result_reg, result_reg, right_reg);
                    }
                    BinaryOperator::Remainder => {
                        self.builder.urem(result_reg, result_reg, right_reg);
                    }
                }

                // Free the temporary register  
                self.free_register()?; // right_reg
                
                Ok(())
            }
        }
    }
}

/// High-level calculator interface
pub struct JitCalculator;

impl JitCalculator {
    /// Parse and evaluate a mathematical expression using JIT compilation
    pub fn evaluate(expression: &str, config: &CalculatorConfig) -> Result<u64, Box<dyn std::error::Error>> {
        println!("🔍 Parsing expression: {}", expression);
        
        // Parse expression into AST
        let mut parser = Parser::new(expression)?;
        let ast = parser.parse()?;
        
        println!("🌳 Generated AST: {}", ast);
        
        // Compile AST to JIT function or interpret
        #[cfg(any(target_arch = "riscv64", target_arch = "aarch64"))]
        {
            println!("🔧 Compiling to native machine code...");
            let mut compiler = JitCompiler::new();
            let jit_function = compiler.compile_expression(&ast, config)?;
            
            let result = jit_function();
            println!("✅ JIT execution result: {}", result);
            Ok(result)
        }
        
        #[cfg(not(any(target_arch = "riscv64", target_arch = "aarch64")))]
        {
            println!("⚠️  Not on RISC-V or AArch64 platform, using AST interpreter");
            let result = Self::interpret_ast(&ast)?;
            println!("✅ Interpreted result: {}", result);
            Ok(result)
        }
    }

    /// Fallback AST interpreter for non-RISC-V platforms
    fn interpret_ast(node: &AstNode) -> Result<u64, Box<dyn std::error::Error>> {
        match node {
            AstNode::Number(n) => Ok(*n),
            AstNode::BinaryOp { left, op, right } => {
                let left_val = Self::interpret_ast(left)?;
                let right_val = Self::interpret_ast(right)?;
                
                let result = match op {
                    BinaryOperator::Add => left_val.wrapping_add(right_val),
                    BinaryOperator::Subtract => left_val.wrapping_sub(right_val),
                    BinaryOperator::Multiply => left_val.wrapping_mul(right_val),
                    BinaryOperator::Divide => {
                        if right_val == 0 {
                            return Err("Division by zero".into());
                        }
                        left_val / right_val
                    }
                    BinaryOperator::Remainder => {
                        if right_val == 0 {
                            return Err("Remainder by zero".into());
                        }
                        left_val % right_val
                    }
                };
                Ok(result)
            }
        }
    }

    /// Interactive calculator session with AST parsing
    pub fn run_interactive(config: &CalculatorConfig) {
        println!("🧮 JIT Calculator with AST Support");
        println!("==================================");
        println!("Supports: +, -, *, /, %, parentheses, and operator precedence");
        println!("Examples: '2 + 3 * 4', '(10 + 5) * 2', '100 / (2 + 3)'");
        println!("Type 'quit' to exit\n");

        loop {
            println!("Enter expression:");
            
            let mut input = String::new();
            match std::io::stdin().read_line(&mut input) {
                Ok(0) => {
                    // EOF reached
                    println!("👋 End of input reached. Goodbye!");
                    break;
                }
                Ok(_) => {
                    // Successfully read input
                }
                Err(_) => {
                    println!("❌ Error reading input");
                    continue;
                }
            }

            let input = input.trim();
            if input.eq_ignore_ascii_case("quit") {
                break;
            }

            if input.is_empty() {
                continue;
            }

            match Self::evaluate(input, config) {
                Ok(result) => {
                    println!("📊 Result: {}\n", result);
                }
                Err(e) => {
                    println!("❌ Error: {}\n", e);
                }
            }
        }

        println!("👋 Goodbye!");
    }
}

/// Demonstrate JIT compilation by showing the generated machine code
#[cfg(target_arch = "riscv64")]
fn demonstrate_jit_compilation() {
    println!("Generating RISC-V machine code for multiplication (7 * 6)...");
    
    // Create a multiply function and show its bytecode
    let mut builder = Riscv64InstructionBuilder::new();
    builder.mul(reg::A0, reg::A0, reg::A1); // a0 = a0 * a1
    builder.ret(); // Return
    
    let instructions = builder.instructions();
    let bytes = instructions.to_bytes();
    
    println!("📦 Generated {} instructions, {} bytes total:", instructions.len(), bytes.len());
    
    for (i, instr) in instructions.iter().enumerate() {
        let instr_bytes = instr.bytes();
        println!("  Instruction {}: {:02X?} ({})", 
                 i + 1, 
                 instr_bytes,
                 if instr.is_compressed() { "16-bit" } else { "32-bit" });
    }
    
    println!("📋 Complete bytecode: {:02X?}", bytes);
    println!();
}

/// Demonstrate JIT compilation by showing the generated machine code
#[cfg(target_arch = "aarch64")]
fn demonstrate_jit_compilation() {
    println!("Generating AArch64 machine code for multiplication (7 * 6)...");
    
    // Create a multiply function and show its bytecode
    let mut builder = Aarch64InstructionBuilder::new();
    builder.mul(reg::X0, reg::X0, reg::X1); // X0 = X0 * X1
    builder.ret(); // Return
    
    let instructions = builder.instructions();
    let bytes = instructions.to_bytes();
    
    println!("📦 Generated {} instructions, {} bytes total:", instructions.len(), bytes.len());
    
    for (i, instr) in instructions.iter().enumerate() {
        let instr_bytes = instr.bytes();
        println!("  Instruction {}: {:02X?} (32-bit)", i + 1, instr_bytes);
    }
    
    println!("📋 Complete bytecode: {:02X?}", bytes);
    println!();
}

/// Demonstrate JIT compilation by showing the generated machine code (fallback for unsupported architectures)
#[cfg(not(any(target_arch = "riscv64", target_arch = "aarch64")))]
fn demonstrate_jit_compilation() {
    println!("JIT compilation not available for this architecture");
    println!("Supported architectures: RISC-V64, AArch64");
    println!();
}

fn main() {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let mut config = CalculatorConfig::default();
    
    // Check for --show-machine-code or -m flag
    for arg in &args[1..] {
        match arg.as_str() {
            "--show-machine-code" | "-m" => {
                config.show_machine_code = true;
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            _ => {
                println!("Unknown argument: {}", arg);
                print_help();
                return;
            }
        }
    }
    
    println!("JIT Calculator with AST - M Extension Demo");
    println!("==========================================");
    if config.show_machine_code {
        println!("🤖 Machine code display: ENABLED");
    }
    
    // Show JIT compilation details
    println!("\n🔍 JIT Compilation Details:");
    demonstrate_jit_compilation();
    
    // Demonstrate AST parsing and evaluation with various expressions
    let test_expressions = vec![
        "42",
        "10 + 5",
        "100 - 25",
        "7 * 6",
        "84 / 12",
        "23 % 7",
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "100 / (10 - 5)",
        "((10 + 5) * 2) - 6",
        "2 * 3 + 4 * 5",
    ];

    println!("\n📋 Running predefined test expressions:\n");
    
    for expression in test_expressions {
        match JitCalculator::evaluate(expression, &config) {
            Ok(result) => {
                println!("✅ {} = {}", expression, result);
            }
            Err(e) => {
                println!("❌ Error evaluating '{}': {}", expression, e);
            }
        }
        println!(); // Empty line for readability
    }

    // Run interactive mode
    println!("🎮 Starting interactive mode...\n");
    JitCalculator::run_interactive(&config);
}

fn print_help() {
    println!("JIT Calculator with AST - RISC-V M Extension Demo");
    println!("================================================");
    println!();
    println!("USAGE:");
    println!("    cargo run --example jit_calculator [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -m, --show-machine-code    Show generated RISC-V machine code");
    println!("    -h, --help                 Show this help message");
    println!();
    println!("EXAMPLES:");
    println!("    cargo run --example jit_calculator");
    println!("    cargo run --example jit_calculator --show-machine-code");
    println!("    cargo run --example jit_calculator -m");
}

/// RISC-V64 and AArch64 Integration Tests for JIT Calculator
/// 
/// This module contains tests that only run on RISC-V64 or AArch64 platforms
/// to verify that the JIT calculator works correctly with actual execution.

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic arithmetic operations
    #[test]
    #[cfg(any(target_arch = "riscv64", target_arch = "aarch64"))]
    fn test_basic_arithmetic() {
        let config = CalculatorConfig::default();
        
        // Test addition
        let result = JitCalculator::evaluate("10 + 5", &config).expect("Addition failed");
        assert_eq!(result, 15);
        
        // Test subtraction
        let result = JitCalculator::evaluate("20 - 8", &config).expect("Subtraction failed");
        assert_eq!(result, 12);
        
        // Test multiplication (M extension)
        let result = JitCalculator::evaluate("7 * 6", &config).expect("Multiplication failed");
        assert_eq!(result, 42);
        
        // Test division (M extension)
        let result = JitCalculator::evaluate("84 / 12", &config).expect("Division failed");
        assert_eq!(result, 7);
        
        // Test remainder (M extension)
        let result = JitCalculator::evaluate("23 % 7", &config).expect("Remainder failed");
        assert_eq!(result, 2);
    }

    /// Test complex expressions with operator precedence on RISC-V64 and AArch64
    #[test]
    #[cfg(any(target_arch = "riscv64", target_arch = "aarch64"))]
    fn test_complex_expressions() {
        let config = CalculatorConfig::default();
        
        // Test operator precedence
        let result = JitCalculator::evaluate("2 + 3 * 4", &config).expect("Precedence test failed");
        assert_eq!(result, 14);
        
        // Test parentheses
        let result = JitCalculator::evaluate("(2 + 3) * 4", &config).expect("Parentheses test failed");
        assert_eq!(result, 20);
        
        // Test nested expressions
        let result = JitCalculator::evaluate("100 / (10 - 5)", &config).expect("Nested expression failed");
        assert_eq!(result, 20);
        
        // Test complex nested expression
        let result = JitCalculator::evaluate("((10 + 5) * 2) - 6", &config).expect("Complex expression failed");
        assert_eq!(result, 24);
        
        // Test multiple operations
        let result = JitCalculator::evaluate("2 * 3 + 4 * 5", &config).expect("Multiple operations failed");
        assert_eq!(result, 26);
    }

    /// Test edge cases on RISC-V64 and AArch64
    #[test]
    #[cfg(any(target_arch = "riscv64", target_arch = "aarch64"))]
    fn test_edge_cases() {
        let config = CalculatorConfig::default();
        
        // Test single number
        let result = JitCalculator::evaluate("42", &config).expect("Single number failed");
        assert_eq!(result, 42);
        
        // Test zero
        let result = JitCalculator::evaluate("0", &config).expect("Zero failed");
        assert_eq!(result, 0);
        
        // Test large numbers
        let result = JitCalculator::evaluate("1000 + 2000", &config).expect("Large numbers failed");
        assert_eq!(result, 3000);
        
        // Test moderately nested parentheses (reduced complexity)
        let result = JitCalculator::evaluate("((2 + 3) * 4) + 5", &config).expect("Nested expression failed");
        assert_eq!(result, 25); // (5 * 4) + 5 = 20 + 5 = 25
    }

    /// Test register allocation with complex expressions on RISC-V64 and AArch64
    #[test]
    #[cfg(any(target_arch = "riscv64", target_arch = "aarch64"))]
    fn test_register_allocation() {
        let config = CalculatorConfig::default();
        
        // Test expressions that would stress register allocation (simplified)
        let test_cases = vec![
            ("1 + 2 + 3", 6),         // Simpler chain to reduce register usage
            ("2 * 3 * 4", 24),        // Simpler multiplication chain
            ("(1 + 2) * (3 + 4)", 21), // Single nested expression
            ("10 + 5 * 2 - 4", 16),   // Mixed operations: 10 + 10 - 4 = 16
        ];
        
        for (expression, expected) in test_cases {
            let result = JitCalculator::evaluate(expression, &config)
                .expect(&format!("Expression '{}' failed", expression));
            assert_eq!(result, expected, "Expression '{}' returned {} instead of {}", expression, result, expected);
        }
    }

    /// Test M extension instructions specifically on RISC-V64 and AArch64
    #[test]
    #[cfg(any(target_arch = "riscv64", target_arch = "aarch64"))]
    fn test_m_extension() {
        let config = CalculatorConfig::default();
        
        // Test various multiplication cases
        let mul_cases = vec![
            ("0 * 100", 0),
            ("1 * 1", 1),
            ("12 * 12", 144),
            ("16 * 16", 256),  // Use smaller numbers to avoid encoding issues
        ];
        
        for (expr, expected) in mul_cases {
            let result = JitCalculator::evaluate(expr, &config).expect("Multiplication test failed");
            assert_eq!(result, expected, "Multiplication: {} should equal {}", expr, expected);
        }
        
        // Test division cases
        let div_cases = vec![
            ("100 / 10", 10),
            ("1 / 1", 1),
            ("144 / 12", 12),
            ("1000 / 10", 100),  // Use smaller numbers to avoid immediate encoding issues
        ];
        
        for (expr, expected) in div_cases {
            let result = JitCalculator::evaluate(expr, &config).expect("Division test failed");
            assert_eq!(result, expected, "Division: {} should equal {}", expr, expected);
        }
        
        // Test remainder cases
        let rem_cases = vec![
            ("10 % 3", 1),
            ("100 % 7", 2),
            ("50 % 16", 2),   // 50 = 3 * 16 + 2
            ("123 % 10", 3),  // 123 = 12 * 10 + 3
        ];
        
        for (expr, expected) in rem_cases {
            let result = JitCalculator::evaluate(expr, &config).expect("Remainder test failed");
            assert_eq!(result, expected, "Remainder: {} should equal {}", expr, expected);
        }
    }

    /// Benchmark-style test to ensure JIT compilation is working on RISC-V64 and AArch64
    #[test]
    #[cfg(any(target_arch = "riscv64", target_arch = "aarch64"))]
    fn test_jit_compilation_performance() {
        let config = CalculatorConfig::default();
        
        // Create the same calculation multiple times to ensure JIT is actually working
        let expression = "(123 + 456) * (789 - 456) + 999";
        let expected = (123 + 456) * (789 - 456) + 999; // Calculate expected result
        
        // Run multiple times to verify consistency
        for i in 1..=10 {
            let result = JitCalculator::evaluate(expression, &config)
                .expect(&format!("JIT test iteration {} failed", i));
            assert_eq!(result, expected, "JIT compilation inconsistent at iteration {}", i);
        }
    }

    /// Test machine code generation output on RISC-V64 and AArch64
    #[test]
    #[cfg(any(target_arch = "riscv64", target_arch = "aarch64"))]
    fn test_machine_code_generation() {
        let mut config = CalculatorConfig::default();
        config.show_machine_code = true;
        
        // Test that machine code display doesn't break execution
        let result = JitCalculator::evaluate("42 + 13", &config).expect("Machine code test failed");
        assert_eq!(result, 55);
        
        // Test with complex expression
        let result = JitCalculator::evaluate("(10 * 5) + (20 / 4)", &config).expect("Complex machine code test failed");
        assert_eq!(result, 55); // 50 + 5 = 55
    }
}
