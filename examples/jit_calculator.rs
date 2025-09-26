//! JIT Calculator with AST Example
//! 
//! This example demonstrates a sophisticated JIT-compiled calculator that uses
//! an Abstract Syntax Tree (AST) to represent mathematical expressions.
//! The calculator compiles expressions to native RISC-V machine code using
//! the M extension for arithmetic operations.
//! 
//! Features:
//! - AST-based expression parsing and evaluation
//! - Support for parentheses and operator precedence
//! - JIT compilation to RISC-V machine code
//! - Optimized code generation for complex expressions
//! 
//! Supported operations: +, -, *, /, % (remainder), and parentheses
//! 
//! Note: This example works on RISC-V hosts or in emulation.
//! On other architectures, the functions will be created successfully
//! but calling them will likely crash.

use jit_assembler::riscv::{reg, Riscv64InstructionBuilder};
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

/// JIT compiler that converts AST to RISC-V machine code
pub struct JitCompiler {
    builder: Riscv64InstructionBuilder,
    register_stack: Vec<jit_assembler::riscv::Register>,
    next_temp_reg: usize,
}

impl JitCompiler {
    /// Available temporary registers for computation
    /// Using only T0-T6 (caller-saved temporaries) - safe to use without preservation
    /// Avoiding:
    /// - A0: function return value
    /// - RA: return address  
    /// - SP: stack pointer
    /// - S0-S11: callee-saved (would need to preserve)
    const TEMP_REGISTERS: &'static [jit_assembler::riscv::Register] = &[
        reg::T0, reg::T1, reg::T2, reg::T3, reg::T4, reg::T5, reg::T6,
    ];

    /// Allocate a temporary register
    fn alloc_register(&mut self) -> Result<jit_assembler::riscv::Register, String> {
        if self.next_temp_reg >= Self::TEMP_REGISTERS.len() {
            return Err("Out of temporary registers".to_string());
        }
        let reg = Self::TEMP_REGISTERS[self.next_temp_reg];
        self.next_temp_reg += 1;
        self.register_stack.push(reg);
        Ok(reg)
    }

    /// Free the last allocated register
    fn free_register(&mut self) -> Result<(), String> {
        if self.register_stack.is_empty() {
            return Err("No registers to free".to_string());
        }
        self.register_stack.pop();
        self.next_temp_reg = self.next_temp_reg.saturating_sub(1);
        Ok(())
    }
}

impl JitCompiler {
    pub fn new() -> Self {
        Self {
            builder: Riscv64InstructionBuilder::new(),
            register_stack: Vec::new(),
            next_temp_reg: 0,
        }
    }

    /// Compile an AST to a JIT function
    /// The result is stored in register A0
    pub fn compile_expression(&mut self, ast: &AstNode, config: &CalculatorConfig) -> Result<Box<dyn Fn() -> u64>, Box<dyn std::error::Error>> {
        // Generate code that computes the expression result in A0
        self.compile_node(ast, reg::A0)?;
        self.builder.ret();

        // Show machine code if requested
        if config.show_machine_code {
            self.show_generated_code();
        }

        let jit_func = unsafe {
            self.builder.function::<fn() -> u64>()?
        };

        Ok(Box::new(move || jit_func.call()))
    }

    /// Display the generated machine code
    pub fn show_generated_code(&self) {
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

    /// Compile an AST node, storing the result in the specified register
    fn compile_node(&mut self, node: &AstNode, result_reg: jit_assembler::riscv::Register) -> Result<(), String> {
        match node {
            AstNode::Number(value) => {
                // Load immediate value into result register
                if *value <= 2047 {
                    // Small immediate: can use ADDI with zero register
                    self.builder.addi(result_reg, reg::ZERO, *value as i16);
                } else {
                    // Large immediate: use LUI + ADDI
                    let upper = (*value >> 12) as u32;
                    let lower = (*value & 0xFFF) as i16;
                    self.builder.lui(result_reg, upper);
                    if lower != 0 {
                        self.builder.addi(result_reg, result_reg, lower);
                    }
                }
                Ok(())
            }
            AstNode::BinaryOp { left, op, right } => {
                // Allocate temporary registers for operands
                let left_reg = self.alloc_register()?;
                let right_reg = self.alloc_register()?;

                // Compile left operand into left_reg
                self.compile_node(left, left_reg)?;
                
                // Compile right operand into right_reg  
                self.compile_node(right, right_reg)?;

                // Perform operation, result in result_reg
                match op {
                    BinaryOperator::Add => {
                        self.builder.add(result_reg, left_reg, right_reg);
                    }
                    BinaryOperator::Subtract => {
                        self.builder.sub(result_reg, left_reg, right_reg);
                    }
                    BinaryOperator::Multiply => {
                        self.builder.mul(result_reg, left_reg, right_reg);
                    }
                    BinaryOperator::Divide => {
                        // Note: This is unsigned division. For signed, use div instead
                        self.builder.divu(result_reg, left_reg, right_reg);
                    }
                    BinaryOperator::Remainder => {
                        // Note: This is unsigned remainder. For signed, use rem instead
                        self.builder.remu(result_reg, left_reg, right_reg);
                    }
                }

                // Free the temporary registers in reverse order
                self.free_register()?; // right_reg
                self.free_register()?; // left_reg
                
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
        
        // Compile AST to JIT function
        println!("🔧 Compiling to RISC-V machine code...");
        let mut compiler = JitCompiler::new();
        let jit_function = compiler.compile_expression(&ast, config)?;
        
        // Execute JIT function
        if cfg!(target_arch = "riscv64") {
            let result = jit_function();
            println!("✅ JIT execution result: {}", result);
            Ok(result)
        } else {
            // On non-RISC-V platforms, fall back to AST interpretation
            println!("⚠️  Not on RISC-V platform, using AST interpreter");
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
            if std::io::stdin().read_line(&mut input).is_err() {
                println!("❌ Error reading input");
                continue;
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
fn demonstrate_jit_compilation() {
    println!("Generating machine code for multiplication (7 * 6)...");
    
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