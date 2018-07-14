//! Translator from LLVM IR to Cranelift IL.

#![deny(missing_docs)]

extern crate cranelift_codegen;
extern crate cranelift_frontend;
extern crate fnv;
extern crate indexmap;
extern crate libc;
extern crate llvm_sys;

mod context;
mod module;
mod operations;
mod reloc_sink;
mod string_table;
mod translate;
mod types;

pub use module::SymbolKind;
pub use translate::{create_llvm_context, read_llvm, translate_module};
