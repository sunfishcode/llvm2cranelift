//! Translator from LLVM IR to Cretonne IL.

#![deny(missing_docs)]

extern crate cretonne;
extern crate cton_frontend;
extern crate fnv;
extern crate llvm_sys;
extern crate libc;
extern crate ordermap;

mod translate;
mod operations;
mod context;
mod module;
mod reloc_sink;
mod types;
mod string_table;

pub use translate::{create_llvm_context, read_llvm, translate_module};
pub use module::SymbolKind;
