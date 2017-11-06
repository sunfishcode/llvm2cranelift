//! Translator from LLVM IR to Cretonne IL.

#![deny(missing_docs)]

extern crate cretonne;
extern crate cton_frontend;
extern crate llvm_sys;
extern crate libc;

mod translate;
mod operations;
mod context;
mod module;
mod reloc_sink;
mod types;

pub use translate::{create_llvm_context, read_llvm, translate_module};
