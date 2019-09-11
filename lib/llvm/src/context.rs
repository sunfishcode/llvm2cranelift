//! Translation state.

use cranelift_codegen;
use cranelift_codegen::ir;
use cranelift_frontend;
use cranelift_frontend::Variable;
use llvm_sys::prelude::*;
use llvm_sys::target::*;
use std::collections::HashMap;
use std::u32;

/// Information about Ebbs that we'll create.
pub struct EbbInfo {
    pub num_preds_left: usize,
}

impl Default for EbbInfo {
    fn default() -> Self {
        Self { num_preds_left: 0 }
    }
}

/// Data structures used throughout translation.
pub struct Context<'a> {
    /// Builder for emitting instructions.
    pub builder: cranelift_frontend::FunctionBuilder<'a>,

    /// Map from LLVM Values to `Variable`s.
    pub value_map: HashMap<LLVMValueRef, Variable>,

    /// Map from LLVM BasicBlocks to bookkeeping data.
    pub ebb_info: HashMap<LLVMBasicBlockRef, EbbInfo>,

    /// Map from LLVM BasicBlocks to Cranelift `Ebb`s. Only contains entries for
    /// blocks that correspond to `Ebb` headers.
    pub ebb_map: HashMap<LLVMBasicBlockRef, ir::Ebb>,

    /// The LLVM DataLayout (formerly known as TargetData, the name still used
    /// in the C API).
    pub dl: LLVMTargetDataRef,
}

impl<'a> Context<'a> {
    pub fn new(
        func: &'a mut ir::Function,
        il_builder: &'a mut cranelift_frontend::FunctionBuilderContext,
        dl: LLVMTargetDataRef,
    ) -> Self {
        Self {
            builder: cranelift_frontend::FunctionBuilder::new(func, il_builder),
            value_map: HashMap::new(),
            ebb_info: HashMap::new(),
            ebb_map: HashMap::new(),
            dl,
        }
    }
}
