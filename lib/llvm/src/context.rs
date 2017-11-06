//! Translation state.

use cretonne;
use cretonne::ir;
use std::collections::HashMap;
use cton_frontend;
use llvm_sys::prelude::*;
use llvm_sys::target::*;
use std::u32;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Variable(pub u32);
impl cretonne::entity::EntityRef for Variable {
    fn new(index: usize) -> Self {
        debug_assert!(index < (u32::MAX as usize));
        Variable(index as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

impl Variable {
    pub fn new(i: usize) -> Self {
        debug_assert_eq!(i as u32 as usize, i);
        Variable(i as u32)
    }
}

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
    pub builder: cton_frontend::FunctionBuilder<'a, Variable>,

    /// Map from LLVM Values to `Variable`s.
    pub value_map: HashMap<LLVMValueRef, Variable>,

    /// Map from LLVM BasicBlocks to bookkeeping data.
    pub ebb_info: HashMap<LLVMBasicBlockRef, EbbInfo>,

    /// Map from LLVM BasicBlocks to Cretonne `Ebb`s. Only contains entries for
    /// blocks that correspond to `Ebb` headers.
    pub ebb_map: HashMap<LLVMBasicBlockRef, ir::Ebb>,

    /// The LLVM DataLayout (formerly known as TargetData, the name still used
    /// in the C API).
    pub dl: LLVMTargetDataRef,
}

impl<'a> Context<'a> {
    pub fn new(
        func: &'a mut ir::Function,
        il_builder: &'a mut cton_frontend::ILBuilder<Variable>,
        dl: LLVMTargetDataRef,
    ) -> Self {
        Self {
            builder: cton_frontend::FunctionBuilder::<Variable>::new(func, il_builder),
            value_map: HashMap::new(),
            ebb_info: HashMap::new(),
            ebb_map: HashMap::new(),
            dl,
        }
    }
}
