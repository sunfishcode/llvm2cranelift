//! Translation from LLVM IR to Cretonne IL.
//!
//! It is very incomplete, currently just enough to exercise some interesting
//! APIs.
//!
//! It doesn't translate LLVM's PHIs, SSA uses, and SSA defs directly into
//! Cretonne; it instead just relies on Cretonne's SSA builder to reconstruct
//! what it needs. That simplifies handling of basic blocks that aren't in
//! RPO -- walking the blocks in layout order, we may see uses before we see
//! their defs.

use cretonne;
use cretonne::ir;
use cretonne::settings;
use cton_frontend;
use std::collections::{HashMap, hash_map};
use std::error::Error;
use std::str;
use std::u32;
use std::ptr;
use std::ffi;
use llvm_sys::prelude::*;
use llvm_sys::core::*;
use llvm_sys::ir_reader::*;
use llvm_sys::target::*;
use llvm_sys::LLVMTypeKind::*;
use libc;

use operations::{translate_function_params, translate_inst};

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

impl Default for Variable {
    fn default() -> Self {
        Variable(u32::MAX)
    }
}

/// Information about Ebbs that we'll create.
struct EbbInfo {
    pub num_preds_left: usize,
}

impl Default for EbbInfo {
    fn default() -> Self {
        Self { num_preds_left: 0 }
    }
}

/// Translate from an llvm-sys C-style string to a Rust String.
fn translate_string(charstar: *const libc::c_char) -> Result<String, String> {
    Ok(
        unsafe { ffi::CStr::from_ptr(charstar) }
            .to_str()
            .map_err(|err| err.description().to_string())?
            .to_string(),
    )
}

/// Translate from an llvm-sys C-style string to an `ir::FunctionName`.
pub fn translate_function_name(charstar: *const libc::c_char) -> Result<ir::FunctionName, String> {
    Ok(ir::FunctionName::new(
        translate_string(charstar)?.as_bytes(),
    ))
}

/// Create an LLVM Context.
pub fn create_llvm_context() -> LLVMContextRef {
    unsafe { LLVMContextCreate() }
}

/// Read LLVM IR (bitcode or text) from a file and return the resulting module.
pub fn read_llvm(ctx: LLVMContextRef, path: &str) -> Result<LLVMModuleRef, String> {
    let mut msg = ptr::null_mut();
    let mut buf = ptr::null_mut();
    let c_str = ffi::CString::new(path).map_err(
        |err| err.description().to_string(),
    )?;
    let llvm_name = c_str.as_ptr();
    if unsafe { LLVMCreateMemoryBufferWithContentsOfFile(llvm_name, &mut buf, &mut msg) } != 0 {
        return Err(format!(
            "error creating LLVM memory buffer for {}: {}",
            path,
            translate_string(msg)?
        ));
    }
    let mut module = ptr::null_mut();
    if unsafe { LLVMParseIRInContext(ctx, buf, &mut module, &mut msg) } != 0 {
        Err(format!(
            "error parsing LLVM IR in {}: {}",
            path,
            translate_string(msg)?
        ))
    } else {
        Ok(module)
    }
}

/// Translate an LLVM module into Cretonne IL.
pub fn translate_module(llvm_mod: LLVMModuleRef) -> Result<(), String> {
    let mut llvm_func = unsafe { LLVMGetFirstFunction(llvm_mod) };
    let data_layout = unsafe { LLVMGetModuleDataLayout(llvm_mod) };
    while llvm_func != ptr::null_mut() {
        if unsafe { LLVMIsDeclaration(llvm_func) } == 0 {
            translate_function(llvm_func, data_layout)?;
        }
        llvm_func = unsafe { LLVMGetNextFunction(llvm_func) };
    }
    Ok(())
}

/// Translate the contents of `llvm_func` to Cretonne IL.
pub fn translate_function(
    llvm_func: LLVMValueRef,
    data_layout: LLVMTargetDataRef,
) -> Result<(), String> {
    let mut ctx = cretonne::Context::new();
    let llvm_name = unsafe { LLVMGetValueName(llvm_func) };
    ctx.func.name = translate_function_name(llvm_name)?;
    ctx.func.signature = translate_sig(
        unsafe { LLVMGetElementType(LLVMTypeOf(llvm_func)) },
        data_layout,
    );

    {
        let mut il_builder = cton_frontend::ILBuilder::<Variable>::new();
        let mut builder =
            cton_frontend::FunctionBuilder::<Variable>::new(&mut ctx.func, &mut il_builder);
        let mut value_map: HashMap<LLVMValueRef, Variable> = HashMap::new();
        let mut ebb_info: HashMap<LLVMBasicBlockRef, EbbInfo> = HashMap::new();
        let mut ebb_map: HashMap<LLVMBasicBlockRef, ir::Ebb> = HashMap::new();

        // Make a pre-pass through the basic blocks to collect predecessor
        // information, which LLVM's C API doesn't expose directly.
        let mut llvm_bb = unsafe { LLVMGetFirstBasicBlock(llvm_func) };
        while llvm_bb != ptr::null_mut() {
            prepare_for_bb(llvm_bb, &mut builder, &mut ebb_info, &mut ebb_map);
            llvm_bb = unsafe { LLVMGetNextBasicBlock(llvm_bb) };
        }

        // Translate the contents of each basic block.
        let mut entry_block = true;
        llvm_bb = unsafe { LLVMGetFirstBasicBlock(llvm_func) };
        while llvm_bb != ptr::null_mut() {
            translate_bb(
                llvm_func,
                llvm_bb,
                &mut builder,
                &mut value_map,
                &mut ebb_info,
                &mut ebb_map,
                data_layout,
                entry_block,
            );
            llvm_bb = unsafe { LLVMGetNextBasicBlock(llvm_bb) };
            entry_block = false;
        }
    }

    // For now, just print and verify the result.
    println!("{}", ctx.func.display(None));
    let flags = settings::Flags::new(&settings::builder());
    ctx.verify_if(&flags).map_err(
        |err| err.description().to_string(),
    )?;
    Ok(())
}

/// Since LLVM's C API doesn't expose predecessor accessors, we make a prepass
/// and collect the information we need from the successor accessors.
fn prepare_for_bb(
    llvm_bb: LLVMBasicBlockRef,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    ebb_info: &mut HashMap<LLVMBasicBlockRef, EbbInfo>,
    ebb_map: &mut HashMap<LLVMBasicBlockRef, ir::Ebb>,
) {
    let term = unsafe { LLVMGetBasicBlockTerminator(llvm_bb) };
    let is_switch = unsafe { LLVMIsASwitchInst(term) } != ptr::null_mut();
    let num_succs = unsafe { LLVMGetNumSuccessors(term) };
    for i in 0..num_succs {
        let llvm_succ = unsafe { LLVMGetSuccessor(term, i) };
        {
            let info = ebb_info.entry(llvm_succ).or_insert(EbbInfo::default());
            info.num_preds_left += 1;
        }
        // If the block is reachable by branch (and not fallthrough), or by
        // a switch non-default edge (which can't use fallthrough), we need
        // an Ebb entry for it.
        if (is_switch && i != 0) || llvm_succ != unsafe { LLVMGetNextBasicBlock(llvm_bb) } {
            ebb_map.insert(llvm_succ, builder.create_ebb());
        }
    }
}

/// Translate the contents of `llvm_bb` to Cretonne IL instructions.
fn translate_bb(
    llvm_func: LLVMValueRef,
    llvm_bb: LLVMBasicBlockRef,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    value_map: &mut HashMap<LLVMValueRef, Variable>,
    ebb_info: &mut HashMap<LLVMBasicBlockRef, EbbInfo>,
    ebb_map: &mut HashMap<LLVMBasicBlockRef, ir::Ebb>,
    data_layout: LLVMTargetDataRef,
    entry_block: bool,
) {
    // Set up the Ebb as needed.
    if ebb_info.get(&llvm_bb).is_none() {
        // Block has no predecessors.
        let ebb = builder.create_ebb();
        builder.seal_block(ebb);
        builder.switch_to_block(ebb, &[]);
        if entry_block {
            // It's the entry block. Add the parameters.
            translate_function_params(llvm_func, ebb, builder, value_map, data_layout);
        }
    } else if let hash_map::Entry::Occupied(entry) = ebb_map.entry(llvm_bb) {
        // Block has predecessors and is branched to, so it starts a new Ebb.
        let ebb = *entry.get();
        builder.switch_to_block(ebb, &[]);
    }

    // Translate each regular instruction.
    let mut llvm_inst = unsafe { LLVMGetFirstInstruction(llvm_bb) };
    while llvm_inst != ptr::null_mut() {
        translate_inst(llvm_bb, llvm_inst, builder, value_map, ebb_map, data_layout);
        llvm_inst = unsafe { LLVMGetNextInstruction(llvm_inst) };
    }

    // Visit each CFG successor and seal blocks that have had all their
    // predecessors visited.
    let term = unsafe { LLVMGetBasicBlockTerminator(llvm_bb) };
    let num_succs = unsafe { LLVMGetNumSuccessors(term) };
    for i in 0..num_succs {
        let llvm_succ = unsafe { LLVMGetSuccessor(term, i) };
        let info = ebb_info.get_mut(&llvm_succ).unwrap();
        debug_assert!(info.num_preds_left > 0);
        info.num_preds_left -= 1;
        if info.num_preds_left == 0 {
            if let Some(ebb) = ebb_map.get(&llvm_succ) {
                builder.seal_block(*ebb);
            }
        }
    }
}

/// Return a Cretonne integer type with the given bit width.
pub fn translate_integer_type(bitwidth: usize) -> ir::Type {
    match bitwidth {
        1 => ir::types::B1,
        8 => ir::types::I8,
        16 => ir::types::I16,
        32 => ir::types::I32,
        64 => ir::types::I64,
        width @ _ => panic!("unimplemented integer bit width {}", width),
    }
}

/// Return the Cretonne integer type for a pointer.
pub fn translate_pointer_type(data_layout: LLVMTargetDataRef) -> ir::Type {
    translate_integer_type(unsafe { LLVMPointerSize(data_layout) * 8 } as usize)
}

/// Translate an LLVM first-class type into a Cretonne type.
pub fn translate_type(llvm_ty: LLVMTypeRef, data_layout: LLVMTargetDataRef) -> ir::Type {
    match unsafe { LLVMGetTypeKind(llvm_ty) } {
        LLVMVoidTypeKind => ir::types::VOID,
        LLVMHalfTypeKind => panic!("unimplemented: f16 type"),
        LLVMFloatTypeKind => ir::types::F32,
        LLVMDoubleTypeKind => ir::types::F64,
        LLVMX86_FP80TypeKind => panic!("unimplemented: x86_fp80 type"),
        LLVMFP128TypeKind => panic!("unimplemented: f128 type"),
        LLVMPPC_FP128TypeKind => panic!("unimplemented: double double type"),
        LLVMLabelTypeKind => panic!("attempted to translate label type"),
        LLVMIntegerTypeKind => translate_integer_type(
            unsafe { LLVMGetIntTypeWidth(llvm_ty) } as usize,
        ),
        LLVMFunctionTypeKind => panic!("use translate_sig to translate function types"),
        LLVMStructTypeKind => panic!("unimplemented: first-class struct types"),
        LLVMArrayTypeKind => panic!("unimplemented: first-class array types"),
        LLVMPointerTypeKind => {
            if unsafe { LLVMGetPointerAddressSpace(llvm_ty) } != 0 {
                panic!("unimplemented: non-default address spaces");
            }
            translate_pointer_type(data_layout)
        }
        LLVMVectorTypeKind => panic!("unimplemented: vector types"),
        LLVMMetadataTypeKind => panic!("attempted to translate a metadata type"),
        LLVMX86_MMXTypeKind => panic!("unimplemented: MMX type"),
        LLVMTokenTypeKind => panic!("attemped to translate a token type"),
    }
}

/// Translate an LLVM function type into a Cretonne signature.
pub fn translate_sig(llvm_ty: LLVMTypeRef, data_layout: LLVMTargetDataRef) -> ir::Signature {
    debug_assert_eq!(unsafe { LLVMGetTypeKind(llvm_ty) }, LLVMFunctionTypeKind);

    let mut sig = ir::Signature::new(ir::CallConv::Native);

    let num_llvm_params = unsafe { LLVMCountParamTypes(llvm_ty) } as usize;
    let mut llvm_params: Vec<LLVMTypeRef> = Vec::with_capacity(num_llvm_params);
    llvm_params.resize(num_llvm_params, ptr::null_mut());

    // TODO: First-class aggregate params and return values.

    unsafe { LLVMGetParamTypes(llvm_ty, llvm_params.as_mut_ptr()) };
    let mut params: Vec<ir::AbiParam> = Vec::with_capacity(num_llvm_params);
    for llvm_param in &llvm_params {
        params.push(ir::AbiParam::new(translate_type(*llvm_param, data_layout)));
    }
    sig.params = params;

    let mut returns: Vec<ir::AbiParam> = Vec::with_capacity(1);
    match translate_type(unsafe { LLVMGetReturnType(llvm_ty) }, data_layout) {
        ir::types::VOID => {}
        ty @ _ => returns.push(ir::AbiParam::new(ty)),
    }
    sig.returns = returns;

    sig
}
