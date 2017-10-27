//! Translation from LLVM IR to Cretonne IL.

use cretonne;
use cretonne::ir;
use cretonne::settings;
use cton_frontend;
use std::collections::hash_map;
use std::error::Error;
use std::str;
use std::ptr;
use std::ffi;
use llvm_sys::prelude::*;
use llvm_sys::core::*;
use llvm_sys::ir_reader::*;
use llvm_sys::target::*;
use llvm_sys::LLVMTypeKind::*;
use libc;

use operations::{translate_function_params, translate_inst};
use context::{Context, EbbInfo, Variable};

/// Translate from an llvm-sys C-style string to a Rust String.
pub fn translate_string(charstar: *const libc::c_char) -> Result<String, String> {
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
pub fn read_llvm(llvm_ctx: LLVMContextRef, path: &str) -> Result<LLVMModuleRef, String> {
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
    if unsafe { LLVMParseIRInContext(llvm_ctx, buf, &mut module, &mut msg) } != 0 {
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
pub fn translate_module(llvm_mod: LLVMModuleRef) -> Result<Vec<ir::Function>, String> {
    // TODO: Use a more sophisticated API rather than just stuffing all
    // the functions into a Vec.
    let mut results = Vec::new();
    let mut llvm_func = unsafe { LLVMGetFirstFunction(llvm_mod) };
    let dl = unsafe { LLVMGetModuleDataLayout(llvm_mod) };
    while !llvm_func.is_null() {
        if unsafe { LLVMIsDeclaration(llvm_func) } == 0 {
            results.push(translate_function(llvm_func, dl)?);
        }
        llvm_func = unsafe { LLVMGetNextFunction(llvm_func) };
    }
    Ok(results)
}

/// Translate the contents of `llvm_func` to Cretonne IL.
pub fn translate_function(
    llvm_func: LLVMValueRef,
    dl: LLVMTargetDataRef,
) -> Result<ir::Function, String> {
    // TODO: Reuse the context between separate invocations.
    let mut cton_ctx = cretonne::Context::new();
    let llvm_name = unsafe { LLVMGetValueName(llvm_func) };
    cton_ctx.func.name = translate_function_name(llvm_name)?;
    cton_ctx.func.signature =
        translate_sig(unsafe { LLVMGetElementType(LLVMTypeOf(llvm_func)) }, dl);

    {
        let mut il_builder = cton_frontend::ILBuilder::<Variable>::new();
        let mut ctx = Context::new(&mut cton_ctx.func, &mut il_builder, dl);

        // Make a pre-pass through the basic blocks to collect predecessor
        // information, which LLVM's C API doesn't expose directly.
        let mut llvm_bb = unsafe { LLVMGetFirstBasicBlock(llvm_func) };
        while !llvm_bb.is_null() {
            prepare_for_bb(llvm_bb, &mut ctx);
            llvm_bb = unsafe { LLVMGetNextBasicBlock(llvm_bb) };
        }

        // Translate the contents of each basic block.
        llvm_bb = unsafe { LLVMGetFirstBasicBlock(llvm_func) };
        while !llvm_bb.is_null() {
            translate_bb(llvm_func, llvm_bb, &mut ctx);
            llvm_bb = unsafe { LLVMGetNextBasicBlock(llvm_bb) };
        }
    }

    // TODO: Make the flags configurable.
    // TODO: This verification pass may be redundant in some settings.
    let flags = settings::Flags::new(&settings::builder());
    cton_ctx.verify_if(&flags).map_err(|err| {
        err.description().to_string()
    })?;

    Ok((cton_ctx.func))
}

/// Since LLVM's C API doesn't expose predecessor accessors, we make a prepass
/// and collect the information we need from the successor accessors.
fn prepare_for_bb(llvm_bb: LLVMBasicBlockRef, ctx: &mut Context) {
    let term = unsafe { LLVMGetBasicBlockTerminator(llvm_bb) };
    let is_switch = !unsafe { LLVMIsASwitchInst(term) }.is_null();
    let num_succs = unsafe { LLVMGetNumSuccessors(term) };
    for i in 0..num_succs {
        let llvm_succ = unsafe { LLVMGetSuccessor(term, i) };
        {
            let info = ctx.ebb_info.entry(llvm_succ).or_insert_with(
                EbbInfo::default,
            );
            info.num_preds_left += 1;
        }
        // If the block is reachable by branch (and not fallthrough), or by
        // a switch non-default edge (which can't use fallthrough), we need
        // an Ebb entry for it.
        if (is_switch && i != 0) || llvm_succ != unsafe { LLVMGetNextBasicBlock(llvm_bb) } {
            ctx.ebb_map.insert(llvm_succ, ctx.builder.create_ebb());
        }
    }
}

/// Translate the contents of `llvm_bb` to Cretonne IL instructions.
fn translate_bb(llvm_func: LLVMValueRef, llvm_bb: LLVMBasicBlockRef, ctx: &mut Context) {
    // Set up the Ebb as needed.
    if ctx.ebb_info.get(&llvm_bb).is_none() {
        // Block has no predecessors.
        let entry_block = !ctx.builder.entry_block_started();
        let ebb = ctx.builder.create_ebb();
        ctx.builder.seal_block(ebb);
        ctx.builder.switch_to_block(ebb, &[]);
        if entry_block {
            // It's the entry block. Add the parameters.
            translate_function_params(llvm_func, ctx);
        }
    } else if let hash_map::Entry::Occupied(entry) = ctx.ebb_map.entry(llvm_bb) {
        // Block has predecessors and is branched to, so it starts a new Ebb.
        let ebb = *entry.get();
        ctx.builder.switch_to_block(ebb, &[]);
    }

    // Translate each regular instruction.
    let mut llvm_inst = unsafe { LLVMGetFirstInstruction(llvm_bb) };
    while !llvm_inst.is_null() {
        translate_inst(llvm_bb, llvm_inst, ctx);
        llvm_inst = unsafe { LLVMGetNextInstruction(llvm_inst) };
    }

    // Visit each CFG successor and seal blocks that have had all their
    // predecessors visited.
    let term = unsafe { LLVMGetBasicBlockTerminator(llvm_bb) };
    let num_succs = unsafe { LLVMGetNumSuccessors(term) };
    for i in 0..num_succs {
        let llvm_succ = unsafe { LLVMGetSuccessor(term, i) };
        let info = ctx.ebb_info.get_mut(&llvm_succ).unwrap();
        debug_assert!(info.num_preds_left > 0);
        info.num_preds_left -= 1;
        if info.num_preds_left == 0 {
            if let Some(ebb) = ctx.ebb_map.get(&llvm_succ) {
                ctx.builder.seal_block(*ebb);
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
        width => panic!("unimplemented integer bit width {}", width),
    }
}

/// Return the Cretonne integer type for a pointer.
pub fn translate_pointer_type(dl: LLVMTargetDataRef) -> ir::Type {
    translate_integer_type(unsafe { LLVMPointerSize(dl) * 8 } as usize)
}

/// Translate an LLVM first-class type into a Cretonne type.
pub fn translate_type(llvm_ty: LLVMTypeRef, dl: LLVMTargetDataRef) -> ir::Type {
    match unsafe { LLVMGetTypeKind(llvm_ty) } {
        LLVMVoidTypeKind => ir::types::VOID,
        LLVMHalfTypeKind => panic!("unimplemented: f16 type"),
        LLVMFloatTypeKind => ir::types::F32,
        LLVMDoubleTypeKind => ir::types::F64,
        LLVMX86_FP80TypeKind => panic!("unimplemented: x86_fp80 type"),
        LLVMFP128TypeKind => panic!("unimplemented: f128 type"),
        LLVMPPC_FP128TypeKind => panic!("unimplemented: double double type"),
        LLVMLabelTypeKind => panic!("unimplemented: label types"),
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
            translate_pointer_type(dl)
        }
        LLVMVectorTypeKind => panic!("unimplemented: vector types"),
        LLVMMetadataTypeKind => panic!("attempted to translate a metadata type"),
        LLVMX86_MMXTypeKind => panic!("unimplemented: MMX type"),
        LLVMTokenTypeKind => panic!("unimplemented: token types"),
    }
}

/// Translate an LLVM function type into a Cretonne signature.
pub fn translate_sig(llvm_ty: LLVMTypeRef, dl: LLVMTargetDataRef) -> ir::Signature {
    debug_assert_eq!(unsafe { LLVMGetTypeKind(llvm_ty) }, LLVMFunctionTypeKind);

    let mut sig = ir::Signature::new(ir::CallConv::Native);

    let num_llvm_params = unsafe { LLVMCountParamTypes(llvm_ty) } as usize;
    let mut llvm_params: Vec<LLVMTypeRef> = Vec::with_capacity(num_llvm_params);
    llvm_params.resize(num_llvm_params, ptr::null_mut());

    // TODO: First-class aggregate params and return values.

    unsafe { LLVMGetParamTypes(llvm_ty, llvm_params.as_mut_ptr()) };
    let mut params: Vec<ir::AbiParam> = Vec::with_capacity(num_llvm_params);
    for llvm_param in &llvm_params {
        params.push(ir::AbiParam::new(translate_type(*llvm_param, dl)));
    }
    sig.params = params;

    let mut returns: Vec<ir::AbiParam> = Vec::with_capacity(1);
    match translate_type(unsafe { LLVMGetReturnType(llvm_ty) }, dl) {
        ir::types::VOID => {}
        ty => returns.push(ir::AbiParam::new(ty)),
    }
    sig.returns = returns;

    sig
}
