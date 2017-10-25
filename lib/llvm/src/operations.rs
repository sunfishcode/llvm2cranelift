//! Translation from LLVM IR operations to Cretonne IL instructions.

use cretonne::ir::{self, InstBuilder};
use cton_frontend;
use std::collections::{HashMap, hash_map};
use std::mem;
use std::ptr;
use llvm_sys::prelude::*;
use llvm_sys::core::*;
use llvm_sys::target::*;
use llvm_sys::LLVMValueKind::*;
use llvm_sys::LLVMOpcode::*;
use llvm_sys::LLVMTypeKind::*;
use llvm_sys::LLVMIntPredicate::*;
use llvm_sys::LLVMRealPredicate::*;
use llvm_sys::LLVMAtomicOrdering::*;
use llvm_sys::LLVMCallConv::*;
use llvm_sys::*;
use libc;

use translate::{Variable, translate_type, translate_pointer_type, translate_sig,
                translate_function_name, translate_string};

/// Translate the incoming parameters for `llvm_func` into Cretonne values
/// defined in the entry block.
pub fn translate_function_params(
    llvm_func: LLVMValueRef,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    value_map: &mut HashMap<LLVMValueRef, Variable>,
    data_layout: LLVMTargetDataRef,
) {
    for i in 0..unsafe { LLVMCountParams(llvm_func) } {
        let llvm_param = unsafe { LLVMGetParam(llvm_func, i) };
        let val = builder.param_value(i as usize);
        def_val(llvm_param, val, builder, value_map, data_layout);
    }
}

/// Translate `llvm_inst`, which is a normal instruction, to Cretonne IL
/// instructions.
pub fn translate_inst(
    llvm_bb: LLVMBasicBlockRef,
    llvm_inst: LLVMValueRef,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    value_map: &mut HashMap<LLVMValueRef, Variable>,
    ebb_map: &HashMap<LLVMBasicBlockRef, ir::Ebb>,
    data_layout: LLVMTargetDataRef,
) {
    let llvm_opcode = unsafe { LLVMGetInstructionOpcode(llvm_inst) };
    match llvm_opcode {
        LLVMPHI => {
            // Nothing to do. Phis are handled elsewhere.
        }
        LLVMAdd => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().iadd(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMSub => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().isub(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMMul => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().imul(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMSDiv => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().sdiv(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMUDiv => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().udiv(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMSRem => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().srem(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMURem => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().urem(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMAShr => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().sshr(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMLShr => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().ushr(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMShl => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().ishl(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMAnd => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().band(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMOr => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().bor(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMXor => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().bxor(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMFAdd => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().fadd(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMFSub => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().fsub(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMFMul => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().fmul(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMFDiv => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().fdiv(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMFRem => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let ty = translate_type_of(llvm_inst, data_layout);
            let mut sig = ir::Signature::new(ir::CallConv::Native);
            sig.params.resize(2, ir::AbiParam::new(ty));
            sig.returns.push(ir::AbiParam::new(ty));
            let data = ir::ExtFuncData {
                name: ir::FunctionName::new(match ty {
                    ir::types::F32 => "fmodf",
                    ir::types::F64 => "fmod",
                    _ => panic!("frem unimplemented for type {:?}", ty),
                }),
                signature: builder.import_signature(sig),
            };
            let callee = builder.import_function(data);
            let call = builder.ins().call(callee, &[lhs, rhs]);
            let result = builder.inst_results(call)[0];
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMICmp => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let condcode = unsafe { LLVMGetICmpPredicate(llvm_inst) };
            let result = builder.ins().icmp(translate_icmp_code(condcode), lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMFCmp => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let condcode = unsafe { LLVMGetFCmpPredicate(llvm_inst) };
            let result = match condcode {
                LLVMRealPredicateFalse => builder.ins().bconst(ir::types::B1, false),
                LLVMRealPredicateTrue => builder.ins().bconst(ir::types::B1, true),
                _ => builder.ins().fcmp(translate_fcmp_code(condcode), lhs, rhs),
            };
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMTrunc => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().ireduce(
                translate_type_of(llvm_inst, data_layout),
                op,
            );
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMZExt => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().uextend(
                translate_type_of(llvm_inst, data_layout),
                op,
            );
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMSExt => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().sextend(
                translate_type_of(llvm_inst, data_layout),
                op,
            );
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMFPToSI => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().fcvt_to_sint(
                translate_type_of(llvm_inst, data_layout),
                op,
            );
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMFPToUI => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().fcvt_to_uint(
                translate_type_of(llvm_inst, data_layout),
                op,
            );
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMSIToFP => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().fcvt_from_sint(
                translate_type_of(llvm_inst, data_layout),
                op,
            );
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMUIToFP => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().fcvt_from_uint(
                translate_type_of(llvm_inst, data_layout),
                op,
            );
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMFPTrunc => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().fdemote(
                translate_type_of(llvm_inst, data_layout),
                op,
            );
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMFPExt => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().fpromote(
                translate_type_of(llvm_inst, data_layout),
                op,
            );
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMBitCast => {
            let llvm_op = unsafe { LLVMGetOperand(llvm_inst, 0) };
            let op = use_val(llvm_op, builder, value_map, data_layout);
            let from = translate_type_of(llvm_op, data_layout);
            let to = translate_type_of(llvm_inst, data_layout);
            let result = if from == to {
                use_val(llvm_op, builder, value_map, data_layout)
            } else {
                builder.ins().bitcast(to, op)
            };
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMPtrToInt | LLVMIntToPtr => {
            let llvm_op = unsafe { LLVMGetOperand(llvm_inst, 0) };
            let op = use_val(llvm_op, builder, value_map, data_layout);
            let from = translate_type_of(llvm_op, data_layout);
            let to = translate_type_of(llvm_inst, data_layout);
            let result = if from == to {
                use_val(llvm_op, builder, value_map, data_layout)
            } else if from.bits() > to.bits() {
                builder.ins().ireduce(to, op)
            } else {
                builder.ins().uextend(to, op)
            };
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMSelect => {
            let (c, t, f) = ternary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().select(c, t, f);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMLoad => {
            let llvm_ty = unsafe { LLVMTypeOf(llvm_inst) };
            let flags = translate_memflags(llvm_inst, llvm_ty, data_layout);
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().load(
                translate_type(llvm_ty, data_layout),
                flags,
                op,
                0,
            );
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMStore => {
            let llvm_ty = unsafe { LLVMTypeOf(llvm_inst) };
            let flags = translate_memflags(llvm_inst, llvm_ty, data_layout);
            let (val, ptr) = binary_operands(llvm_inst, builder, value_map, data_layout);
            builder.ins().store(flags, val, ptr, 0);
        }
        LLVMBr => {
            if unsafe { LLVMIsConditional(llvm_inst) } != 0 {
                let cond = unary_operands(llvm_inst, builder, value_map, data_layout);
                let llvm_true_succ = successor(llvm_inst, 0);
                let llvm_false_succ = successor(llvm_inst, 1);
                let llvm_next_bb = unsafe { LLVMGetNextBasicBlock(llvm_bb) };
                if llvm_next_bb == llvm_true_succ {
                    handle_phi_operands(llvm_bb, llvm_false_succ, builder, value_map, data_layout);
                    builder.ins().brz(cond, ebb_map[&llvm_false_succ], &[]);
                    jump(
                        llvm_bb,
                        llvm_true_succ,
                        builder,
                        value_map,
                        ebb_map,
                        data_layout,
                    );
                } else {
                    handle_phi_operands(llvm_bb, llvm_true_succ, builder, value_map, data_layout);
                    builder.ins().brnz(cond, ebb_map[&llvm_true_succ], &[]);
                    jump(
                        llvm_bb,
                        llvm_false_succ,
                        builder,
                        value_map,
                        ebb_map,
                        data_layout,
                    );
                }
            } else {
                let llvm_succ = successor(llvm_inst, 0);
                jump(llvm_bb, llvm_succ, builder, value_map, ebb_map, data_layout);
            }
        }
        LLVMSwitch => {
            // TODO: We'd really want getNumCases or ConstCaseIt here, but
            // LLVM's C API doesn't expose those currently.
            let num_cases = unsafe { LLVMGetNumOperands(llvm_inst) / 2 - 1 };
            let mut data = ir::JumpTableData::with_capacity(num_cases as usize);
            for i in 0..num_cases {
                let llvm_val = unsafe { LLVMGetOperand(llvm_inst, ((i as libc::c_uint + 1) * 2)) };
                if unsafe { LLVMConstIntGetZExtValue(llvm_val) } != i as u64 {
                    panic!("unimplemented: switches with non-sequential cases");
                }
                let llvm_case = unsafe { LLVMGetSuccessor(llvm_inst, i as libc::c_uint) };
                data.push_entry(ebb_map[&llvm_case]);
            }
            let jt = builder.create_jump_table(data);
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            builder.ins().br_table(op, jt);
            let llvm_default = unsafe { LLVMGetSwitchDefaultDest(llvm_inst) };
            jump(
                llvm_bb,
                llvm_default,
                builder,
                value_map,
                ebb_map,
                data_layout,
            );
        }
        LLVMAlloca => {
            let llvm_op = unsafe { LLVMGetOperand(llvm_inst, 0) };
            let align = unsafe { LLVMGetAlignment(llvm_inst) };
            let llvm_allocty = unsafe { LLVMGetAllocatedType(llvm_inst) };
            // TODO: We'd really want isArrayAllocation and isStaticAlloca here, but
            // LLVM's C API doesn't expose those currently.
            if unsafe { LLVMIsConstant(llvm_op) } == 0 {
                panic!("unimplemented: dynamic alloca");
            }
            if llvm_bb != unsafe { LLVMGetEntryBasicBlock(LLVMGetBasicBlockParent(llvm_bb)) } {
                panic!("unimplemented: alloca outside of entry block");
            }
            if unsafe { LLVMGetValueKind(llvm_op) } != LLVMConstantIntValueKind {
                panic!("unimplemented: ConstantExpr alloca size");
            }
            if align > unsafe { LLVMABIAlignmentOfType(data_layout, llvm_allocty) } {
                panic!("unimplemented: supernaturally aligned alloca");
            }
            let size = unsafe { LLVMConstIntGetZExtValue(llvm_op) } *
                unsafe { LLVMABISizeOfType(data_layout, llvm_allocty) };
            if u64::from(size as u32) != size {
                panic!("unimplemented: alloca size computation doesn't fit in u32");
            }
            let stack_slot_data = ir::StackSlotData::new(ir::StackSlotKind::Local, size as u32);
            let stack_slot = builder.create_stack_slot(stack_slot_data);
            let pointer_type = translate_pointer_type(data_layout);
            let result = builder.ins().stack_addr(pointer_type, stack_slot, 0);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        LLVMGetElementPtr => {
            let llvm_ptr = unsafe { LLVMGetOperand(llvm_inst, 0) };
            let mut ptr = use_val(llvm_ptr, builder, value_map, data_layout);
            let pointer_type = translate_pointer_type(data_layout);
            let mut llvm_gepty = unsafe { LLVMTypeOf(llvm_ptr) };
            for i in 1..unsafe { LLVMGetNumOperands(llvm_inst) } as libc::c_uint {
                let index = unsafe { LLVMGetOperand(llvm_inst, i) };
                let (new_ptr, new_llvm_gepty) = translate_gep_index(
                    llvm_gepty,
                    ptr,
                    pointer_type,
                    index,
                    builder,
                    value_map,
                    data_layout,
                );
                ptr = new_ptr;
                llvm_gepty = new_llvm_gepty;
            }
            def_val(llvm_inst, ptr, builder, value_map, data_layout);
        }
        LLVMCall => {
            let callee_index = unsafe { LLVMGetNumOperands(llvm_inst) } - 1;
            let llvm_callee = unsafe { LLVMGetOperand(llvm_inst, callee_index as libc::c_uint) };

            if unsafe { LLVMGetIntrinsicID(llvm_callee) } != 0 {
                translate_intrinsic(llvm_inst, llvm_callee, builder, value_map, data_layout);
                return;
            }

            // Fast and cold are not ABI-exposed, so we can handle them however
            // we like. Just handle them the same as normal calls for now.
            let callconv = unsafe { LLVMGetInstructionCallConv(llvm_inst) };
            if callconv != LLVMCCallConv as libc::c_uint &&
                callconv != LLVMFastCallConv as libc::c_uint &&
                callconv != LLVMColdCallConv as libc::c_uint
            {
                panic!("unimplemented calling convention: {}", callconv);
            }
            // Collect the call arguments.
            // TODO: Implement ABI-exposed argument attributes such as byval,
            // signext, zeroext, inreg, inalloca, align.
            let num_args = unsafe { LLVMGetNumArgOperands(llvm_inst) } as usize;
            let mut args = Vec::with_capacity(num_args);
            for i in 0..num_args {
                debug_assert_eq!(i as libc::c_uint as usize, i);
                args.push(use_val(
                    unsafe { LLVMGetOperand(llvm_inst, i as libc::c_uint) },
                    builder,
                    value_map,
                    data_layout,
                ));
            }
            let signature = builder.import_signature(translate_sig(
                unsafe { LLVMGetElementType(LLVMTypeOf(llvm_callee)) },
                data_layout,
            ));
            let name = translate_function_name(unsafe { LLVMGetValueName(llvm_callee) })
                .expect("unimplemented: unusual function names");
            let call = if unsafe { LLVMGetValueKind(llvm_callee) } == LLVMFunctionValueKind {
                let data = ir::ExtFuncData { name, signature };
                let callee = builder.import_function(data);
                builder.ins().call(callee, &args)
            } else {
                let callee = use_val(llvm_callee, builder, value_map, data_layout);
                builder.ins().call_indirect(signature, callee, &args)
            };
            if translate_type_of(llvm_inst, data_layout) != ir::types::VOID {
                let result = {
                    let results = builder.inst_results(call);
                    debug_assert_eq!(results.len(), 1);
                    results[0]
                };
                def_val(llvm_inst, result, builder, value_map, data_layout);
            }
        }
        LLVMRet => {
            if unsafe { LLVMGetNumOperands(llvm_inst) } == 0 {
                builder.ins().return_(&[]);
            } else {
                // TODO: multiple return values
                let op = unary_operands(llvm_inst, builder, value_map, data_layout);
                builder.ins().return_(&[op]);
            }
        }
        LLVMUnreachable => {
            builder.ins().trap(ir::TrapCode::User(0));
        }
        _ => {
            panic!("unimplemented opcode: {:?}", llvm_opcode);
        }
    }
}

/// Produce a Cretonne Value holding the value of an LLVM IR Constant.
fn materialize_constant(
    llvm_val: LLVMValueRef,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    data_layout: LLVMTargetDataRef,
) -> ir::Value {
    let llvm_kind = unsafe { LLVMGetValueKind(llvm_val) };
    match llvm_kind {
        LLVMUndefValueValueKind => {
            // Cretonne has no undef; emit a zero.
            let ty = translate_type_of(llvm_val, data_layout);
            if ty.is_int() {
                builder.ins().iconst(ty, 0)
            } else if ty.is_bool() {
                builder.ins().bconst(ty, false)
            } else if ty == ir::types::F32 {
                builder.ins().f32const(ir::immediates::Ieee32::with_bits(0))
            } else if ty == ir::types::F64 {
                builder.ins().f64const(ir::immediates::Ieee64::with_bits(0))
            } else {
                panic!("unimplemented undef type: {}", ty);
            }
        }
        LLVMConstantIntValueKind => {
            builder.ins().iconst(
                translate_type_of(llvm_val, data_layout),
                unsafe { LLVMConstIntGetSExtValue(llvm_val) },
            )
        }
        LLVMConstantPointerNullValueKind => {
            builder.ins().iconst(translate_pointer_type(data_layout), 0)
        }
        LLVMFunctionValueKind => {
            let signature = builder.import_signature(translate_sig(
                unsafe { LLVMGetElementType(LLVMTypeOf(llvm_val)) },
                data_layout,
            ));
            let name = translate_function_name(unsafe { LLVMGetValueName(llvm_val) })
                .expect("unimplemented: unusual function names");
            let callee = builder.import_function(ir::ExtFuncData { name, signature });
            builder.ins().func_addr(
                translate_pointer_type(data_layout),
                callee,
            )
        }
        LLVMConstantFPValueKind => {
            let mut loses_info = [0];
            let val = unsafe { LLVMConstRealGetDouble(llvm_val, loses_info.as_mut_ptr()) };
            debug_assert_eq!(loses_info[0], 0);
            match translate_type_of(llvm_val, data_layout) {
                ir::types::F32 => {
                    let f32val = val as f32;
                    builder.ins().f32const(ir::immediates::Ieee32::with_bits(
                        unsafe { mem::transmute(f32val) },
                    ))
                }
                ir::types::F64 => {
                    builder.ins().f64const(ir::immediates::Ieee64::with_bits(
                        unsafe { mem::transmute(val) },
                    ))
                }
                ty => {
                    panic!("unimplemented floating-point constant type: {}", ty);
                }
            }
        }
        _ => {
            panic!("unimplemented constant kind: {:?}", llvm_kind);
        }
    }
}

/// Translate an LLVM IR intrinsic call to Cretonne.
fn translate_intrinsic(
    llvm_inst: LLVMValueRef,
    llvm_callee: LLVMValueRef,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    value_map: &mut HashMap<LLVMValueRef, Variable>,
    data_layout: LLVMTargetDataRef,
) {
    // LLVM's C API doesn't expose intrinsic IDs yet, so we match by name.
    let name = translate_string(unsafe { LLVMGetValueName(llvm_callee) })
        .expect("unimplemented: unusual function names");
    match name.as_ref() {
        "llvm.dbg.declare" |
        "llvm.dbg.value" |
        "llvm.prefetch" |
        "llvm.assume" |
        "llvm.lifetime.start.p0i8" |
        "llvm.lifetime.end.p0i8" |
        "llvm.invariant.start.p0i8" |
        "llvm.invariant.end.p0i8" |
        "llvm.sideeffect" => {
            // For now, just discard this informtion.
        }
        "llvm.invariant.group.barrier" |
        "llvm.expect.i1" |
        "llvm.expect.i8" |
        "llvm.expect.i16" |
        "llvm.expect.i32" |
        "llvm.expect.i64" => {
            // For now, just discard this informtion.
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            def_val(llvm_inst, op, builder, value_map, data_layout);
        }
        "llvm.objectsize.i8.p0i8" |
        "llvm.objectsize.i16.p0i8" |
        "llvm.objectsize.i32.p0i8" |
        "llvm.objectsize.i64.p0i8" => {
            let min = unsafe { LLVMConstIntGetZExtValue(LLVMGetOperand(llvm_inst, 1)) } != 0;
            let result = builder.ins().iconst(
                translate_type_of(llvm_inst, data_layout),
                ir::immediates::Imm64::new(if min { 0 } else { -1 }),
            );
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        "llvm.sqrt.f32" | "llvm.sqrt.f64" => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().sqrt(op);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        "llvm.fabs.f32" | "llvm.fabs.f64" => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().fabs(op);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        "llvm.copysign.f32" |
        "llvm.copysign.f64" => {
            let (lhs, rhs) = binary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().fcopysign(lhs, rhs);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        "llvm.ceil.f32" | "llvm.ceil.f64" => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().ceil(op);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        "llvm.floor.f32" | "llvm.floor.f64" => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().floor(op);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        "llvm.trunc.f32" | "llvm.trunc.f64" => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().trunc(op);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        "llvm.nearbyint.f32" |
        "llvm.nearbyint.f64" => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().nearest(op);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        "llvm.ctpop.i8" | "llvm.ctpop.i16" | "llvm.ctpop.i32" | "llvm.ctpop.i64" => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().popcnt(op);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        "llvm.ctlz.i8" | "llvm.ctlz.i16" | "llvm.ctlz.i32" | "llvm.ctlz.i64" => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().clz(op);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        "llvm.cttz.i8" | "llvm.cttz.i16" | "llvm.cttz.i32" | "llvm.cttz.i64" => {
            let op = unary_operands(llvm_inst, builder, value_map, data_layout);
            let result = builder.ins().ctz(op);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        "llvm.trap" => {
            builder.ins().trap(ir::TrapCode::User(1));
        }
        "llvm.debugtrap" => {
            builder.ins().trap(ir::TrapCode::User(2));
        }
        "llvm.fmuladd.f32" |
        "llvm.fmuladd.f64" => {
            // Cretonne currently has no fma instruction, so just lower these
            // as non-fused operations.
            let (a, b, c) = ternary_operands(llvm_inst, builder, value_map, data_layout);
            let t = builder.ins().fmul(a, b);
            let result = builder.ins().fadd(t, c);
            def_val(llvm_inst, result, builder, value_map, data_layout);
        }
        "llvm.minnum.f32" => {
            translate_intr_libcall(
                "fminf",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.minnum.f64" => {
            translate_intr_libcall(
                "fmin",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.maxnum.f32" => {
            translate_intr_libcall(
                "fmaxf",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.maxnum.f64" => {
            translate_intr_libcall(
                "fmax",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.sin.f32" => {
            translate_intr_libcall(
                "sinf",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.sin.f64" => {
            translate_intr_libcall(
                "sin",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.cos.f32" => {
            translate_intr_libcall(
                "cosf",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.cos.f64" => {
            translate_intr_libcall(
                "cos",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.tan.f32" => {
            translate_intr_libcall(
                "tanf",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.tan.f64" => {
            translate_intr_libcall(
                "tan",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.exp.f32" => {
            translate_intr_libcall(
                "expf",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.exp.f64" => {
            translate_intr_libcall(
                "exp",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.exp2.f32" => {
            translate_intr_libcall(
                "exp2f",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.exp2.f64" => {
            translate_intr_libcall(
                "exp2",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.log.f32" => {
            translate_intr_libcall(
                "logf",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.log.f64" => {
            translate_intr_libcall(
                "log",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.log2.f32" => {
            translate_intr_libcall(
                "log2f",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.log2.f64" => {
            translate_intr_libcall(
                "log2",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.log10.f32" => {
            translate_intr_libcall(
                "log10f",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.log10.f64" => {
            translate_intr_libcall(
                "log10",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.pow.f32" => {
            translate_intr_libcall(
                "powf",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.pow.f64" => {
            translate_intr_libcall(
                "pow",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.rint.f32" => {
            translate_intr_libcall(
                "rintf",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.rint.f64" => {
            translate_intr_libcall(
                "rint",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.round.f32" => {
            translate_intr_libcall(
                "roundf",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.round.f64" => {
            translate_intr_libcall(
                "round",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.fma.f32" => {
            translate_intr_libcall(
                "fmaf",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.fma.f64" => {
            translate_intr_libcall(
                "fma",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.memcpy.p0i8.p0i8.i8" |
        "llvm.memcpy.p0i8.p0i8.i16" |
        "llvm.memcpy.p0i8.p0i8.i32" |
        "llvm.memcpy.p0i8.p0i8.i64" => {
            translate_mem_intrinsic(
                "memcpy",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.memmove.p0i8.p0i8.i8" |
        "llvm.memmove.p0i8.p0i8.i16" |
        "llvm.memmove.p0i8.p0i8.i32" |
        "llvm.memmove.p0i8.p0i8.i64" => {
            translate_mem_intrinsic(
                "memmove",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        "llvm.memset.p0i8.i8" |
        "llvm.memset.p0i8.i16" |
        "llvm.memset.p0i8.i32" |
        "llvm.memset.p0i8.i64" => {
            translate_mem_intrinsic(
                "memset",
                llvm_inst,
                llvm_callee,
                builder,
                value_map,
                data_layout,
            )
        }
        _ => {
            panic!("unimplemented: intrinsic: {}", name);
        }
    }
}

/// Translate an LLVM IR intrinsic call which corresponds directly to a
/// C library call.
fn translate_intr_libcall(
    name: &str,
    llvm_inst: LLVMValueRef,
    llvm_callee: LLVMValueRef,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    value_map: &mut HashMap<LLVMValueRef, Variable>,
    data_layout: LLVMTargetDataRef,
) {
    let num_args = unsafe { LLVMGetNumArgOperands(llvm_inst) } as usize;
    let mut args = Vec::with_capacity(num_args);
    for i in 0..num_args {
        debug_assert_eq!(i as libc::c_uint as usize, i);
        args.push(use_val(
            unsafe { LLVMGetOperand(llvm_inst, i as libc::c_uint) },
            builder,
            value_map,
            data_layout,
        ));
    }

    let name = ir::FunctionName::new(name);
    let signature = builder.import_signature(translate_sig(
        unsafe { LLVMGetElementType(LLVMTypeOf(llvm_callee)) },
        data_layout,
    ));
    let data = ir::ExtFuncData { name, signature };
    let callee = builder.import_function(data);
    let call = builder.ins().call(callee, &args);
    let result = {
        let results = builder.inst_results(call);
        debug_assert_eq!(results.len(), 1);
        results[0]
    };
    def_val(llvm_inst, result, builder, value_map, data_layout);
}

/// Translate an LLVM IR llvm.memcpy/memmove/memset call.
fn translate_mem_intrinsic(
    name: &str,
    llvm_inst: LLVMValueRef,
    llvm_callee: LLVMValueRef,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    value_map: &mut HashMap<LLVMValueRef, Variable>,
    data_layout: LLVMTargetDataRef,
) {
    let pointer_type = translate_pointer_type(data_layout);
    let llvm_len_arg = unsafe { LLVMGetOperand(llvm_inst, 2) };
    let mut len_arg = use_val(llvm_len_arg, builder, value_map, data_layout);
    let orig_type = translate_type_of(llvm_len_arg, data_layout);
    if orig_type != pointer_type {
        if orig_type.bits() > pointer_type.bits() {
            len_arg = builder.ins().ireduce(pointer_type, len_arg)
        } else if orig_type.bits() < pointer_type.bits() {
            len_arg = builder.ins().uextend(pointer_type, len_arg)
        }
    }
    let args = [
        use_val(
            unsafe { LLVMGetOperand(llvm_inst, 0) },
            builder,
            value_map,
            data_layout,
        ),
        use_val(
            unsafe { LLVMGetOperand(llvm_inst, 1) },
            builder,
            value_map,
            data_layout,
        ),
        len_arg,
    ];

    let name = ir::FunctionName::new(name);
    let mut sig = translate_sig(
        unsafe { LLVMGetElementType(LLVMTypeOf(llvm_callee)) },
        data_layout,
    );
    // Discard the alignment hint for now.
    // This also drops the volatile flag, which is safe as long as Cretonne
    // doesn't start optimizing these libcalls.
    sig.params.resize(3, ir::AbiParam::new(ir::types::VOID));
    let signature = builder.import_signature(sig);
    let data = ir::ExtFuncData { name, signature };
    let callee = builder.import_function(data);
    builder.ins().call(callee, &args);
}

/// Record PHI uses and defs for a branch from `llvm_bb` to `llvm_succ`.
fn handle_phi_operands(
    llvm_bb: LLVMBasicBlockRef,
    llvm_succ: LLVMBasicBlockRef,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    value_map: &mut HashMap<LLVMValueRef, Variable>,
    data_layout: LLVMTargetDataRef,
) {
    let mut defs = Vec::new();

    let mut llvm_inst = unsafe { LLVMGetFirstInstruction(llvm_succ) };
    while !llvm_inst.is_null() {
        if let LLVMPHI = unsafe { LLVMGetInstructionOpcode(llvm_inst) } {
            // TODO: We'd really want getBasicBlockIndex or getIncomingValueForBlock
            // here, but LLVM's C API doesn't expose those currently.
            let mut llvm_val = ptr::null_mut();
            for i in 0..unsafe { LLVMCountIncoming(llvm_inst) } {
                if unsafe { LLVMGetIncomingBlock(llvm_inst, i) } == llvm_bb {
                    llvm_val = unsafe { LLVMGetIncomingValue(llvm_inst, i) };
                    break;
                }
            }

            let val = use_val(llvm_val, builder, value_map, data_layout);
            defs.push((llvm_inst, val));
        } else {
            break;
        }
        llvm_inst = unsafe { LLVMGetNextInstruction(llvm_inst) };
    }

    for (llvm_inst, val) in defs {
        def_val(llvm_inst, val, builder, value_map, data_layout);
    }
}

/// Translate a GetElementPtr index operand into Cretonne IL.
fn translate_gep_index(
    llvm_gepty: LLVMTypeRef,
    ptr: ir::Value,
    pointer_type: ir::Type,
    index: LLVMValueRef,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    value_map: &mut HashMap<LLVMValueRef, Variable>,
    data_layout: LLVMTargetDataRef,
) -> (ir::Value, LLVMTypeRef) {
    // TODO: We'd really want gep_type_iterator etc. here, but
    // LLVM's C API doesn't expose those currently.
    let (offset, ty) = match unsafe { LLVMGetTypeKind(llvm_gepty) } {
        LLVMStructTypeKind => {
            let i = unsafe { LLVMConstIntGetZExtValue(index) };
            debug_assert_eq!(u64::from(i as libc::c_uint), i);
            let offset = unsafe { LLVMOffsetOfElement(data_layout, llvm_gepty, i as libc::c_uint) };
            (
                builder.ins().iconst(
                    pointer_type,
                    ir::immediates::Imm64::new(offset as i64),
                ),
                unsafe { LLVMStructGetTypeAtIndex(llvm_gepty, i as libc::c_uint) },
            )
        }
        LLVMPointerTypeKind | LLVMArrayTypeKind | LLVMVectorTypeKind => {
            let index_type = translate_type_of(index, data_layout);
            let mut x = use_val(index, builder, value_map, data_layout);
            if index_type != pointer_type {
                if index_type.bits() < pointer_type.bits() {
                    x = builder.ins().sextend(pointer_type, x);
                } else {
                    x = builder.ins().ireduce(pointer_type, x);
                }
            }

            let llvm_eltty = unsafe { LLVMGetElementType(llvm_gepty) };
            let size = unsafe { LLVMABISizeOfType(data_layout, llvm_eltty) };
            if size != 1 {
                let scale = builder.ins().iconst(
                    pointer_type,
                    ir::immediates::Imm64::new(size as i64),
                );
                x = builder.ins().imul(x, scale)
            }

            (x, llvm_eltty)
        }
        _ => panic!("unexpected GEP indexing type: {:?}", llvm_gepty),
    };
    (builder.ins().iadd(ptr, offset), ty)
}

/// Emit a Cretonne jump to the destination corresponding to `llvm_succ`, if
/// one is needed.
fn jump(
    llvm_bb: LLVMBasicBlockRef,
    llvm_succ: LLVMBasicBlockRef,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    value_map: &mut HashMap<LLVMValueRef, Variable>,
    ebb_map: &HashMap<LLVMBasicBlockRef, ir::Ebb>,
    data_layout: LLVMTargetDataRef,
) {
    if let Some(ebb) = ebb_map.get(&llvm_succ) {
        handle_phi_operands(llvm_bb, llvm_succ, builder, value_map, data_layout);
        builder.ins().jump(*ebb, &[]);
    }
}

/// Record a "use" of an LLVM IR value.
fn use_val(
    llvm_val: LLVMValueRef,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    value_map: &mut HashMap<LLVMValueRef, Variable>,
    data_layout: LLVMTargetDataRef,
) -> ir::Value {
    if unsafe { LLVMIsConstant(llvm_val) } != 0 {
        materialize_constant(llvm_val, builder, data_layout)
    } else {
        let num_values = value_map.len();
        let var = match value_map.entry(llvm_val) {
            hash_map::Entry::Occupied(entry) => *entry.get(),
            hash_map::Entry::Vacant(entry) => {
                let var = Variable::new(num_values);
                builder.declare_var(var, translate_type_of(llvm_val, data_layout));
                *entry.insert(var)
            }
        };
        builder.use_var(var)
    }
}

/// Record a "definition" of an LLVM IR value.
fn def_val(
    llvm_val: LLVMValueRef,
    value: ir::Value,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    value_map: &mut HashMap<LLVMValueRef, Variable>,
    data_layout: LLVMTargetDataRef,
) {
    let num_values = value_map.len();
    let var = match value_map.entry(llvm_val) {
        hash_map::Entry::Occupied(entry) => *entry.get(),
        hash_map::Entry::Vacant(entry) => {
            let var = Variable::new(num_values);
            builder.declare_var(var, translate_type_of(llvm_val, data_layout));
            *entry.insert(var)
        }
    };
    builder.def_var(var, value)
}

/// Translate the operands for a unary operation.
fn unary_operands(
    llvm_inst: LLVMValueRef,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    value_map: &mut HashMap<LLVMValueRef, Variable>,
    data_layout: LLVMTargetDataRef,
) -> ir::Value {
    use_val(
        unsafe { LLVMGetOperand(llvm_inst, 0) },
        builder,
        value_map,
        data_layout,
    )
}

/// Translate the operands for a binary operation.
fn binary_operands(
    llvm_inst: LLVMValueRef,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    value_map: &mut HashMap<LLVMValueRef, Variable>,
    data_layout: LLVMTargetDataRef,
) -> (ir::Value, ir::Value) {
    (
        use_val(
            unsafe { LLVMGetOperand(llvm_inst, 0) },
            builder,
            value_map,
            data_layout,
        ),
        use_val(
            unsafe { LLVMGetOperand(llvm_inst, 1) },
            builder,
            value_map,
            data_layout,
        ),
    )
}

/// Translate the operands for a binary operation.
fn ternary_operands(
    llvm_inst: LLVMValueRef,
    builder: &mut cton_frontend::FunctionBuilder<Variable>,
    value_map: &mut HashMap<LLVMValueRef, Variable>,
    data_layout: LLVMTargetDataRef,
) -> (ir::Value, ir::Value, ir::Value) {
    (
        use_val(
            unsafe { LLVMGetOperand(llvm_inst, 0) },
            builder,
            value_map,
            data_layout,
        ),
        use_val(
            unsafe { LLVMGetOperand(llvm_inst, 1) },
            builder,
            value_map,
            data_layout,
        ),
        use_val(
            unsafe { LLVMGetOperand(llvm_inst, 2) },
            builder,
            value_map,
            data_layout,
        ),
    )
}

/// Return the successor of `llvm_inst` with index `i`.
fn successor(llvm_inst: LLVMValueRef, i: ::libc::c_uint) -> LLVMBasicBlockRef {
    unsafe { LLVMGetSuccessor(llvm_inst, i) }
}

/// Translate an LLVM integer predicate into a Cretonne one.
fn translate_icmp_code(llvm_pred: LLVMIntPredicate) -> ir::condcodes::IntCC {
    match llvm_pred {
        LLVMIntEQ => ir::condcodes::IntCC::Equal,
        LLVMIntNE => ir::condcodes::IntCC::NotEqual,
        LLVMIntUGT => ir::condcodes::IntCC::UnsignedGreaterThan,
        LLVMIntUGE => ir::condcodes::IntCC::UnsignedGreaterThanOrEqual,
        LLVMIntULT => ir::condcodes::IntCC::UnsignedLessThan,
        LLVMIntULE => ir::condcodes::IntCC::UnsignedLessThanOrEqual,
        LLVMIntSGT => ir::condcodes::IntCC::SignedGreaterThan,
        LLVMIntSGE => ir::condcodes::IntCC::SignedGreaterThanOrEqual,
        LLVMIntSLT => ir::condcodes::IntCC::SignedLessThan,
        LLVMIntSLE => ir::condcodes::IntCC::SignedLessThanOrEqual,
    }
}

/// Translate an LLVM floating-point predicate into a Cretonne one.
fn translate_fcmp_code(llvm_pred: LLVMRealPredicate) -> ir::condcodes::FloatCC {
    match llvm_pred {
        LLVMRealOEQ => ir::condcodes::FloatCC::Equal,
        LLVMRealOGT => ir::condcodes::FloatCC::GreaterThan,
        LLVMRealOGE => ir::condcodes::FloatCC::GreaterThanOrEqual,
        LLVMRealOLT => ir::condcodes::FloatCC::LessThan,
        LLVMRealOLE => ir::condcodes::FloatCC::LessThanOrEqual,
        LLVMRealONE => ir::condcodes::FloatCC::OrderedNotEqual,
        LLVMRealORD => ir::condcodes::FloatCC::Ordered,
        LLVMRealUNO => ir::condcodes::FloatCC::Unordered,
        LLVMRealUEQ => ir::condcodes::FloatCC::UnorderedOrEqual,
        LLVMRealUGT => ir::condcodes::FloatCC::UnorderedOrGreaterThan,
        LLVMRealUGE => ir::condcodes::FloatCC::UnorderedOrGreaterThanOrEqual,
        LLVMRealULT => ir::condcodes::FloatCC::UnorderedOrLessThan,
        LLVMRealULE => ir::condcodes::FloatCC::UnorderedOrLessThanOrEqual,
        LLVMRealUNE => ir::condcodes::FloatCC::NotEqual,
        LLVMRealPredicateFalse | LLVMRealPredicateTrue => panic!(),
    }
}

/// Translate LLVM load/store information into Cretonne `MemFlags`.
fn translate_memflags(
    llvm_inst: LLVMValueRef,
    llvm_ty: LLVMTypeRef,
    data_layout: LLVMTargetDataRef,
) -> ir::MemFlags {
    if unsafe { LLVMGetVolatile(llvm_inst) } != 0 {
        panic!("unimplemented: volatile memory reference");
    }
    if unsafe { LLVMGetOrdering(llvm_inst) } != LLVMAtomicOrderingNotAtomic {
        panic!("unimplemented: atomic memory reference");
    }

    let mut flags = ir::MemFlags::new();

    // LLVM IR has UB.
    flags.set_notrap();

    if u64::from(unsafe { LLVMGetAlignment(llvm_inst) }) >=
        unsafe { LLVMABISizeOfType(data_layout, llvm_ty) }
    {
        flags.set_aligned();
    }

    flags
}

/// Translate the type of an LLVM Value into a Cretonne type.
fn translate_type_of(llvm_val: LLVMValueRef, data_layout: LLVMTargetDataRef) -> ir::Type {
    translate_type(unsafe { LLVMTypeOf(llvm_val) }, data_layout)
}
