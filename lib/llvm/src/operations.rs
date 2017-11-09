//! Translation from LLVM IR operations to Cretonne IL instructions.

use cretonne::ir::{self, InstBuilder, Ebb};
use std::collections::hash_map;
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

use translate::{translate_symbol_name, translate_string};
use context::{Context, Variable};
use types::{translate_type, translate_pointer_type, translate_sig};

/// Translate the incoming parameters for `llvm_func` into Cretonne values
/// defined in the entry block.
pub fn translate_function_params(llvm_func: LLVMValueRef, entry_ebb: Ebb, ctx: &mut Context) {
    ctx.builder.append_ebb_params_for_function_params(entry_ebb);
    for i in 0..unsafe { LLVMCountParams(llvm_func) } {
        let llvm_param = unsafe { LLVMGetParam(llvm_func, i) };
        let val = ctx.builder.ebb_params(entry_ebb)[i as usize];
        def_val(llvm_param, val, ctx);
    }
}

/// Translate `llvm_inst`, which is a normal instruction, to Cretonne IL
/// instructions.
pub fn translate_inst(llvm_bb: LLVMBasicBlockRef, llvm_inst: LLVMValueRef, ctx: &mut Context) {
    let llvm_opcode = unsafe { LLVMGetInstructionOpcode(llvm_inst) };
    if let Some(result) = translate_operation(llvm_bb, llvm_opcode, llvm_inst, ctx) {
        def_val(llvm_inst, result, ctx);
    }
}

/// Translate `llvm_val`, which is either an `Instruction` or a `ConstantExpr`,
/// with `llvm_opcode` as the opcode.
fn translate_operation(
    llvm_bb: LLVMBasicBlockRef,
    llvm_opcode: LLVMOpcode,
    llvm_val: LLVMValueRef,
    ctx: &mut Context,
) -> Option<ir::Value> {
    Some(match llvm_opcode {
        LLVMPHI => {
            // Nothing to do. Phis are handled elsewhere.
            return None;
        }
        LLVMAdd => {
            match binary_operands_r_ri(llvm_val, ctx) {
                RegImmOperands::Bool(lhs, rhs) => ctx.builder.ins().bxor(lhs, rhs),
                RegImmOperands::RegReg(lhs, rhs) => ctx.builder.ins().iadd(lhs, rhs),
                RegImmOperands::RegImm(lhs, rhs) => ctx.builder.ins().iadd_imm(lhs, rhs),
            }
        }
        LLVMSub => {
            // TODO: use irsub_imm when applicable.
            match binary_operands_r_ri(llvm_val, ctx) {
                RegImmOperands::Bool(lhs, rhs) => ctx.builder.ins().bxor_not(lhs, rhs),
                RegImmOperands::RegReg(lhs, rhs) => ctx.builder.ins().isub(lhs, rhs),
                RegImmOperands::RegImm(lhs, rhs) => {
                    // Cretonne has no isub_imm; it uses iadd_imm with a
                    // negated immediate instead.
                    let raw_rhs: i64 = rhs.into();
                    ctx.builder.ins().iadd_imm(
                        lhs,
                        ir::immediates::Imm64::from(
                            raw_rhs.wrapping_neg(),
                        ),
                    )
                }
            }
        }
        LLVMMul => {
            match binary_operands_r_ri(llvm_val, ctx) {
                RegImmOperands::Bool(lhs, rhs) => ctx.builder.ins().band(lhs, rhs),
                RegImmOperands::RegReg(lhs, rhs) => ctx.builder.ins().imul(lhs, rhs),
                RegImmOperands::RegImm(lhs, rhs) => ctx.builder.ins().imul_imm(lhs, rhs),
            }
        }
        LLVMSDiv => {
            match binary_operands_r_ri(llvm_val, ctx) {
                RegImmOperands::Bool(lhs, _rhs) => lhs,
                RegImmOperands::RegReg(lhs, rhs) => ctx.builder.ins().sdiv(lhs, rhs),
                RegImmOperands::RegImm(lhs, rhs) => ctx.builder.ins().sdiv_imm(lhs, rhs),
            }
        }
        LLVMUDiv => {
            match binary_operands_r_ri(llvm_val, ctx) {
                RegImmOperands::Bool(lhs, _rhs) => lhs,
                RegImmOperands::RegReg(lhs, rhs) => ctx.builder.ins().udiv(lhs, rhs),
                RegImmOperands::RegImm(lhs, rhs) => ctx.builder.ins().udiv_imm(lhs, rhs),
            }
        }
        LLVMSRem => {
            match binary_operands_r_ri(llvm_val, ctx) {
                RegImmOperands::Bool(_lhs, _rhs) => {
                    let ty = translate_type_of(llvm_val, ctx.dl);
                    ctx.builder.ins().bconst(ty, false)
                }
                RegImmOperands::RegReg(lhs, rhs) => ctx.builder.ins().srem(lhs, rhs),
                RegImmOperands::RegImm(lhs, rhs) => ctx.builder.ins().srem_imm(lhs, rhs),
            }
        }
        LLVMURem => {
            match binary_operands_r_ri(llvm_val, ctx) {
                RegImmOperands::Bool(_lhs, _rhs) => {
                    let ty = translate_type_of(llvm_val, ctx.dl);
                    ctx.builder.ins().bconst(ty, false)
                }
                RegImmOperands::RegReg(lhs, rhs) => ctx.builder.ins().urem(lhs, rhs),
                RegImmOperands::RegImm(lhs, rhs) => ctx.builder.ins().urem_imm(lhs, rhs),
            }
        }
        LLVMAShr => {
            match binary_operands_r_ri(llvm_val, ctx) {
                RegImmOperands::Bool(lhs, _rhs) => lhs,
                RegImmOperands::RegReg(lhs, rhs) => ctx.builder.ins().sshr(lhs, rhs),
                RegImmOperands::RegImm(lhs, rhs) => ctx.builder.ins().sshr_imm(lhs, rhs),
            }
        }
        LLVMLShr => {
            match binary_operands_r_ri(llvm_val, ctx) {
                RegImmOperands::Bool(lhs, _rhs) => lhs,
                RegImmOperands::RegReg(lhs, rhs) => ctx.builder.ins().ushr(lhs, rhs),
                RegImmOperands::RegImm(lhs, rhs) => ctx.builder.ins().ushr_imm(lhs, rhs),
            }
        }
        LLVMShl => {
            match binary_operands_r_ri(llvm_val, ctx) {
                RegImmOperands::Bool(lhs, _rhs) => lhs,
                RegImmOperands::RegReg(lhs, rhs) => ctx.builder.ins().ishl(lhs, rhs),
                RegImmOperands::RegImm(lhs, rhs) => ctx.builder.ins().ishl_imm(lhs, rhs),
            }
        }
        LLVMAnd => {
            match binary_operands_r_ri(llvm_val, ctx) {
                RegImmOperands::Bool(lhs, rhs) |
                RegImmOperands::RegReg(lhs, rhs) => ctx.builder.ins().band(lhs, rhs),
                RegImmOperands::RegImm(lhs, rhs) => ctx.builder.ins().band_imm(lhs, rhs),
            }
        }
        LLVMOr => {
            match binary_operands_r_ri(llvm_val, ctx) {
                RegImmOperands::Bool(lhs, rhs) |
                RegImmOperands::RegReg(lhs, rhs) => ctx.builder.ins().bor(lhs, rhs),
                RegImmOperands::RegImm(lhs, rhs) => ctx.builder.ins().bor_imm(lhs, rhs),
            }
        }
        LLVMXor => {
            match binary_operands_r_ri(llvm_val, ctx) {
                RegImmOperands::Bool(lhs, rhs) |
                RegImmOperands::RegReg(lhs, rhs) => ctx.builder.ins().bxor(lhs, rhs),
                RegImmOperands::RegImm(lhs, rhs) => ctx.builder.ins().bxor_imm(lhs, rhs),
            }
        }
        LLVMFAdd => {
            let (lhs, rhs) = binary_operands(llvm_val, ctx);
            ctx.builder.ins().fadd(lhs, rhs)
        }
        LLVMFSub => {
            let (lhs, rhs) = binary_operands(llvm_val, ctx);
            ctx.builder.ins().fsub(lhs, rhs)
        }
        LLVMFMul => {
            let (lhs, rhs) = binary_operands(llvm_val, ctx);
            ctx.builder.ins().fmul(lhs, rhs)
        }
        LLVMFDiv => {
            let (lhs, rhs) = binary_operands(llvm_val, ctx);
            ctx.builder.ins().fdiv(lhs, rhs)
        }
        LLVMFRem => {
            let (lhs, rhs) = binary_operands(llvm_val, ctx);
            let ty = translate_type_of(llvm_val, ctx.dl);
            let mut sig = ir::Signature::new(ir::CallConv::Native);
            sig.params.resize(2, ir::AbiParam::new(ty));
            sig.returns.push(ir::AbiParam::new(ty));
            let data = ir::ExtFuncData {
                name: ir::ExternalName::new(match ty {
                    ir::types::F32 => "fmodf",
                    ir::types::F64 => "fmod",
                    _ => panic!("frem unimplemented for type {:?}", ty),
                }),
                signature: ctx.builder.import_signature(sig),
            };
            let callee = ctx.builder.import_function(data);
            let call = ctx.builder.ins().call(callee, &[lhs, rhs]);
            ctx.builder.inst_results(call)[0]
        }
        LLVMICmp => {
            let llvm_condcode = unsafe { LLVMGetICmpPredicate(llvm_val) };
            let condcode = translate_icmp_code(llvm_condcode);
            match binary_operands_r_ri(llvm_val, ctx) {
                RegImmOperands::Bool(lhs, rhs) => translate_bool_cmp(condcode, lhs, rhs, ctx),
                RegImmOperands::RegReg(lhs, rhs) => ctx.builder.ins().icmp(condcode, lhs, rhs),
                RegImmOperands::RegImm(lhs, rhs) => ctx.builder.ins().icmp_imm(condcode, lhs, rhs),
            }
        }
        LLVMFCmp => {
            let (lhs, rhs) = binary_operands(llvm_val, ctx);
            let condcode = unsafe { LLVMGetFCmpPredicate(llvm_val) };
            match condcode {
                LLVMRealPredicateFalse => ctx.builder.ins().bconst(ir::types::B1, false),
                LLVMRealPredicateTrue => ctx.builder.ins().bconst(ir::types::B1, true),
                _ => {
                    ctx.builder.ins().fcmp(
                        translate_fcmp_code(condcode),
                        lhs,
                        rhs,
                    )
                }
            }
        }
        LLVMTrunc | LLVMZExt => {
            let llvm_op = unsafe { LLVMGetOperand(llvm_val, 0) };
            let op = use_val(llvm_op, ctx);
            let from = translate_type_of(llvm_op, ctx.dl);
            let to = translate_type_of(llvm_val, ctx.dl);
            unsigned_cast(from, to, op, ctx)
        }
        LLVMSExt => {
            let llvm_op = unsafe { LLVMGetOperand(llvm_val, 0) };
            let op = use_val(llvm_op, ctx);
            let from = translate_type_of(llvm_op, ctx.dl);
            let to = translate_type_of(llvm_val, ctx.dl);
            if from.is_bool() {
                ctx.builder.ins().bmask(to, op)
            } else {
                ctx.builder.ins().sextend(to, op)
            }
        }
        LLVMFPToSI => {
            let op = unary_operands(llvm_val, ctx);
            let ty = translate_type_of(llvm_val, ctx.dl);
            if ty.is_bool() {
                let sint = ctx.builder.ins().fcvt_to_sint(ir::types::I32, op);
                ctx.builder.ins().icmp_imm(
                    ir::condcodes::IntCC::NotEqual,
                    sint,
                    0,
                )
            } else {
                ctx.builder.ins().fcvt_to_sint(ty, op)
            }
        }
        LLVMFPToUI => {
            let op = unary_operands(llvm_val, ctx);
            let ty = translate_type_of(llvm_val, ctx.dl);
            if ty.is_bool() {
                let uint = ctx.builder.ins().fcvt_to_uint(ir::types::I32, op);
                ctx.builder.ins().icmp_imm(
                    ir::condcodes::IntCC::NotEqual,
                    uint,
                    0,
                )
            } else {
                ctx.builder.ins().fcvt_to_uint(ty, op)
            }
        }
        LLVMSIToFP => {
            let llvm_op = unsafe { LLVMGetOperand(llvm_val, 0) };
            let op = use_val(llvm_op, ctx);
            let from = translate_type_of(llvm_op, ctx.dl);
            let to = translate_type_of(llvm_val, ctx.dl);
            if from.is_bool() {
                let sint = ctx.builder.ins().bmask(ir::types::I32, op);
                ctx.builder.ins().fcvt_from_sint(to, sint)
            } else {
                ctx.builder.ins().fcvt_from_sint(to, op)
            }
        }
        LLVMUIToFP => {
            let llvm_op = unsafe { LLVMGetOperand(llvm_val, 0) };
            let op = use_val(llvm_op, ctx);
            let from = translate_type_of(llvm_op, ctx.dl);
            let to = translate_type_of(llvm_val, ctx.dl);
            if from.is_bool() {
                let uint = ctx.builder.ins().bint(ir::types::I32, op);
                ctx.builder.ins().fcvt_from_uint(to, uint)
            } else {
                ctx.builder.ins().fcvt_from_uint(to, op)
            }
        }
        LLVMFPTrunc => {
            let op = unary_operands(llvm_val, ctx);
            let ty = translate_type_of(llvm_val, ctx.dl);
            ctx.builder.ins().fdemote(ty, op)
        }
        LLVMFPExt => {
            let op = unary_operands(llvm_val, ctx);
            let ty = translate_type_of(llvm_val, ctx.dl);
            ctx.builder.ins().fpromote(ty, op)
        }
        LLVMBitCast => {
            let llvm_op = unsafe { LLVMGetOperand(llvm_val, 0) };
            let op = use_val(llvm_op, ctx);
            let from = translate_type_of(llvm_op, ctx.dl);
            let to = translate_type_of(llvm_val, ctx.dl);
            if from == to {
                // No-op bitcast.
                op
            } else {
                ctx.builder.ins().bitcast(to, op)
            }
        }
        LLVMPtrToInt | LLVMIntToPtr => {
            let llvm_op = unsafe { LLVMGetOperand(llvm_val, 0) };
            let op = use_val(llvm_op, ctx);
            let from = translate_type_of(llvm_op, ctx.dl);
            let to = translate_type_of(llvm_val, ctx.dl);
            unsigned_cast(from, to, op, ctx)
        }
        LLVMSelect => {
            let (c, t, f) = ternary_operands(llvm_val, ctx);
            ctx.builder.ins().select(c, t, f)
        }
        LLVMLoad => {
            let llvm_ty = unsafe { LLVMTypeOf(llvm_val) };
            let flags = translate_memflags(llvm_val, llvm_ty, ctx);
            let op = unary_operands(llvm_val, ctx);
            let ty = translate_type(llvm_ty, ctx.dl);
            if ty.is_bool() {
                let load = ctx.builder.ins().load(ir::types::I8, flags, op, 0);
                unsigned_cast(ir::types::I8, ty, load, ctx)
            } else {
                ctx.builder.ins().load(ty, flags, op, 0)
            }
        }
        LLVMStore => {
            let llvm_ty = unsafe { LLVMTypeOf(LLVMGetOperand(llvm_val, 0)) };
            let flags = translate_memflags(llvm_val, llvm_ty, ctx);
            let (val, ptr) = binary_operands(llvm_val, ctx);
            let ty = translate_type(llvm_ty, ctx.dl);
            let store_val = if ty.is_bool() {
                unsigned_cast(ty, ir::types::I8, val, ctx)
            } else {
                val
            };
            ctx.builder.ins().store(flags, store_val, ptr, 0);
            return None;
        }
        LLVMBr => {
            if unsafe { LLVMIsConditional(llvm_val) } != 0 {
                let cond = unary_operands(llvm_val, ctx);
                let llvm_true_succ = successor(llvm_val, 0);
                let llvm_false_succ = successor(llvm_val, 1);
                let llvm_next_bb = unsafe { LLVMGetNextBasicBlock(llvm_bb) };
                // A conditional branch in Cretonne always falls through in
                // the not-taken case, so test whether either successor of
                // the LLVM IR conditional branch can be a fallthrough. If not,
                // an unconditional branch can be added.
                if llvm_next_bb == llvm_true_succ {
                    // It's valid for both destinations to be fallthroughs.
                    if llvm_next_bb != llvm_false_succ {
                        handle_phi_operands(llvm_bb, llvm_false_succ, ctx);
                        ctx.builder.ins().brz(
                            cond,
                            ctx.ebb_map[&llvm_false_succ],
                            &[],
                        );
                    }
                    jump(llvm_bb, llvm_true_succ, ctx);
                } else {
                    handle_phi_operands(llvm_bb, llvm_true_succ, ctx);
                    ctx.builder.ins().brnz(
                        cond,
                        ctx.ebb_map[&llvm_true_succ],
                        &[],
                    );
                    jump(llvm_bb, llvm_false_succ, ctx);
                }
            } else {
                let llvm_succ = successor(llvm_val, 0);
                jump(llvm_bb, llvm_succ, ctx);
            }
            return None;
        }
        LLVMSwitch => {
            // TODO: We'd really want getNumCases or ConstCaseIt here, but
            // LLVM's C API doesn't expose those currently.
            let num_cases = unsafe { LLVMGetNumOperands(llvm_val) / 2 - 1 };
            let mut data = ir::JumpTableData::with_capacity(num_cases as usize);
            for i in 0..num_cases {
                let llvm_key = unsafe { LLVMGetOperand(llvm_val, ((i as libc::c_uint + 1) * 2)) };
                if unsafe { LLVMConstIntGetZExtValue(llvm_key) } != i as u64 {
                    panic!("unimplemented: switches with non-sequential cases");
                }
                let llvm_case = unsafe { LLVMGetSuccessor(llvm_val, (i + 1) as libc::c_uint) };
                data.push_entry(ctx.ebb_map[&llvm_case]);
            }
            let jt = ctx.builder.create_jump_table(data);
            let llvm_op = unsafe { LLVMGetOperand(llvm_val, 0) };
            let op = use_val(llvm_op, ctx);
            let op_ty = translate_type_of(llvm_op, ctx.dl);
            let index = if op_ty.is_bool() {
                unsigned_cast(op_ty, ir::types::I8, op, ctx)
            } else {
                op
            };
            ctx.builder.ins().br_table(index, jt);
            let llvm_default = unsafe { LLVMGetSwitchDefaultDest(llvm_val) };
            jump(llvm_bb, llvm_default, ctx);
            return None;
        }
        LLVMAlloca => {
            let llvm_op = unsafe { LLVMGetOperand(llvm_val, 0) };
            let align = unsafe { LLVMGetAlignment(llvm_val) };
            let llvm_allocty = unsafe { LLVMGetAllocatedType(llvm_val) };
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
            if align > unsafe { LLVMABIAlignmentOfType(ctx.dl, llvm_allocty) } {
                panic!("unimplemented: supernaturally aligned alloca");
            }
            let size = unsafe { LLVMConstIntGetZExtValue(llvm_op) } *
                unsafe { LLVMABISizeOfType(ctx.dl, llvm_allocty) };
            if u64::from(size as u32) != size {
                panic!("unimplemented: alloca size computation doesn't fit in u32");
            }
            let stack_slot_data = ir::StackSlotData::new(ir::StackSlotKind::Local, size as u32);
            let stack_slot = ctx.builder.create_stack_slot(stack_slot_data);
            let pointer_type = translate_pointer_type(ctx.dl);
            ctx.builder.ins().stack_addr(pointer_type, stack_slot, 0)
        }
        LLVMGetElementPtr => {
            let pointer_type = translate_pointer_type(ctx.dl);
            let llvm_ptr = unsafe { LLVMGetOperand(llvm_val, 0) };
            let mut llvm_gepty = unsafe { LLVMTypeOf(llvm_ptr) };
            let mut ptr = use_val(llvm_ptr, ctx);
            let mut imm = 0;
            for i in 1..unsafe { LLVMGetNumOperands(llvm_val) } as libc::c_uint {
                let index = unsafe { LLVMGetOperand(llvm_val, i) };
                let (new_llvm_gepty, new_ptr, new_imm) =
                    translate_gep_index(llvm_gepty, ptr, imm, pointer_type, index, ctx);
                llvm_gepty = new_llvm_gepty;
                ptr = new_ptr;
                imm = new_imm;
            }
            if imm != 0 {
                ptr = ctx.builder.ins().iadd_imm(ptr, imm);
            }
            ptr
        }
        LLVMCall => {
            let num_args = unsafe { LLVMGetNumArgOperands(llvm_val) } as usize;
            let callee_index = unsafe { LLVMGetNumOperands(llvm_val) } - 1;
            let llvm_callee = unsafe { LLVMGetOperand(llvm_val, callee_index as libc::c_uint) };

            if unsafe { LLVMGetIntrinsicID(llvm_callee) } != 0 {
                return translate_intrinsic(llvm_val, llvm_callee, ctx);
            }

            let llvm_functy = unsafe { LLVMGetElementType(LLVMTypeOf(llvm_callee)) };
            if unsafe { LLVMIsFunctionVarArg(llvm_functy) } != 0 &&
                unsafe { LLVMCountParamTypes(llvm_functy) } as usize != num_args
            {
                panic!("unimplemented: variadic arguments");
            }

            // Fast and cold are not ABI-exposed, so we can handle them however
            // we like. Just handle them the same as normal calls for now.
            let callconv = unsafe { LLVMGetInstructionCallConv(llvm_val) };
            if callconv != LLVMCCallConv as libc::c_uint &&
                callconv != LLVMFastCallConv as libc::c_uint &&
                callconv != LLVMColdCallConv as libc::c_uint
            {
                panic!("unimplemented calling convention: {}", callconv);
            }
            // Collect the call arguments.
            // TODO: Implement ABI-exposed argument attributes such as byval,
            // signext, zeroext, inreg, inalloca, align.
            let mut args = Vec::with_capacity(num_args);
            for i in 0..num_args {
                debug_assert_eq!(i as libc::c_uint as usize, i);
                args.push(use_val(
                    unsafe { LLVMGetOperand(llvm_val, i as libc::c_uint) },
                    ctx,
                ));
            }
            let signature = ctx.builder.import_signature(
                translate_sig(llvm_functy, ctx.dl),
            );
            let name = translate_symbol_name(unsafe { LLVMGetValueName(llvm_callee) })
                .expect("unimplemented: unusual function names");
            let call = if unsafe { LLVMGetValueKind(llvm_callee) } == LLVMFunctionValueKind {
                let data = ir::ExtFuncData { name, signature };
                let callee = ctx.builder.import_function(data);
                ctx.builder.ins().call(callee, &args)
            } else {
                let callee = use_val(llvm_callee, ctx);
                ctx.builder.ins().call_indirect(signature, callee, &args)
            };
            if translate_type_of(llvm_val, ctx.dl) != ir::types::VOID {
                let results = ctx.builder.inst_results(call);
                debug_assert_eq!(results.len(), 1);
                results[0]
            } else {
                return None;
            }
        }
        LLVMRet => {
            if unsafe { LLVMGetNumOperands(llvm_val) } == 0 {
                ctx.builder.ins().return_(&[]);
            } else {
                // TODO: multiple return values
                let op = unary_operands(llvm_val, ctx);
                ctx.builder.ins().return_(&[op]);
            }
            return None;
        }
        LLVMUnreachable => {
            ctx.builder.ins().trap(ir::TrapCode::User(0));
            return None;
        }
        _ => panic!("unimplemented opcode: {:?}", llvm_opcode),
    })
}

/// Produce a Cretonne Value holding the value of an LLVM IR Constant.
fn materialize_constant(llvm_val: LLVMValueRef, ctx: &mut Context) -> ir::Value {
    let llvm_kind = unsafe { LLVMGetValueKind(llvm_val) };
    match llvm_kind {
        LLVMUndefValueValueKind => {
            // Cretonne has no undef; emit a zero.
            let ty = translate_type_of(llvm_val, ctx.dl);
            if ty.is_int() {
                ctx.builder.ins().iconst(ty, 0)
            } else if ty.is_bool() {
                ctx.builder.ins().bconst(ty, false)
            } else if ty == ir::types::F32 {
                ctx.builder.ins().f32const(
                    ir::immediates::Ieee32::with_bits(0),
                )
            } else if ty == ir::types::F64 {
                ctx.builder.ins().f64const(
                    ir::immediates::Ieee64::with_bits(0),
                )
            } else {
                panic!("unimplemented undef type: {}", ty);
            }
        }
        LLVMConstantIntValueKind => {
            let ty = translate_type_of(llvm_val, ctx.dl);
            let raw = unsafe { LLVMConstIntGetSExtValue(llvm_val) };
            if ty.is_int() {
                ctx.builder.ins().iconst(ty, raw)
            } else if ty.is_bool() {
                ctx.builder.ins().bconst(ty, raw != 0)
            } else {
                panic!("unexpected ConstantInt type");
            }
        }
        LLVMConstantPointerNullValueKind => {
            let ty = translate_pointer_type(ctx.dl);
            ctx.builder.ins().iconst(ty, 0)
        }
        LLVMFunctionValueKind => {
            let signature = ctx.builder.import_signature(translate_sig(
                unsafe { LLVMGetElementType(LLVMTypeOf(llvm_val)) },
                ctx.dl,
            ));
            let name = translate_symbol_name(unsafe { LLVMGetValueName(llvm_val) })
                .expect("unimplemented: unusual symbol names");
            let callee = ctx.builder.import_function(
                ir::ExtFuncData { name, signature },
            );
            let ty = translate_pointer_type(ctx.dl);
            ctx.builder.ins().func_addr(ty, callee)
        }
        LLVMGlobalAliasValueKind |
        LLVMGlobalVariableValueKind => {
            let name = translate_symbol_name(unsafe { LLVMGetValueName(llvm_val) })
                .expect("unimplemented: unusual symbol names");
            let global = ctx.builder.create_global_var(
                ir::GlobalVarData::Sym { name },
            );
            let ty = translate_pointer_type(ctx.dl);
            ctx.builder.ins().global_addr(ty, global)
        }
        LLVMConstantFPValueKind => {
            let mut loses_info = [0];
            let val = unsafe { LLVMConstRealGetDouble(llvm_val, loses_info.as_mut_ptr()) };
            debug_assert_eq!(
                loses_info[0],
                0,
                "unimplemented floating-point constant value"
            );
            match translate_type_of(llvm_val, ctx.dl) {
                ir::types::F32 => {
                    let f32val = val as f32;
                    ctx.builder.ins().f32const(
                        ir::immediates::Ieee32::with_bits(
                            unsafe { mem::transmute(f32val) },
                        ),
                    )
                }
                ir::types::F64 => {
                    ctx.builder.ins().f64const(
                        ir::immediates::Ieee64::with_bits(
                            unsafe { mem::transmute(val) },
                        ),
                    )
                }
                ty => panic!("unimplemented floating-point constant type: {}", ty),
            }
        }
        LLVMConstantExprValueKind => {
            let llvm_opcode = unsafe { LLVMGetConstOpcode(llvm_val) };
            translate_operation(ptr::null_mut(), llvm_opcode, llvm_val, ctx)
                .expect("Constants don't return void")
        }
        _ => panic!("unimplemented constant kind: {:?}", llvm_kind),
    }
}

/// Translate an LLVM IR intrinsic call to Cretonne.
fn translate_intrinsic(
    llvm_inst: LLVMValueRef,
    llvm_callee: LLVMValueRef,
    ctx: &mut Context,
) -> Option<ir::Value> {
    // LLVM's C API doesn't expose intrinsic IDs yet, so we match by name.
    let name = translate_string(unsafe { LLVMGetValueName(llvm_callee) })
        .expect("unimplemented: unusual function names");
    Some(match name.as_ref() {
        "llvm.dbg.addr" |
        "llvm.dbg.declare" |
        "llvm.dbg.value" |
        "llvm.prefetch" |
        "llvm.assume" |
        "llvm.lifetime.start.p0i8" |
        "llvm.lifetime.end.p0i8" |
        "llvm.invariant.start.p0i8" |
        "llvm.invariant.end.p0i8" |
        "llvm.sideeffect" |
        "llvm.codeview.annotation" => {
            // For now, just discard this informtion.
            return None;
        }
        "llvm.ssa_copy.i1" |
        "llvm.ssa_copy.i8" |
        "llvm.ssa_copy.i16" |
        "llvm.ssa_copy.i32" |
        "llvm.ssa_copy.i64" |
        "llvm.ssa_copy.f32" |
        "llvm.ssa_copy.f64" |
        "llvm.annotation.i8" |
        "llvm.annotation.i16" |
        "llvm.annotation.i32" |
        "llvm.annotation.i64" |
        "llvm.invariant.group.barrier" |
        "llvm.expect.i1" |
        "llvm.expect.i8" |
        "llvm.expect.i16" |
        "llvm.expect.i32" |
        "llvm.expect.i64" => {
            // For now, just discard the extra informtion these intrinsics
            // provide and just return their first operand.
            unary_operands(llvm_inst, ctx)
        }
        "llvm.objectsize.i8.p0i8" |
        "llvm.objectsize.i16.p0i8" |
        "llvm.objectsize.i32.p0i8" |
        "llvm.objectsize.i64.p0i8" => {
            let min = unsafe { LLVMConstIntGetZExtValue(LLVMGetOperand(llvm_inst, 1)) } != 0;
            let ty = translate_type_of(llvm_inst, ctx.dl);
            ctx.builder.ins().iconst(
                ty,
                ir::immediates::Imm64::new(
                    if min { 0 } else { -1 },
                ),
            )
        }
        "llvm.sqrt.f32" | "llvm.sqrt.f64" => {
            let op = unary_operands(llvm_inst, ctx);
            ctx.builder.ins().sqrt(op)
        }
        "llvm.fabs.f32" | "llvm.fabs.f64" => {
            let op = unary_operands(llvm_inst, ctx);
            ctx.builder.ins().fabs(op)
        }
        "llvm.copysign.f32" |
        "llvm.copysign.f64" => {
            let (lhs, rhs) = binary_operands(llvm_inst, ctx);
            ctx.builder.ins().fcopysign(lhs, rhs)
        }
        "llvm.ceil.f32" | "llvm.ceil.f64" => {
            let op = unary_operands(llvm_inst, ctx);
            ctx.builder.ins().ceil(op)
        }
        "llvm.floor.f32" | "llvm.floor.f64" => {
            let op = unary_operands(llvm_inst, ctx);
            ctx.builder.ins().floor(op)
        }
        "llvm.trunc.f32" | "llvm.trunc.f64" => {
            let op = unary_operands(llvm_inst, ctx);
            ctx.builder.ins().trunc(op)
        }
        "llvm.nearbyint.f32" |
        "llvm.nearbyint.f64" => {
            let op = unary_operands(llvm_inst, ctx);
            ctx.builder.ins().nearest(op)
        }
        "llvm.ctpop.i8" | "llvm.ctpop.i16" | "llvm.ctpop.i32" | "llvm.ctpop.i64" => {
            let op = unary_operands(llvm_inst, ctx);
            ctx.builder.ins().popcnt(op)
        }
        "llvm.ctlz.i8" | "llvm.ctlz.i16" | "llvm.ctlz.i32" | "llvm.ctlz.i64" => {
            let op = unary_operands(llvm_inst, ctx);
            ctx.builder.ins().clz(op)
        }
        "llvm.cttz.i8" | "llvm.cttz.i16" | "llvm.cttz.i32" | "llvm.cttz.i64" => {
            let op = unary_operands(llvm_inst, ctx);
            ctx.builder.ins().ctz(op)
        }
        "llvm.trap" => {
            // This intrinsic isn't a terminator in LLVM IR, but trap is a
            // terminator in Cretonne. After the trap, start a new basic block.
            ctx.builder.ins().trap(ir::TrapCode::User(1));
            let ebb = ctx.builder.create_ebb();
            ctx.builder.seal_block(ebb);
            ctx.builder.switch_to_block(ebb);
            return None;
        }
        "llvm.debugtrap" => {
            // See the comment on "llvm.trap".
            ctx.builder.ins().trap(ir::TrapCode::User(2));
            let ebb = ctx.builder.create_ebb();
            ctx.builder.seal_block(ebb);
            ctx.builder.switch_to_block(ebb);
            return None;
        }
        "llvm.fmuladd.f32" |
        "llvm.fmuladd.f64" => {
            // Cretonne currently has no fma instruction, so just lower these
            // as non-fused operations.
            let (a, b, c) = ternary_operands(llvm_inst, ctx);
            let t = ctx.builder.ins().fmul(a, b);
            ctx.builder.ins().fadd(t, c)
        }
        "llvm.minnum.f32" => translate_intr_libcall("fminf", llvm_inst, llvm_callee, ctx),
        "llvm.minnum.f64" => translate_intr_libcall("fmin", llvm_inst, llvm_callee, ctx),
        "llvm.maxnum.f32" => translate_intr_libcall("fmaxf", llvm_inst, llvm_callee, ctx),
        "llvm.maxnum.f64" => translate_intr_libcall("fmax", llvm_inst, llvm_callee, ctx),
        "llvm.sin.f32" => translate_intr_libcall("sinf", llvm_inst, llvm_callee, ctx),
        "llvm.sin.f64" => translate_intr_libcall("sin", llvm_inst, llvm_callee, ctx),
        "llvm.cos.f32" => translate_intr_libcall("cosf", llvm_inst, llvm_callee, ctx),
        "llvm.cos.f64" => translate_intr_libcall("cos", llvm_inst, llvm_callee, ctx),
        "llvm.tan.f32" => translate_intr_libcall("tanf", llvm_inst, llvm_callee, ctx),
        "llvm.tan.f64" => translate_intr_libcall("tan", llvm_inst, llvm_callee, ctx),
        "llvm.exp.f32" => translate_intr_libcall("expf", llvm_inst, llvm_callee, ctx),
        "llvm.exp.f64" => translate_intr_libcall("exp", llvm_inst, llvm_callee, ctx),
        "llvm.exp2.f32" => translate_intr_libcall("exp2f", llvm_inst, llvm_callee, ctx),
        "llvm.exp2.f64" => translate_intr_libcall("exp2", llvm_inst, llvm_callee, ctx),
        "llvm.log.f32" => translate_intr_libcall("logf", llvm_inst, llvm_callee, ctx),
        "llvm.log.f64" => translate_intr_libcall("log", llvm_inst, llvm_callee, ctx),
        "llvm.log2.f32" => translate_intr_libcall("log2f", llvm_inst, llvm_callee, ctx),
        "llvm.log2.f64" => translate_intr_libcall("log2", llvm_inst, llvm_callee, ctx),
        "llvm.log10.f32" => translate_intr_libcall("log10f", llvm_inst, llvm_callee, ctx),
        "llvm.log10.f64" => translate_intr_libcall("log10", llvm_inst, llvm_callee, ctx),
        "llvm.pow.f32" => translate_intr_libcall("powf", llvm_inst, llvm_callee, ctx),
        "llvm.pow.f64" => translate_intr_libcall("pow", llvm_inst, llvm_callee, ctx),
        "llvm.rint.f32" => translate_intr_libcall("rintf", llvm_inst, llvm_callee, ctx),
        "llvm.rint.f64" => translate_intr_libcall("rint", llvm_inst, llvm_callee, ctx),
        "llvm.round.f32" => translate_intr_libcall("roundf", llvm_inst, llvm_callee, ctx),
        "llvm.round.f64" => translate_intr_libcall("round", llvm_inst, llvm_callee, ctx),
        "llvm.fma.f32" => translate_intr_libcall("fmaf", llvm_inst, llvm_callee, ctx),
        "llvm.fma.f64" => translate_intr_libcall("fma", llvm_inst, llvm_callee, ctx),
        "llvm.memcpy.p0i8.p0i8.i8" |
        "llvm.memcpy.p0i8.p0i8.i16" |
        "llvm.memcpy.p0i8.p0i8.i32" |
        "llvm.memcpy.p0i8.p0i8.i64" => {
            translate_mem_intrinsic("memcpy", llvm_inst, ctx);
            return None;
        }
        "llvm.memmove.p0i8.p0i8.i8" |
        "llvm.memmove.p0i8.p0i8.i16" |
        "llvm.memmove.p0i8.p0i8.i32" |
        "llvm.memmove.p0i8.p0i8.i64" => {
            translate_mem_intrinsic("memmove", llvm_inst, ctx);
            return None;
        }
        "llvm.memset.p0i8.i8" |
        "llvm.memset.p0i8.i16" |
        "llvm.memset.p0i8.i32" |
        "llvm.memset.p0i8.i64" => {
            translate_mem_intrinsic("memset", llvm_inst, ctx);
            return None;
        }
        _ => panic!("unimplemented: intrinsic: {}", name),
    })
}

/// Translate an LLVM IR intrinsic call which corresponds directly to a
/// C library call.
fn translate_intr_libcall(
    name: &str,
    llvm_inst: LLVMValueRef,
    llvm_callee: LLVMValueRef,
    ctx: &mut Context,
) -> ir::Value {
    let num_args = unsafe { LLVMGetNumArgOperands(llvm_inst) } as usize;
    let mut args = Vec::with_capacity(num_args);
    for i in 0..num_args {
        debug_assert_eq!(i as libc::c_uint as usize, i);
        args.push(use_val(
            unsafe { LLVMGetOperand(llvm_inst, i as libc::c_uint) },
            ctx,
        ));
    }

    let name = ir::ExternalName::new(name);
    let signature = ctx.builder.import_signature(translate_sig(
        unsafe { LLVMGetElementType(LLVMTypeOf(llvm_callee)) },
        ctx.dl,
    ));
    let data = ir::ExtFuncData { name, signature };
    let callee = ctx.builder.import_function(data);
    let call = ctx.builder.ins().call(callee, &args);
    let results = ctx.builder.inst_results(call);
    debug_assert_eq!(results.len(), 1);
    results[0]
}

/// Translate an LLVM IR llvm.memcpy/memmove/memset call.
fn translate_mem_intrinsic(name: &str, llvm_inst: LLVMValueRef, ctx: &mut Context) {
    let pointer_type = translate_pointer_type(ctx.dl);

    let dst_arg = use_val(unsafe { LLVMGetOperand(llvm_inst, 0) }, ctx);

    let llvm_src_arg = unsafe { LLVMGetOperand(llvm_inst, 1) };
    let orig_src_arg = use_val(llvm_src_arg, ctx);
    let src_arg = if name == "memset" {
        ctx.builder.ins().uextend(ir::types::I32, orig_src_arg)
    } else {
        orig_src_arg
    };

    let llvm_len_arg = unsafe { LLVMGetOperand(llvm_inst, 2) };
    let len_arg = unsigned_cast(
        translate_type_of(llvm_len_arg, ctx.dl),
        pointer_type,
        use_val(llvm_len_arg, ctx),
        ctx,
    );

    // Discard the alignment hint for now.
    // Discard the volatile flag too, which is safe as long as Cretonne doesn't
    // start optimizing these libcalls.
    let args = [dst_arg, src_arg, len_arg];

    let funcname = ir::ExternalName::new(name);
    let mut sig = ir::Signature::new(ir::CallConv::Native);
    sig.params.resize(3, ir::AbiParam::new(pointer_type));
    if name == "memset" {
        sig.params[1] = ir::AbiParam::new(ir::types::I32);
    }
    sig.returns.resize(1, ir::AbiParam::new(pointer_type));
    let signature = ctx.builder.import_signature(sig);
    let data = ir::ExtFuncData {
        name: funcname,
        signature,
    };
    let callee = ctx.builder.import_function(data);
    ctx.builder.ins().call(callee, &args);
}

/// Record PHI uses and defs for a branch from `llvm_bb` to `llvm_succ`.
fn handle_phi_operands(
    llvm_bb: LLVMBasicBlockRef,
    llvm_succ: LLVMBasicBlockRef,
    ctx: &mut Context,
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

            let val = use_val(llvm_val, ctx);
            defs.push((llvm_inst, val));
        } else {
            break;
        }
        llvm_inst = unsafe { LLVMGetNextInstruction(llvm_inst) };
    }

    for (llvm_inst, val) in defs {
        def_val(llvm_inst, val, ctx);
    }
}

/// Translate a GetElementPtr index operand into Cretonne IL.
fn translate_gep_index(
    llvm_gepty: LLVMTypeRef,
    ptr: ir::Value,
    imm: i64,
    pointer_type: ir::Type,
    index: LLVMValueRef,
    ctx: &mut Context,
) -> (LLVMTypeRef, ir::Value, i64) {
    // TODO: We'd really want gep_type_iterator etc. here, but
    // LLVM's C API doesn't expose those currently.
    let (offset, ty) = match unsafe { LLVMGetTypeKind(llvm_gepty) } {
        LLVMStructTypeKind => {
            let i = unsafe { LLVMConstIntGetZExtValue(index) };
            debug_assert_eq!(u64::from(i as libc::c_uint), i);
            let llvm_eltty = unsafe { LLVMStructGetTypeAtIndex(llvm_gepty, i as libc::c_uint) };
            let imm_offset =
                unsafe { LLVMOffsetOfElement(ctx.dl, llvm_gepty, i as libc::c_uint) } as i64;
            return (llvm_eltty, ptr, imm.wrapping_add(imm_offset));
        }
        LLVMPointerTypeKind | LLVMArrayTypeKind | LLVMVectorTypeKind => {
            let llvm_eltty = unsafe { LLVMGetElementType(llvm_gepty) };
            let size = unsafe { LLVMABISizeOfType(ctx.dl, llvm_eltty) };

            if unsafe { LLVMIsConstant(index) } != 0 {
                let index_val = unsafe { LLVMConstIntGetSExtValue(index) };
                let imm_offset = index_val.wrapping_mul(size as i64);
                return (llvm_eltty, ptr, imm.wrapping_add(imm_offset));
            }

            let index_type = translate_type_of(index, ctx.dl);
            let mut x = use_val(index, ctx);
            if index_type != pointer_type {
                if index_type.is_bool() {
                    x = ctx.builder.ins().bmask(pointer_type, x);
                } else if index_type.bits() < pointer_type.bits() {
                    x = ctx.builder.ins().sextend(pointer_type, x);
                } else {
                    x = ctx.builder.ins().ireduce(pointer_type, x);
                }
            }

            if size != 1 {
                x = ctx.builder.ins().imul_imm(
                    x,
                    ir::immediates::Imm64::new(size as i64),
                );
            }

            (x, llvm_eltty)
        }
        _ => panic!("unexpected GEP indexing type: {:?}", llvm_gepty),
    };
    (ty, ctx.builder.ins().iadd(ptr, offset), imm)
}

/// Emit a Cretonne jump to the destination corresponding to `llvm_succ`, if
/// one is needed.
fn jump(llvm_bb: LLVMBasicBlockRef, llvm_succ: LLVMBasicBlockRef, ctx: &mut Context) {
    if let Some(&ebb) = ctx.ebb_map.get(&llvm_succ) {
        handle_phi_operands(llvm_bb, llvm_succ, ctx);
        ctx.builder.ins().jump(ebb, &[]);
    }
}

fn translate_bool_cmp(
    condcode: ir::condcodes::IntCC,
    lhs: ir::Value,
    rhs: ir::Value,
    ctx: &mut Context,
) -> ir::Value {
    match condcode {
        ir::condcodes::IntCC::Equal => ctx.builder.ins().bxor_not(lhs, rhs),
        ir::condcodes::IntCC::NotEqual => ctx.builder.ins().bxor(lhs, rhs),
        ir::condcodes::IntCC::SignedLessThan => ctx.builder.ins().band_not(lhs, rhs),
        ir::condcodes::IntCC::SignedLessThanOrEqual => ctx.builder.ins().bor_not(lhs, rhs),
        ir::condcodes::IntCC::SignedGreaterThan => ctx.builder.ins().band_not(rhs, lhs),
        ir::condcodes::IntCC::SignedGreaterThanOrEqual => ctx.builder.ins().bor_not(rhs, lhs),
        ir::condcodes::IntCC::UnsignedLessThan => ctx.builder.ins().band_not(rhs, lhs),
        ir::condcodes::IntCC::UnsignedLessThanOrEqual => ctx.builder.ins().bor_not(rhs, lhs),
        ir::condcodes::IntCC::UnsignedGreaterThan => ctx.builder.ins().band_not(lhs, rhs),
        ir::condcodes::IntCC::UnsignedGreaterThanOrEqual => ctx.builder.ins().bor_not(lhs, rhs),
    }
}

fn unsigned_cast(
    from: ir::types::Type,
    to: ir::types::Type,
    op: ir::Value,
    ctx: &mut Context,
) -> ir::Value {
    if from == to {
        // No-op cast.
        op
    } else if from.bits() > to.bits() {
        if to.is_bool() {
            let band = ctx.builder.ins().band_imm(op, 1);
            ctx.builder.ins().icmp_imm(
                ir::condcodes::IntCC::NotEqual,
                band,
                0,
            )
        } else {
            ctx.builder.ins().ireduce(to, op)
        }
    } else {
        if from.is_bool() {
            ctx.builder.ins().bint(to, op)
        } else {
            ctx.builder.ins().uextend(to, op)
        }
    }
}

/// Record a "use" of an LLVM IR value.
fn use_val(llvm_val: LLVMValueRef, ctx: &mut Context) -> ir::Value {
    if unsafe { LLVMIsConstant(llvm_val) } != 0 {
        materialize_constant(llvm_val, ctx)
    } else {
        let num_values = ctx.value_map.len();
        let var = match ctx.value_map.entry(llvm_val) {
            hash_map::Entry::Occupied(entry) => *entry.get(),
            hash_map::Entry::Vacant(entry) => {
                let var = Variable::new(num_values);
                let ty = translate_type_of(llvm_val, ctx.dl);
                ctx.builder.declare_var(var, ty);
                *entry.insert(var)
            }
        };
        ctx.builder.use_var(var)
    }
}

/// Record a "definition" of an LLVM IR value.
fn def_val(llvm_val: LLVMValueRef, value: ir::Value, ctx: &mut Context) {
    if unsafe { LLVMIsConstant(llvm_val) } != 0 {
        // Do nothing. In use_val, we special-case constants and materialize
        // them for each use.
    } else {
        let num_values = ctx.value_map.len();
        let var = match ctx.value_map.entry(llvm_val) {
            hash_map::Entry::Occupied(entry) => *entry.get(),
            hash_map::Entry::Vacant(entry) => {
                let var = Variable::new(num_values);
                let ty = translate_type_of(llvm_val, ctx.dl);
                ctx.builder.declare_var(var, ty);
                *entry.insert(var)
            }
        };
        ctx.builder.def_var(var, value)
    }
}

/// Translate the operands for a unary operation.
fn unary_operands(llvm_val: LLVMValueRef, ctx: &mut Context) -> ir::Value {
    use_val(unsafe { LLVMGetOperand(llvm_val, 0) }, ctx)
}

/// A boolean operation with two register inputs, an integer operation
/// with two register operands, or an integer operation with a register
/// operand and an immediate operand.
enum RegImmOperands {
    Bool(ir::Value, ir::Value),
    RegReg(ir::Value, ir::Value),
    RegImm(ir::Value, ir::immediates::Imm64),
}

/// Translate the operands for a binary operation taking one register and
/// one operand which may be either a register or an immediate.
///
/// Using the `*_imm` forms of instructions isn't necessary, as Cretonne
/// should fold constants into immediates just as well, but doing it in
/// translation makes the output tidier and exercises more of the
/// builder API.
fn binary_operands_r_ri(llvm_val: LLVMValueRef, ctx: &mut Context) -> RegImmOperands {
    // For most instructions, we don't need to check whether the lhs is
    // a constant, because constants are canonicalized the the rhs when
    // possible.
    let lhs = use_val(unsafe { LLVMGetOperand(llvm_val, 0) }, ctx);

    // Optimize the rhs if it's a constant and not boolean, since Cretonne's
    // `*_imm` instructions don't support boolean types.
    let llvm_rhs = unsafe { LLVMGetOperand(llvm_val, 1) };
    let llvm_rhs_type = unsafe { LLVMTypeOf(llvm_rhs) };
    if unsafe { LLVMGetIntTypeWidth(llvm_rhs_type) } == 1 {
        RegImmOperands::Bool(lhs, use_val(llvm_rhs, ctx))
    } else if unsafe { LLVMIsConstant(llvm_rhs) } != 0 {
        RegImmOperands::RegImm(
            lhs,
            ir::immediates::Imm64::from(unsafe { LLVMConstIntGetZExtValue(llvm_rhs) } as i64),
        )
    } else {
        RegImmOperands::RegReg(lhs, use_val(llvm_rhs, ctx))
    }
}

/// Translate the operands for a binary operation.
fn binary_operands(llvm_val: LLVMValueRef, ctx: &mut Context) -> (ir::Value, ir::Value) {
    (
        use_val(unsafe { LLVMGetOperand(llvm_val, 0) }, ctx),
        use_val(unsafe { LLVMGetOperand(llvm_val, 1) }, ctx),
    )
}

/// Translate the operands for a binary operation.
fn ternary_operands(
    llvm_val: LLVMValueRef,
    ctx: &mut Context,
) -> (ir::Value, ir::Value, ir::Value) {
    (
        use_val(unsafe { LLVMGetOperand(llvm_val, 0) }, ctx),
        use_val(unsafe { LLVMGetOperand(llvm_val, 1) }, ctx),
        use_val(unsafe { LLVMGetOperand(llvm_val, 2) }, ctx),
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
    ctx: &Context,
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
        unsafe { LLVMABISizeOfType(ctx.dl, llvm_ty) }
    {
        flags.set_aligned();
    }

    flags
}

/// Translate the type of an LLVM Value into a Cretonne type.
fn translate_type_of(llvm_val: LLVMValueRef, dl: LLVMTargetDataRef) -> ir::Type {
    translate_type(unsafe { LLVMTypeOf(llvm_val) }, dl)
}
