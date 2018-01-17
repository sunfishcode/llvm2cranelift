# llvm2cretonne
LLVM IR to Cretonne translator

This is a work in progress which currently is complete enough to be usable as a
testing and demonstration tool. It supports all the basic integer, floating-point
control flow, call, and memory operations, and more.

Here's a quick example of it in action:

```
$ cat test.ll
define void @foo(i32 %arg, i32 %arg1) {
bb:
  %tmp = icmp ult i32 %arg, 4
  %tmp2 = icmp eq i32 %arg1, 0
  %tmp3 = or i1 %tmp2, %tmp
  br i1 %tmp3, label %bb4, label %bb6

bb4:
  %tmp5 = call i32 @bar()
  ret void

bb6:
  ret void
}

declare i32 @bar()

$ llvm2cretonne -p test.ll
; foo
function u0:0(i32, i32) native {
    sig0 = () -> i32 native
    fn0 = sig0 u0:1

ebb1(v0: i32, v1: i32):
    v2 = icmp_imm ult v0, 4
    v3 = icmp_imm eq v1, 0
    v4 = bor v3, v2
    brz v4, ebb0
    v5 = call fn0()
    return

ebb0:
    return
}
```

Features not yet implemented, but which could be reasonably implemented include:
 - switch instructions with non-sequential cases
 - first-class aggregates
 - SIMD types and operations
 - most function argument attributes
 - integer types with unusual sizes
 - indirectbr

Features not yet implemented that would require features that Cretonne does not
yet fully implement include:
 - exception handling (invoke, landingpad, etc.)
 - dynamic alloca
 - atomic operations
 - thread-local globals
 - volatile operations
 - add-with-overflow intrinsics
 - inline asm
 - GC hooks
 - varargs
 - ifuncs
 - comdat groups

Features not yet implemented that LLVM's C API does not yet sufficiently expose:
 - global aliases
 - debug source locations for instructions
 - debug info

Optimizations that are commonly done for LLVM IR during codegen that aren't yet
implemented include:
 - Optimize @llvm.memcpy et al for small and/or aligned cases.
 - Optimize switch for small, sparse, and/or other special cases.
 - Pattern-match operations that are not present in LLVM IR, such as
   rotates, load/store offsets, wrapping/extending loads and stores,
   `br_icmp`, etc.
 - Expand @llvm.powi with small integer constants into efficient sequences.
 - Handling of i1 is suboptimal in cases like zext(load(p)) because the result
   of the load is converted to boolean and then converted back.

Also of note is that it doesn't currently translate LLVM's PHIs, SSA uses, and
SSA defs directly into Cretonne; it instead just reports uses and defs to
Cretonne's SSA builder, which then builds its own SSA form. That simplifies
handling of basic blocks that aren't in RPO -- in layout order, we may see uses
before we see their defs.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
