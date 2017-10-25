# llvm2cretonne
LLVM IR to Cretonne translator

This is a work in progress which currently is just complete enough
to be usable as a testing and demonstration tool.

Since it operates directly on LLVM IR, it doesn't use LLVM's legalization
framework. Some esoteric LLVM IR features, such as non-power-of-two integer
types (which do appear in clang-generated code) may be complicated to
support.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
