use cranelift_codegen::ir;
use cranelift_codegen::isa::TargetIsa;
use std::fmt;

use reloc_sink::RelocSink;
use string_table::StringTable;

/// The kind of symbol, either data or function.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum SymbolKind {
    /// Data symbol
    Data,
    /// Function symbol
    Function,
}

pub struct DataSymbol {
    pub name: ir::ExternalName,
    pub contents: Vec<u8>,
}

impl fmt::Display for DataSymbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: ", self.name,)?;
        for byte in &self.contents {
            write!(f, "0x{:02x}, ", byte)?;
        }
        Ok(())
    }
}

pub struct Compilation {
    pub body: Vec<u8>,
    pub relocs: RelocSink,
}

pub struct CompiledFunction {
    pub il: ir::Function,
    pub compilation: Option<Compilation>,
}

impl CompiledFunction {
    pub fn display<'a>(&'a self, isa: Option<&'a TargetIsa>) -> DisplayCompiledFunction<'a> {
        DisplayCompiledFunction {
            compiled_func: &self,
            isa,
        }
    }
}

pub struct DisplayCompiledFunction<'a> {
    compiled_func: &'a CompiledFunction,
    isa: Option<&'a TargetIsa>,
}

impl<'a> fmt::Display for DisplayCompiledFunction<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.compiled_func.il.display(self.isa),)?;
        if let Some(ref compilation) = self.compiled_func.compilation {
            for byte in &compilation.body {
                write!(f, "0x{:02x}, ", byte)?;
            }
            write!(f, "\n")?;
            for &(ref reloc, ref name, ref offset, addend) in &compilation.relocs.relocs {
                match addend {
                    0 => write!(f, "reloc: {}:{}@{}\n", reloc, name, offset)?,
                    _ => write!(f, "reloc: {}:{}{}@{}\n", reloc, name, addend, offset)?,
                }
            }
        }
        Ok(())
    }
}

pub struct Module {
    pub functions: Vec<CompiledFunction>,
    pub data_symbols: Vec<DataSymbol>,
    pub imports: Vec<(ir::ExternalName, SymbolKind)>,
    pub strings: StringTable,
}

impl Module {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            data_symbols: Vec::new(),
            imports: Vec::new(),
            strings: StringTable::new(),
        }
    }
}
