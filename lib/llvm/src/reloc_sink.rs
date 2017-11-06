use cretonne::ir;
use cretonne::binemit;

pub struct RelocSink {
    pub relocs: Vec<(binemit::Reloc, ir::ExternalName, binemit::CodeOffset)>,
}

impl RelocSink {
    pub fn new() -> Self {
        Self { relocs: Vec::new() }
    }
}

impl<'func> binemit::RelocSink for RelocSink {
    fn reloc_ebb(
        &mut self,
        _offset: binemit::CodeOffset,
        _reloc: binemit::Reloc,
        _ebb_offset: binemit::CodeOffset,
    ) {
        panic!("ebb header addresses not yet implemented");
    }

    fn reloc_external(
        &mut self,
        offset: binemit::CodeOffset,
        reloc: binemit::Reloc,
        name: &ir::ExternalName,
    ) {
        self.relocs.push((reloc, name.clone(), offset));
    }

    fn reloc_jt(
        &mut self,
        _offset: binemit::CodeOffset,
        _reloc: binemit::Reloc,
        _jt: ir::JumpTable,
    ) {
        panic!("jump table addresses not yet implemented");
    }
}
