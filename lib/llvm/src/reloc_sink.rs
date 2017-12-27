use cretonne::ir;
use cretonne::binemit;

pub struct RelocSink {
    pub relocs: Vec<(binemit::Reloc, ir::ExternalName, binemit::CodeOffset, binemit::Addend)>,
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
        addend: binemit::Addend,
    ) {
        // TODO: How should addend be handled? Should it be added to
        // offset and stored in self.relocs or carried through beside
        // offset?
        self.relocs.push((reloc, name.clone(), offset, addend));
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
