use cranelift_codegen::ir;
use fnv::FnvBuildHasher;
use indexmap::IndexMap;

type FnvIndexSet<T> = IndexMap<T, (), FnvBuildHasher>;

pub struct StringTable {
    names: [FnvIndexSet<String>; 2],
}

impl StringTable {
    pub fn new() -> Self {
        Self {
            names: [FnvIndexSet::default(), FnvIndexSet::default()],
        }
    }

    pub fn get_ns(&self, is_func: bool) -> u32 {
        if is_func { 0 } else { 1 }
    }

    /// Return the string name for a given cranelift `ExternalName`.
    pub fn get_str(&self, extname: &ir::ExternalName) -> &str {
        match *extname {
            ir::ExternalName::User { namespace, index } => {
                debug_assert!(namespace == 0 || namespace == 1, "alternate namespaces not yet implemented");
                self.names[namespace as usize]
                    .get_index(index as usize)
                    .expect("name has not yet been declared")
                    .0
                    .as_str()
            }
            _ => panic!("non-user names not yet implemented"),
        }
    }

    /// Enter a string name into the table.
    pub fn declare_extname<S: Into<String>>(&mut self, string: S, is_func: bool) {
        let previous = self.names[self.get_ns(is_func) as usize].insert(string.into(), ());
        debug_assert!(previous.is_none());
    }

    /// Return the cranelift `ExternalName` for a given string name.
    pub fn get_extname<S: Into<String>>(&self, string: S, is_func: bool) -> ir::ExternalName {
        let namespace = self.get_ns(is_func);
        let index = self.names[namespace as usize].get_full(&string.into()).unwrap().0;
        debug_assert!(index as u32 as usize == index);
        ir::ExternalName::user(namespace, index as u32)
    }
}
