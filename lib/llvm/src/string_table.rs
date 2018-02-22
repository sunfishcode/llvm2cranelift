use cretonne::ir;
use fnv::FnvBuildHasher;
use indexmap::IndexMap;

type FnvIndexSet<T> = IndexMap<T, (), FnvBuildHasher>;

pub struct StringTable {
    names: FnvIndexSet<String>,
}

impl StringTable {
    pub fn new() -> Self {
        Self { names: FnvIndexSet::default() }
    }

    /// Return the string name for a given cretonne `ExternalName`.
    pub fn get_str(&self, extname: &ir::ExternalName) -> &str {
        match *extname {
            ir::ExternalName::User { namespace, index } => {
                debug_assert!(namespace == 0, "alternate namespaces not yet implemented");
                self.names
                    .get_index(index as usize)
                    .expect("name has not yet been declared")
                    .0
                    .as_str()
            }
            _ => panic!("non-user names not yet implemented"),
        }
    }

    /// Enter a string name into the table.
    pub fn declare_extname<S: Into<String>>(&mut self, string: S) {
        let previous = self.names.insert(string.into(), ());
        debug_assert!(previous.is_none());
    }

    /// Return the cretonne `ExternalName` for a given string name.
    pub fn get_extname<S: Into<String>>(&self, string: S) -> ir::ExternalName {
        let index = self.names.get_full(&string.into()).unwrap().0;
        debug_assert!(index as u32 as usize == index);
        ir::ExternalName::user(0, index as u32)
    }
}
