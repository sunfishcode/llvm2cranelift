use cretonne::ir;
use ordermap::OrderMap;

pub struct StringTable {
    names: OrderMap<String, ()>,
}

impl StringTable {
    pub fn new() -> Self {
        Self { names: OrderMap::new() }
    }

    // TODO: Can we avoid returning a clone of the string?
    pub fn get_str(&mut self, extname: ir::ExternalName) -> String {
        match extname {
            ir::ExternalName::User { namespace, index } => {
                debug_assert!(namespace == 0, "alternate namespaces not yet implemented");
                self.names
                    .get_index(index as usize)
                    .expect("name has not yet been declared")
                    .0
            }
            _ => panic!("non-user names not yet implemented"),
        }.clone()
    }

    // TODO: Can we minimize how often our users have to clone strings?
    pub fn get_extname<S: Into<String>>(&mut self, string: S) -> ir::ExternalName {
        let entry = self.names.entry(string.into());
        let index = entry.index();
        entry.or_insert(());
        debug_assert!(index as u32 as usize == index);
        ir::ExternalName::user(0, index as u32)
    }
}
