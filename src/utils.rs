//! Utility functions.

use cranelift_codegen::isa;
use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::settings::{self, FlagsOrIsa};
use cranelift_reader::{parse_options, Location};

/// Like `FlagsOrIsa`, but holds ownership.
pub enum OwnedFlagsOrIsa {
    Flags(settings::Flags),
    Isa(Box<TargetIsa>),
}

impl OwnedFlagsOrIsa {
    /// Produce a FlagsOrIsa reference.
    pub fn as_fisa(&self) -> FlagsOrIsa {
        match *self {
            OwnedFlagsOrIsa::Flags(ref flags) => FlagsOrIsa::from(flags),
            OwnedFlagsOrIsa::Isa(ref isa) => FlagsOrIsa::from(&**isa),
        }
    }
}

/// Parse "set" and "isa" commands.
pub fn parse_sets_and_isa(
    flag_set: Vec<String>,
    flag_isa: String,
) -> Result<OwnedFlagsOrIsa, String> {
    let mut flag_builder = settings::builder();
    parse_options(
        flag_set.iter().map(|x| x.as_str()),
        &mut flag_builder,
        &Location { line_number: 0 },
    ).map_err(|err| err.to_string())?;

    let mut words = flag_isa.trim().split_whitespace();
    // Look for `isa foo`.
    if let Some(isa_name) = words.next() {
        let mut isa_builder = isa::lookup(isa_name).map_err(|err| match err {
            isa::LookupError::Unknown => format!("unknown ISA '{}'", isa_name),
            isa::LookupError::Unsupported => format!("support for ISA '{}' not enabled", isa_name),
        })?;
        // Apply the ISA-specific settings to `isa_builder`.
        parse_options(words, &mut isa_builder, &Location { line_number: 0 })
            .map_err(|err| err.to_string())?;

        Ok(OwnedFlagsOrIsa::Isa(
            isa_builder.finish(settings::Flags::new(&flag_builder)),
        ))
    } else {
        Ok(OwnedFlagsOrIsa::Flags(settings::Flags::new(&flag_builder)))
    }
}
