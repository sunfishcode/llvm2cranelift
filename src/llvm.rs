//! CLI tool to use the functions provided by the [cretonne-llvm](../cton_llvm/index.html) crate.
//!
//! Reads LLVM IR files, translates the functions' code to Cretonne IL.

use cton_llvm::{create_llvm_context, read_llvm, translate_module};
use std::path::PathBuf;
use cretonne::isa::TargetIsa;
use std::path::Path;
use std::str;
use std::error::Error;
use std::fmt::format;
use term;
use utils::{parse_sets_and_isa, OwnedFlagsOrIsa};
use faerie::{Artifact, Elf, Target};

macro_rules! vprintln {
    ($x: expr, $($tts:tt)*) => {
        if $x {
            println!($($tts)*);
        }
    }
}

macro_rules! vprint {
    ($x: expr, $($tts:tt)*) => {
        if $x {
            print!($($tts)*);
        }
    }
}

pub fn run(
    files: Vec<String>,
    arg_output: &str,
    flag_verbose: bool,
    flag_print: bool,
    flag_set: Vec<String>,
    flag_isa: String,
) -> Result<(), String> {
    let parsed = parse_sets_and_isa(flag_set, flag_isa)?;

    for filename in files {
        let path = Path::new(&filename);
        let name = String::from(path.as_os_str().to_string_lossy());
        handle_module(
            arg_output,
            flag_verbose,
            flag_print,
            path.to_path_buf(),
            name,
            &parsed,
        )?;
    }
    Ok(())
}

fn handle_module(
    arg_output: &str,
    flag_verbose: bool,
    flag_print: bool,
    path: PathBuf,
    name: String,
    parsed: &OwnedFlagsOrIsa,
) -> Result<(), String> {
    let mut terminal = term::stdout().unwrap();
    terminal.fg(term::color::YELLOW).unwrap();
    vprint!(flag_verbose, "Handling: ");
    terminal.reset().unwrap();
    vprintln!(flag_verbose, "\"{}\"", name);
    terminal.fg(term::color::MAGENTA).unwrap();
    vprint!(flag_verbose, "Translating... ");
    terminal.reset().unwrap();

    // If we have an isa from the command-line, use that. Otherwise if the
    // file contins a unique isa, use that.
    let isa = parsed.as_fisa().isa;

    let ctx = create_llvm_context();
    let llvm_module = read_llvm(ctx, path.to_str().ok_or_else(|| "invalid utf8 in path")?)?;
    let module = translate_module(llvm_module, isa)?;

    if flag_print {
        vprintln!(flag_verbose, "");
        for func in &module.functions {
            println!("function: {}", func.display(isa));
            vprintln!(flag_verbose, "");
        }
        for sym in &module.data_symbols {
            println!("data symbol: {}", sym);
        }
    } else {
        let isa: &TargetIsa = isa.expect("compilation requires a target isa");

        let mut obj = Artifact::new(faerie_target(isa)?, Some(String::from(arg_output)));

        for import in &module.imports {
            let name = str::from_utf8(import.as_ref()).map_err(|err| {
                err.description().to_string()
            })?;
            obj.import(name);
        }
        for func in module.functions {
            let func_name = str::from_utf8(func.il.name.as_ref()).map_err(|err| {
                err.description().to_string()
            })?;
            let compilation = func.compilation.unwrap();
            obj.add_code(func_name, compilation.body);
            for &(ref _reloc, ref external_name, ref offset) in &compilation.relocs.relocs {
                // FIXME: What about other types of relocs?
                // TODO: Faerie API: Seems like it might be inefficient to
                // identify the caller by name each time.
                // TODO: Faerie API: It's inconvenient to keep track of which
                // symbols are imports and which aren't.
                let name = str::from_utf8(external_name.as_ref()).map_err(|err| {
                    err.description().to_string()
                })?;
                if module.unique_imports.contains(&name.to_string()) {
                    obj.link_import(func_name, name, *offset as usize);
                } else {
                    obj.link(func_name, name, *offset as usize);
                }
            }
        }
        for data in &module.data_symbols {
            let data_name = str::from_utf8(data.name.as_ref()).map_err(|err| {
                err.description().to_string()
            })?;
            obj.add_data(data_name, data.contents.clone());
        }

        let file = ::std::fs::File::create(Path::new(arg_output)).map_err(
            |x| {
                format(format_args!("{}", x))
            },
        )?;

        // FIXME: Make the format a parameter.
        obj.write::<Elf>(file).map_err(
            |x| format(format_args!("{}", x)),
        )?;
    }

    terminal.fg(term::color::GREEN).unwrap();
    vprintln!(flag_verbose, "ok");
    terminal.reset().unwrap();
    Ok(())
}

fn faerie_target(isa: &TargetIsa) -> Result<Target, String> {
    let name = isa.name();
    match name {
        "intel" => Ok(if isa.flags().is_64bit() {
            Target::X86_64
        } else {
            Target::X86
        }),
        "arm32" => Ok(Target::ARMv7),
        "arm64" => Ok(Target::ARM64),
        _ => Err(format!("unsupported isa: {}", name)),
    }
}
