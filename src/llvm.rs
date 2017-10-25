//! CLI tool to use the functions provided by the [cretonne-llvm](../cton_llvm/index.html) crate.
//!
//! Reads LLVM IR files, translates the functions' code to Cretonne IL.

use cton_llvm::{create_llvm_context, read_llvm, translate_module};
use std::path::PathBuf;
use cretonne::settings::FlagsOrIsa;
use std::path::Path;
use term;
use utils::parse_sets_and_isa;

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
    flag_verbose: bool,
    flag_just_decode: bool,
    flag_check_translation: bool,
    flag_print: bool,
    flag_set: Vec<String>,
    flag_isa: String,
) -> Result<(), String> {
    let parsed = parse_sets_and_isa(flag_set, flag_isa)?;

    for filename in files {
        let path = Path::new(&filename);
        let name = String::from(path.as_os_str().to_string_lossy());
        handle_module(
            flag_verbose,
            flag_just_decode,
            flag_check_translation,
            flag_print,
            path.to_path_buf(),
            name,
            parsed.as_fisa(),
        )?;
    }
    Ok(())
}

fn handle_module(
    flag_verbose: bool,
    _flag_just_decode: bool,
    _flag_check_translation: bool,
    _flag_print: bool,
    path: PathBuf,
    name: String,
    _fisa: FlagsOrIsa,
) -> Result<(), String> {
    let mut terminal = term::stdout().unwrap();
    terminal.fg(term::color::YELLOW).unwrap();
    vprint!(flag_verbose, "Handling: ");
    terminal.reset().unwrap();
    vprintln!(flag_verbose, "\"{}\"", name);
    terminal.fg(term::color::MAGENTA).unwrap();
    vprint!(flag_verbose, "Translating... ");
    terminal.reset().unwrap();
    let ctx = create_llvm_context();
    let module = read_llvm(ctx, path.to_str().ok_or_else(|| "invalid utf8 in path")?)?;
    for func in translate_module(module)? {
        // For now, just print the result.
        println!("{}", func.display(None));
    }
    terminal.fg(term::color::GREEN).unwrap();
    vprintln!(flag_verbose, "ok");
    terminal.reset().unwrap();
    Ok(())
}
