//! CLI tool to use the functions provided by the [cranelift-llvm](../cranelift_llvm/index.html) crate.
//!
//! Reads LLVM IR files, translates the functions' code to Cranelift IL.

use cranelift_codegen::binemit::Reloc;
use cranelift_codegen::isa::TargetIsa;
use cranelift_llvm::{create_llvm_context, read_llvm, translate_module, SymbolKind};
use faerie::{Artifact, Decl, Elf, ImportKind, Link, RelocOverride, Target};
use goblin::elf;
use std::fmt::format;
use std::path::Path;
use std::path::PathBuf;
use std::str;
use term;
use utils::{parse_sets_and_isa, OwnedFlagsOrIsa};

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
    mut flag_set: Vec<String>,
    flag_isa: String,
) -> Result<(), String> {
    // Enable the verifier by default, since we're reading IL in from a
    // text file.
    flag_set.insert(0, "enable_verifier=1".to_string());

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
            println!("; {}", module.strings.get_str(&func.il.name));
            println!("{}", func.display(isa));
            vprintln!(flag_verbose, "");
        }
        for sym in &module.data_symbols {
            println!("data symbol: {}", sym);
        }
    } else {
        let isa: &TargetIsa = isa.expect("compilation requires a target isa");

        let mut obj = Artifact::new(faerie_target(isa)?, String::from(arg_output));

        for import in &module.imports {
            obj.import(
                module.strings.get_str(&import.0),
                translate_symbolkind(import.1),
            ).expect("faerie import");
        }
        for func in &module.functions {
            // FIXME: non-global functions.
            obj.declare(
                module.strings.get_str(&func.il.name),
                Decl::Function { global: true },
            ).expect("faerie declare");
        }
        // FIXME: non-global and non-writeable data.
        for data in &module.data_symbols {
            obj.declare(
                module.strings.get_str(&data.name),
                Decl::Data {
                    global: true,
                    writeable: true,
                },
            ).expect("faerie declare");
        }
        for func in module.functions {
            let func_name = module.strings.get_str(&func.il.name);
            let compilation = func.compilation.unwrap();
            obj.define(&func_name, compilation.body)
                .expect("faerie define");
            // TODO: reloc should derive from Copy
            for &(ref reloc, ref external_name, offset, addend) in &compilation.relocs.relocs {
                // FIXME: What about other types of relocs?
                // TODO: Faerie API: Seems like it might be inefficient to
                // identify the caller by name each time.
                // TODO: Faerie API: It's inconvenient to keep track of which
                // symbols are imports and which aren't.
                debug_assert!(addend as i32 as i64 == addend);
                let addend = addend as i32;
                let name = module.strings.get_str(external_name);
                obj.link_with(
                    Link {
                        from: &func_name,
                        to: &name,
                        at: offset as usize,
                    },
                    RelocOverride {
                        reloc: translate_reloc(reloc),
                        addend: addend,
                    },
                ).expect("faerie link");
            }
        }
        for data in module.data_symbols {
            let data_name = module.strings.get_str(&data.name);
            obj.define(data_name, data.contents.clone())
                .expect("faerie define");
        }

        let file = ::std::fs::File::create(Path::new(arg_output))
            .map_err(|x| format(format_args!("{}", x)))?;

        // FIXME: Make the format a parameter.
        obj.write::<Elf>(file)
            .map_err(|x| format(format_args!("{}", x)))?;
    }

    terminal.fg(term::color::GREEN).unwrap();
    vprintln!(flag_verbose, "ok");
    terminal.reset().unwrap();
    Ok(())
}

// TODO: Reloc should by Copy
fn translate_reloc(reloc: &Reloc) -> u32 {
    match *reloc {
        Reloc::IntelPCRel4 => elf::reloc::R_X86_64_PC32,
        Reloc::IntelAbs4 => elf::reloc::R_X86_64_32,
        Reloc::IntelAbs8 => elf::reloc::R_X86_64_64,
        Reloc::IntelGOTPCRel4 => elf::reloc::R_X86_64_GOTPCREL,
        Reloc::IntelPLTRel4 => elf::reloc::R_X86_64_PLT32,
        _ => panic!("unsupported reloc kind"),
    }
}

fn translate_symbolkind(kind: SymbolKind) -> ImportKind {
    match kind {
        SymbolKind::Function => ImportKind::Function,
        SymbolKind::Data => ImportKind::Data,
    }
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
