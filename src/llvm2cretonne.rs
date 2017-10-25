//! llvm2cretonne driver program.

extern crate cretonne;
extern crate cton_llvm;
extern crate cton_reader;
extern crate docopt;
extern crate term;
#[macro_use]
extern crate serde_derive;

use cretonne::VERSION;
use docopt::Docopt;
use std::io::{self, Write};
use std::process;

mod llvm;
mod utils;

const USAGE: &str = "
Cretonne code generator utility

Usage:
    llvm2cretonne [-ctvp] [--set <set>]... [--isa <isa>] <file>...
    llvm2cretonne --help | --version

Options:
    -v, --verbose   be more verbose
    -t, --just-decode
                    just decode WebAssembly to Cretonne IL
    -c, --check-translation
                    just checks the correctness of Cretonne IL translated from WebAssembly
    -p, --print     print the resulting Cretonne IL
    -h, --help      print this help message
    --set=<set>     configure Cretonne settings
    --isa=<isa>     specify the Cretonne ISA
    --version       print the Cretonne version

";

#[derive(Deserialize, Debug)]
struct Args {
    arg_file: Vec<String>,
    flag_just_decode: bool,
    flag_check_translation: bool,
    flag_print: bool,
    flag_verbose: bool,
    flag_set: Vec<String>,
    flag_isa: String,
}

/// A command either succeeds or fails with an error message.
pub type CommandResult = Result<(), String>;

/// Parse the command line arguments and run the requested command.
fn cton_util() -> CommandResult {
    // Parse command line arguments.
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| {
            d.help(true)
                .version(Some(format!("Cretonne {}", VERSION)))
                .deserialize()
        })
        .unwrap_or_else(|e| e.exit());

    llvm::run(
        args.arg_file,
        args.flag_verbose,
        args.flag_just_decode,
        args.flag_check_translation,
        args.flag_print,
        args.flag_set,
        args.flag_isa,
    )
}

fn main() {
    if let Err(mut msg) = cton_util() {
        if !msg.ends_with('\n') {
            msg.push('\n');
        }
        io::stdout().flush().expect("flushing stdout");
        io::stderr().write_all(msg.as_bytes()).unwrap();
        process::exit(1);
    }
}
