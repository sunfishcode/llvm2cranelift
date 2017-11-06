//! llvm2cretonne driver program.

extern crate cretonne;
extern crate cton_llvm;
extern crate cton_reader;
extern crate docopt;
extern crate term;
#[macro_use]
extern crate serde_derive;
extern crate faerie;

use cretonne::VERSION;
use docopt::Docopt;
use std::io::{self, Write};
use std::process;

mod llvm;
mod utils;

const USAGE: &str = "
Cretonne code generator utility

Usage:
    llvm2cretonne -p [-v] [--set <set>]... [--isa <isa>] <file>...
    llvm2cretonne [-v] [--set <set>]... [--isa <isa>] -o <output> <file>...
    llvm2cretonne --help | --version

Options:
    -v, --verbose   be more verbose
    -p, --print     print the resulting Cretonne IL
    -h, --help      print this help message
    --set=<set>     configure Cretonne settings
    --isa=<isa>     specify the Cretonne ISA
    -o, --output    specify the output file
    --version       print the Cretonne version

";

#[derive(Deserialize, Debug)]
struct Args {
    arg_file: Vec<String>,
    arg_output: String,
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
        &args.arg_output,
        args.flag_verbose,
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
