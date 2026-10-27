//! Parses and exposes data from Signal Hound Spike spectrum analyzer files.
//!
//! File decoding, domain types, sweep calculations, and parser output are kept
//! in focused modules while the existing public API remains available here.
mod decode;
mod header;
mod parser;
mod sweep;
mod types;

pub use parser::SHRParser;
pub use types::{
    BasicHeaderInfo, SHRFile, SHRFileError, SHRFileHeader, SHRParsingType, SHRSweep,
    SHRSweepHeader, SHRSweepPoint,
};
