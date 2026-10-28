use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::decode;
use crate::types::{SHRFile, SHRFileError, SHRFileHeader, SHRParsingType, SHRSweep};

/// A parsed SHR file together with its source path.
#[derive(Serialize, Deserialize, Debug)]
pub struct SHRParser {
    pub file_path: PathBuf,
    pub shr_file: SHRFile,
}

impl SHRFile {
    /// Opens and parses an SHR file using the selected sweep summary metric.
    pub fn new<P: AsRef<Path>>(
        path: P,
        parsing_type: SHRParsingType,
    ) -> Result<Self, SHRFileError> {
        decode::parse_file(path.as_ref(), parsing_type)
    }
}

impl SHRParser {
    /// Opens and parses an SHR file, retaining its source path.
    pub fn new(file_path: PathBuf, parsing_type: SHRParsingType) -> Result<Self, SHRFileError> {
        let shr_file = SHRFile::new(&file_path, parsing_type)?;
        Ok(Self {
            file_path,
            shr_file,
        })
    }

    /// Returns the CSV representation of the file metadata and sweep summaries.
    pub fn to_str(&self) -> String {
        let mut output = Vec::new();
        self.write_csv(&mut output)
            .expect("writing CSV to memory cannot fail");
        String::from_utf8(output).expect("CSV output is always UTF-8")
    }

    /// Streams the CSV representation to a file, reporting write and flush errors.
    pub fn to_csv(&self, path: PathBuf) -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        self.write_csv(&mut writer)?;
        writer.flush()
    }

    /// Returns a clone of the parsed sweeps.
    pub fn get_sweeps(&self) -> Vec<SHRSweep> {
        self.shr_file.sweeps.clone()
    }

    /// Returns a clone of the file header.
    pub fn get_file_header(&self) -> SHRFileHeader {
        self.shr_file.file_header.clone()
    }

    /// Returns a clone of the source path.
    pub fn get_file_path(&self) -> PathBuf {
        self.file_path.clone()
    }

    fn write_csv<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        self.shr_file
            .file_header
            .write_csv_preamble(writer, &self.file_path)?;
        writer.write_all(b"Sweep,Timestamp,Peak Freq MHz,Peak Ampl dBM,Latitude,Longitude\n")?;

        for sweep in &self.shr_file.sweeps {
            writeln!(
                writer,
                "{},{},{},{},{},{}",
                sweep.number,
                sweep.header.timestamp,
                sweep.frequency,
                sweep.amplitude,
                sweep.header.latitude,
                sweep.header.longitude,
            )?;
        }
        Ok(())
    }
}
