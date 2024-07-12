//! This module provides functionalities to parse and handle SHR files.
//!
//! The SHR file format includes a file header and multiple sweeps. Each sweep
//! has its own header and data. This module uses memory mapping for efficient
//! file reading and rayon for parallel processing of sweeps.

use std::fmt;
use std::fs::File;
use std::io::{self, Cursor, Read, Write};
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt};
use memmap::MmapOptions;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Struct representing the header of an SHR file.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SHRFileHeader {
    signature: u16,
    version: u16,
    reserved1: u32,
    data_offset: u64,
    sweep_count: u32,
    sweep_length: u32,
    first_bin_freq_hz: f64,
    bin_size_hz: f64,
    title: Vec<u16>,
    center_freq_hz: f64,
    span_hz: f64,
    rbw_hz: f64,
    vbw_hz: f64,
    ref_level: f32,
    ref_scale: SHRScale,
    div: f32,
    window: u32,
    attenuation: i32,
    gain: i32,
    detector: i32,
    processing_units: i32,
    window_bandwidth: f64,
    decimation_type: SHRDecimationType,
    decimation_detector: SHRDecimationDetector,
    decimation_count: i32,
    decimation_time_ms: i32,
    channelize_enabled: i32,
    channel_output_units: i32,
    channel_center_hz: f64,
    channel_width_hz: f64,
    reserved2: [u32; 16],
}

/// Struct representing the header of a sweep within an SHR file.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SHRSweepHeader {
    timestamp: u64,
    latitude: f64,
    longitude: f64,
    altitude: f64,
    adc_overflow: u8,
    reserved: [u8; 15],
}

/// Struct representing a sweep within an SHR file.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SHRSweep {
    #[serde(rename = "Sweep")]
    pub sweep_number: i32,
    #[serde(rename = "Timestamp")]
    pub timestamp: u64,
    #[serde(rename = "Peak Freq MHz")]
    pub frequency: f64,
    #[serde(rename = "Peak Ampl dBM")]
    pub amplitude: f64,
}

/// Struct representing the entire SHR file, including its header and sweeps.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SHRFile {
    pub file_header: SHRFileHeader,
    pub sweeps: Vec<SHRSweep>,
}

/// Struct representing a parser for SHR files.
#[derive(Serialize, Deserialize, Debug)]
pub struct SHRParser {
    pub file_path: PathBuf,
    pub shr_file: SHRFile,
}

/// Enumeration representing the scale used in the SHR file.
#[derive(Serialize, Deserialize, Debug, Clone)]
enum SHRScale {
    #[serde(rename = "dBm")]
    Dbm = 0,
    #[serde(rename = "mV")]
    MV = 1,
}

/// Enumeration representing the type of decimation used in the SHR file.
#[derive(Serialize, Deserialize, Debug, Clone)]
enum SHRDecimationType {
    Time = 0,
    Count = 1,
}

/// Enumeration representing the detector used for decimation in the SHR file.
#[derive(Serialize, Deserialize, Debug, Clone)]
enum SHRDecimationDetector {
    Average = 0,
    Max = 1,
}

/// Enumeration representing the types of parsing available for an SHR file.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
pub enum SHRParsingType {
    Peak = 0,
    Mean = 1,
    Low = 2,
}

/// Enumeration representing possible errors that can occur when handling SHR files.
#[derive(Debug)]
pub enum SHRFileError {
    IOError(io::Error),
    InvalidSignature,
    InvalidVersion,
    InvalidFile,
}

pub struct BasicHeaderInfo {
    pub min_freq: f64,
    pub max_freq: f64,
    pub ref_scale: &'static str,
    pub sweep_count: u32,
    pub sweep_length: u32,
    pub first_bin_freq_hz: f64,
    pub bin_size_hz: f64,
    pub center_freq_hz: f64,
    pub span_hz: f64,
    pub rbw_hz: f64,
    pub ref_level: f32,
}

/// Implementation for converting integer values to SHRScale enumeration values.
impl TryFrom<i32> for SHRScale {
    type Error = &'static str;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(SHRScale::Dbm),
            1 => Ok(SHRScale::MV),
            _ => Err("Invalid value for SHRScale"),
        }
    }
}

/// Implementation for converting integer values to SHRDecimationDetector enumeration values.
impl TryFrom<i32> for SHRDecimationDetector {
    type Error = &'static str;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(SHRDecimationDetector::Average),
            1 => Ok(SHRDecimationDetector::Max),
            _ => Err("Invalid value for SHRDecimationDetector"),
        }
    }
}

/// Implementation for converting integer values to SHRDecimationType enumeration values.
impl TryFrom<i32> for SHRDecimationType {
    type Error = &'static str;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(SHRDecimationType::Time),
            1 => Ok(SHRDecimationType::Count),
            _ => Err("Invalid value for SHRDecimationType"),
        }
    }
}

/// Implementation for converting integer values to SHRParsingType enumeration values.
impl TryFrom<i32> for SHRParsingType {
    type Error = &'static str;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(SHRParsingType::Peak),
            1 => Ok(SHRParsingType::Mean),
            2 => Ok(SHRParsingType::Low),
            _ => Err("Invalid value for SHRParsingType"),
        }
    }
}

/// Implementation for displaying SHRParsingType enumeration values.
impl fmt::Display for SHRParsingType {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SHRParsingType::Peak => write!(formatter, "SHRParsingType.Peak"),
            SHRParsingType::Mean => write!(formatter, "SHRParsingType.Mean"),
            SHRParsingType::Low => write!(formatter, "SHRParsingType.Low"),
        }
    }
}

/// Implementation for displaying SHRFileError enumeration values.
impl fmt::Display for SHRFileError {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SHRFileError::IOError(ref err) => write!(formatter, "IO error: {}", err),
            SHRFileError::InvalidSignature => write!(formatter, "Invalid file signature"),
            SHRFileError::InvalidVersion => write!(formatter, "Invalid file version"),
            SHRFileError::InvalidFile => write!(formatter, "Invalid file"),
        }
    }
}

/// Implementation for converting io::Error values to SHRFileError enumeration values.
impl From<io::Error> for SHRFileError {
    fn from(err: io::Error) -> SHRFileError {
        SHRFileError::IOError(err)
    }
}

impl SHRSweep {
    /// Constructs a new `SHRSweep` instance.
    ///
    /// # Arguments
    ///
    /// * `sweep_number` - The number of the sweep.
    /// * `sweep_header` - The header of the sweep.
    /// * `sweep_data_raw` - The raw data of the sweep.
    /// * `parsing_type` - The type of parsing to be performed.
    /// * `first_bin_freq_hz` - The frequency of the first bin in Hz.
    /// * `bin_size_hz` - The size of each bin in Hz.
    pub fn new(
        sweep_number: i32,
        sweep_header: SHRSweepHeader,
        sweep_data_raw: Vec<f32>,
        parsing_type: SHRParsingType,
        first_bin_freq_hz: f64,
        bin_size_hz: f64,
    ) -> Self {
        let (out_freq, out_power) = Self::calculate_sweep_metrics(
            sweep_data_raw.as_slice(),
            parsing_type,
            first_bin_freq_hz,
            bin_size_hz,
        );

        Self {
            sweep_number,
            timestamp: sweep_header.timestamp,
            frequency: out_freq,
            amplitude: out_power,
        }
    }

    /// Calculates metrics for a sweep based on the parsing type.
    ///
    /// # Arguments
    ///
    /// * `sweep_data` - The data of the sweep.
    /// * `parsing_type` - The type of parsing to be performed.
    /// * `first_bin_freq_hz` - The frequency of the first bin in Hz.
    /// * `bin_size_hz` - The size of each bin in Hz.
    ///
    /// # Returns
    ///
    /// A tuple containing the frequency and power.
    fn calculate_sweep_metrics(
        sweep_data: &[f32],
        parsing_type: SHRParsingType,
        first_bin_freq_hz: f64,
        bin_size_hz: f64,
    ) -> (f64, f64) {
        match parsing_type {
            SHRParsingType::Peak => {
                let (peak_index, &peak_power) = sweep_data
                    .iter()
                    .enumerate()
                    .max_by(|(_, &a), (_, &b)| {
                        a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or((0, &0.0));

                let peak_freq = (first_bin_freq_hz + peak_index as f64 * bin_size_hz) / 1.0e6;
                (peak_freq, peak_power as f64)
            }
            SHRParsingType::Mean => {
                let mean_power = if !sweep_data.is_empty() {
                    sweep_data.iter().sum::<f32>() / sweep_data.len() as f32
                } else {
                    0.0
                };

                let mean_freq =
                    (first_bin_freq_hz + bin_size_hz * sweep_data.len() as f64 / 2.0) / 1.0e6;
                (mean_freq, mean_power as f64)
            }
            SHRParsingType::Low => {
                let (low_index, &low_power) = sweep_data
                    .iter()
                    .enumerate()
                    .min_by(|(_, &a), (_, &b)| {
                        a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or((0, &0.0));

                let low_freq = (first_bin_freq_hz + low_index as f64 * bin_size_hz) / 1.0e6;
                (low_freq, low_power as f64)
            }
        }
    }
}

impl SHRFileHeader {
    const EXPECTED_SIGNATURE: u16 = 0xAA10; // Expected signature value
    const EXPECTED_VERSION_1: u16 = 0x1; // Expected version value 1
    const EXPECTED_VERSION_2: u16 = 0x2; // Expected version value 2

    /// Validates the file signature.
    ///
    /// # Returns
    ///
    /// A result indicating whether the signature is valid.
    fn validate_signature(&self) -> Result<(), SHRFileError> {
        if self.signature != Self::EXPECTED_SIGNATURE {
            Err(SHRFileError::InvalidSignature)
        } else {
            Ok(())
        }
    }

    /// Validates the file version.
    ///
    /// # Returns
    ///
    /// A result indicating whether the version is valid.
    fn validate_version(&self) -> Result<(), SHRFileError> {
        match self.version {
            Self::EXPECTED_VERSION_1 | Self::EXPECTED_VERSION_2 => Ok(()),
            _ => Err(SHRFileError::InvalidVersion),
        }
    }

    /// Returns a string representation of the file version.
    ///
    /// # Returns
    ///
    /// A string indicating whether the file version is valid.
    fn get_version(&self) -> String {
        match self.validate_version() {
            Ok(_) => {
                format!(
                    "#File version is valid; Found file version {:#X}, expected: {:#X} or {:#X}\n",
                    self.version,
                    Self::EXPECTED_VERSION_1,
                    Self::EXPECTED_VERSION_2
                )
            }
            Err(_) => {
                format!(
                    "#File version is not valid; Found file version {:#X}, expected: {:#X} or {:#X}\n",
                    self.version,
                    Self::EXPECTED_VERSION_1,
                    Self::EXPECTED_VERSION_2
                )
            }
        }
    }

    /// Returns a string representation of the file signature.
    ///
    /// # Returns
    ///
    /// A string indicating whether the file signature is valid.
    fn get_signature(&self) -> String {
        match self.validate_version() {
            Ok(_) => {
                format!(
                    "#File signature is valid; Found file signature {:#X}, expected: {:#X}\n",
                    self.signature,
                    Self::EXPECTED_SIGNATURE
                )
            }
            Err(_) => {
                format!(
                    "#File signature is not valid; Found file signature {:#X}, expected: {:#X}\n",
                    self.signature,
                    Self::EXPECTED_SIGNATURE
                )
            }
        }
    }

    /// Returns a string representation of the reference scale.
    ///
    /// # Returns
    ///
    /// A string representing the reference scale.
    fn get_ref_scale(&self) -> &'static str {
        match self.ref_scale {
            SHRScale::Dbm => "dBm",
            SHRScale::MV => "mV",
        }
    }

    /// Returns a string representation of the basic file header information.
    ///
    /// # Returns
    ///
    /// A vector containing the basic file header information.
    fn get_basic_header_info(&self) -> BasicHeaderInfo {
        let min_freq = (self.center_freq_hz - self.span_hz / 2.0) * 1e-6;
        let max_freq = (self.center_freq_hz + self.span_hz / 2.0) * 1e-6;
        let ref_scale = self.get_ref_scale();
        BasicHeaderInfo {
            min_freq,
            max_freq,
            ref_scale,
            sweep_count: self.sweep_count,
            sweep_length: self.sweep_length,
            first_bin_freq_hz: self.first_bin_freq_hz,
            bin_size_hz: self.bin_size_hz,
            center_freq_hz: self.center_freq_hz,
            span_hz: self.span_hz,
            rbw_hz: self.rbw_hz,
            ref_level: self.ref_level,
        }
    }

    /// Returns a string representation of the decimation information.
    ///
    /// # Returns
    ///
    /// A string containing the decimation information.
    fn get_decimation_info(&self) -> String {
        let channelized = if self.channelize_enabled != 0 {
            "channelized"
        } else {
            "not channelized"
        };
        let is_averaged = matches!(self.decimation_detector, SHRDecimationDetector::Average);

        let decimation_info = match self.decimation_type {
            SHRDecimationType::Count => {
                if is_averaged {
                    format!(
                        "#Averaged {} traces(s) per output trace\n",
                        self.decimation_count
                    )
                } else {
                    format!(
                        "#Max held {} traces(s) per output trace\n",
                        self.decimation_count
                    )
                }
            }
            SHRDecimationType::Time => {
                if is_averaged {
                    format!(
                        "#Averaged traces(s) for {} seconds per output trace\n",
                        self.decimation_time_ms / 1000 // Assuming the time is in milliseconds
                    )
                } else {
                    format!(
                        "#Max held trace(s) for {} seconds per output trace\n",
                        self.decimation_time_ms / 1000 // Assuming the time is in milliseconds
                    )
                }
            }
        };
        format!("{}#Was {}\n", decimation_info, channelized)
    }

    fn get_formatted_header_info(&self) -> String {
        let basic_header_info = self.get_basic_header_info();
        format!(
            "#Sweep count: {}\n\
             #Sweep size: {}\n\
             #Sweep Start Freq: {}\n\
             #Sweep Bin Size: {}\n\
             #Sweep Center Freq: {}\n\
             #Sweep Span Freq: {}\n\
             #Sweep Freq Range: {} MHz to {} MHz\n\
             #RBW: {} kHz\n\
             #Reference Level: {} {}\n",
            basic_header_info.sweep_count,
            basic_header_info.sweep_length,
            basic_header_info.first_bin_freq_hz,
            basic_header_info.bin_size_hz,
            basic_header_info.center_freq_hz,
            basic_header_info.span_hz,
            basic_header_info.min_freq,
            basic_header_info.max_freq,
            basic_header_info.rbw_hz,
            basic_header_info.ref_level,
            basic_header_info.ref_scale
        )
    }
}

/// Trait extension for the `Read` trait, providing utility functions to read specific types in little-endian order.
trait FileReaderExt: Read {
    fn read_i32_le(&mut self) -> io::Result<i32> {
        self.read_i32::<LittleEndian>()
    }

    fn read_u32_le(&mut self) -> io::Result<u32> {
        self.read_u32::<LittleEndian>()
    }

    fn read_f32_le(&mut self) -> io::Result<f32> {
        self.read_f32::<LittleEndian>()
    }

    fn read_f64_le(&mut self) -> io::Result<f64> {
        self.read_f64::<LittleEndian>()
    }

    fn read_u64_le(&mut self) -> io::Result<u64> {
        self.read_u64::<LittleEndian>()
    }

    fn read_u16_le(&mut self) -> io::Result<u16> {
        self.read_u16::<LittleEndian>()
    }
}

impl<R: Read + ?Sized> FileReaderExt for R {}

impl SHRFile {
    /// Constructs a new `SHRFile` instance.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the SHR file.
    /// * `parsing_type` - The type of parsing to be performed.
    ///
    /// # Returns
    ///
    /// A result containing the new `SHRFile` instance or an error.
    pub fn new<P: AsRef<Path>>(
        path: P,
        parsing_type: SHRParsingType,
    ) -> Result<Self, SHRFileError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let buffer = &mmap[..];

        Self::parse_file(buffer, parsing_type)
    }

    /// Reads the SHR file header from a reader.
    ///
    /// # Arguments
    ///
    /// * `reader` - The reader to read the header from.
    ///
    /// # Returns
    ///
    /// A result containing the `SHRFileHeader` or an error.
    fn read_shr_header<R: Read>(reader: &mut R) -> io::Result<SHRFileHeader> {
        let read_array_u16 = |reader: &mut R, array: &mut [u16]| -> io::Result<()> {
            array.iter_mut().try_for_each(|item| {
                *item = reader.read_u16_le()?;
                Ok(())
            })
        };

        let read_array_u32 = |reader: &mut R, array: &mut [u32]| -> io::Result<()> {
            array.iter_mut().try_for_each(|item| {
                *item = reader.read_u32_le()?;
                Ok(())
            })
        };

        Ok(SHRFileHeader {
            signature: reader.read_u16_le()?,
            version: reader.read_u16_le()?,
            reserved1: reader.read_u32_le()?,
            data_offset: reader.read_u64_le()?,
            sweep_count: reader.read_u32_le()?,
            sweep_length: reader.read_u32_le()?,
            first_bin_freq_hz: reader.read_f64_le()?,
            bin_size_hz: reader.read_f64_le()?,
            title: {
                let mut title = [0u16; 128];
                read_array_u16(reader, &mut title)?;
                title.to_vec()
            },
            center_freq_hz: reader.read_f64_le()?,
            span_hz: reader.read_f64_le()?,
            rbw_hz: reader.read_f64_le()?,
            vbw_hz: reader.read_f64_le()?,
            ref_level: reader.read_f32_le()?,
            ref_scale: SHRScale::try_from(reader.read_i32_le()?)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
            div: reader.read_f32_le()?,
            window: reader.read_u32_le()?,
            attenuation: reader.read_i32_le()?,
            gain: reader.read_i32_le()?,
            detector: reader.read_i32_le()?,
            processing_units: reader.read_i32_le()?,
            window_bandwidth: reader.read_f64_le()?,
            decimation_type: SHRDecimationType::try_from(reader.read_i32_le()?)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
            decimation_detector: SHRDecimationDetector::try_from(reader.read_i32_le()?)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
            decimation_count: reader.read_i32_le()?,
            decimation_time_ms: reader.read_i32_le()?,
            channelize_enabled: reader.read_i32_le()?,
            channel_output_units: reader.read_i32_le()?,
            channel_center_hz: reader.read_f64_le()?,
            channel_width_hz: reader.read_f64_le()?,
            reserved2: {
                let mut reserved2 = [0u32; 16];
                read_array_u32(reader, &mut reserved2)?;
                reserved2
            },
        })
    }

    /// Validates the SHR file.
    ///
    /// # Returns
    ///
    /// A result indicating whether the file is valid.
    fn validate_file(&self) -> Result<(), SHRFileError> {
        let valid_signature = self.file_header.validate_signature();
        let valid_version = self.file_header.validate_version();

        if valid_signature.is_ok() && valid_version.is_ok() {
            Ok(())
        } else {
            Err(SHRFileError::InvalidFile)
        }
    }

    /// Parses the SHR file from a buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer containing the SHR file data.
    /// * `parsing_type` - The type of parsing to be performed.
    ///
    /// # Returns
    ///
    /// A result containing the new `SHRFile` instance or an error.
    fn parse_file(buffer: &[u8], parsing_type: SHRParsingType) -> Result<Self, SHRFileError> {
        let mut cursor = Cursor::new(buffer);
        let file_header = Self::read_shr_header(&mut cursor)?;
        let mut shr_file = Self {
            file_header,
            sweeps: Vec::new(),
        };

        if shr_file.validate_file().is_ok() {
            shr_file.parse_sweeps(buffer, parsing_type);
            Ok(shr_file)
        } else {
            Err(SHRFileError::InvalidFile)
        }
    }

    /// Calculates the size of a sweep in bytes.
    ///
    /// # Arguments
    ///
    /// * `sweep_length` - The length of the sweep.
    ///
    /// # Returns
    ///
    /// The size of the sweep in bytes.
    fn calculate_sweep_size_in_bytes(sweep_length: usize) -> usize {
        let header_size = size_of::<SHRSweepHeader>();
        let data_size = size_of::<f32>() * sweep_length;
        header_size + data_size
    }

    /// Reads a sweep header from a reader.
    ///
    /// # Arguments
    ///
    /// * `reader` - The reader to read the header from.
    ///
    /// # Returns
    ///
    /// A result containing the `SHRSweepHeader` or an error.
    fn read_sweep_header<R: Read>(reader: &mut R) -> io::Result<SHRSweepHeader> {
        Ok(SHRSweepHeader {
            timestamp: reader.read_u64_le()?,
            latitude: reader.read_f64_le()?,
            longitude: reader.read_f64_le()?,
            altitude: reader.read_f64_le()?,
            adc_overflow: reader.read_u8()?,
            reserved: {
                let mut buf = [0u8; 15];
                reader.read_exact(&mut buf)?;
                buf
            },
        })
    }

    /// Reads sweep data from a reader.
    ///
    /// # Arguments
    ///
    /// * `reader` - The reader to read the data from.
    /// * `sweep_length` - The length of the sweep.
    ///
    /// # Returns
    ///
    /// A result containing the sweep data as a vector of floats or an error.
    fn read_sweep_data<R: Read>(reader: &mut R, sweep_length: usize) -> io::Result<Vec<f32>> {
        let mut buffer = vec![0u8; sweep_length * size_of::<f32>()];
        reader.read_exact(&mut buffer)?;

        // Use transmute to convert the byte buffer to Vec<f32>
        let floats = unsafe {
            Vec::from_raw_parts(buffer.as_mut_ptr() as *mut f32, sweep_length, sweep_length)
        };

        // Prevent the original buffer from being deallocated
        std::mem::forget(buffer);

        Ok(floats)
    }

    /// Parses the sweeps from the SHR file buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer containing the SHR file data.
    /// * `parsing_type` - The type of parsing to be performed.
    ///
    /// # Returns
    ///
    /// A result indicating success or an error.
    fn parse_sweeps(&mut self, buffer: &[u8], parsing_type: SHRParsingType) {
        let sweep_count = self.file_header.sweep_count as usize;
        let sweep_length = self.file_header.sweep_length as usize;
        let data_offset = self.file_header.data_offset;
        let sweep_size_in_bytes = Self::calculate_sweep_size_in_bytes(sweep_length);
        let buffer = Arc::new(buffer.to_vec());

        self.sweeps = (0..sweep_count)
            .into_par_iter()
            .map(|i| {
                let buffer = Arc::clone(&buffer);
                let mut cursor = Cursor::new(&buffer[data_offset as usize..]);
                cursor.set_position((sweep_size_in_bytes * i) as u64);

                let sweep_header = Self::read_sweep_header(&mut cursor).unwrap();
                let sweep_data = Self::read_sweep_data(&mut cursor, sweep_length).unwrap();
                drop(cursor);

                SHRSweep::new(
                    i as i32,
                    sweep_header,
                    sweep_data,
                    parsing_type,
                    self.file_header.first_bin_freq_hz,
                    self.file_header.bin_size_hz,
                )
            })
            .collect();

        // Drop the buffer to free the memory
        drop(buffer);
    }
}

impl SHRParser {
    /// Constructs a new `SHRParser` instance.
    ///
    /// # Arguments
    ///
    /// * `file_path` - The path to the SHR file.
    /// * `parsing_type` - The type of parsing to be performed.
    ///
    /// # Returns
    ///
    /// A result containing the new `SHRParser` instance or an error.
    pub fn new(file_path: PathBuf, parsing_type: SHRParsingType) -> Result<Self, SHRFileError> {
        let shr_file = SHRFile::new(&file_path, parsing_type)?;

        Ok(Self {
            file_path,
            shr_file,
        })
    }

    /// Returns the file information as a string.
    ///
    /// # Returns
    ///
    /// A string containing the file information.
    fn get_csv_header(&self) -> String {
        let file_name_display = format!("#File name: {}\n", self.file_path.display());
        let decimation_info = self.shr_file.file_header.get_decimation_info();
        let basic_header_display = self.shr_file.file_header.get_formatted_header_info();

        format!(
            "{}{}{}",
            file_name_display, basic_header_display, decimation_info
        )
    }

    /// Converts the SHR file to a string representation.
    ///
    /// # Returns
    ///
    /// A string representation of the SHR file.
    pub fn to_str(&self) -> String {
        let header_info = vec![
            self.shr_file.file_header.get_signature(),
            self.shr_file.file_header.get_version(),
            self.get_csv_header(),
        ];

        let sweep_info: Vec<String> = self
            .shr_file
            .sweeps
            .iter()
            .map(|sweep| {
                format!(
                    "{},{},{},{}\n",
                    sweep.sweep_number, sweep.timestamp, sweep.frequency, sweep.amplitude
                )
            })
            .collect();

        let mut data_holder = header_info;
        data_holder.push(String::from(
            "Sweep,Timestamp,Peak Freq MHz,Peak Ampl dBM\n",
        ));
        data_holder.extend(sweep_info);

        data_holder.join("")
    }

    /// Writes the SHR file data to a CSV file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the CSV file.
    ///
    /// # Returns
    ///
    /// A result indicating success or an error.
    pub fn to_csv(&self, path: PathBuf) -> io::Result<()> {
        let csv_data = self.to_str();
        let mut csv_file = File::create(path)?;
        csv_file.write_all(csv_data.as_bytes())?;
        Ok(())
    }

    pub fn get_sweeps(&self) -> Vec<SHRSweep> {
        self.shr_file.sweeps.clone()
    }

    pub fn get_file_header(&self) -> SHRFileHeader {
        self.shr_file.file_header.clone()
    }

    pub fn get_file_path(&self) -> PathBuf {
        self.file_path.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn sweeps_returned_successfully_for_valid_shr_file() {
        let parser = SHRParser::new(
            PathBuf::from("Raster2024-07-11 09h16m38s.shr"),
            SHRParsingType::Peak,
        )
        .unwrap();
        let sweeps = parser.get_sweeps();
        let header = parser.get_file_header();
        let file_path = parser.get_file_path();

        println!("{:?}", header);
        println!("{:?}", file_path);

        parser.to_csv(PathBuf::from("output.csv")).unwrap();

        assert!(
            !sweeps.is_empty(),
            "Sweeps should not be empty for a valid SHR file"
        );
    }

    #[test]
    fn sweeps_return_empty_for_shr_file_with_no_sweeps() {
        let parser = SHRParser::new(PathBuf::from("no_sweeps.shr"), SHRParsingType::Peak).unwrap();
        let sweeps = parser.get_sweeps();
        assert!(
            sweeps.is_empty(),
            "Sweeps should be empty for an SHR file with no sweeps"
        );
    }

    #[test]
    fn sweeps_return_correct_number_of_sweeps() {
        let parser =
            SHRParser::new(PathBuf::from("multiple_sweeps.shr"), SHRParsingType::Peak).unwrap();
        let sweeps = parser.get_sweeps();
        assert_eq!(
            sweeps.len(),
            5,
            "Sweeps count should match the actual number of sweeps in the SHR file"
        );
    }

    #[test]
    fn sweeps_return_error_for_invalid_shr_file() {
        let parser_result = SHRParser::new(PathBuf::from("invalid.shr"), SHRParsingType::Peak);
        assert!(
            parser_result.is_err(),
            "Parser should return an error for an invalid SHR file"
        );
    }

    #[test]
    fn sweeps_return_error_for_nonexistent_shr_file() {
        let parser_result = SHRParser::new(PathBuf::from("nonexistent.shr"), SHRParsingType::Peak);
        assert!(
            parser_result.is_err(),
            "Parser should return an error for a nonexistent SHR file"
        );
    }
}
