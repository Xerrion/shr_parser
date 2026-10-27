use std::{fmt, io};

use serde::{Deserialize, Serialize};

/// Struct representing the header of an SHR file.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SHRFileHeader {
    pub(crate) signature: u16,
    pub(crate) version: u16,
    pub(crate) reserved1: u32,
    pub(crate) data_offset: u64,
    pub(crate) sweep_count: u32,
    pub(crate) sweep_length: u32,
    pub(crate) first_bin_freq_hz: f64,
    pub(crate) bin_size_hz: f64,
    pub(crate) title: Vec<u16>,
    pub(crate) center_freq_hz: f64,
    pub(crate) span_hz: f64,
    pub(crate) rbw_hz: f64,
    pub(crate) vbw_hz: f64,
    pub(crate) ref_level: f32,
    pub(crate) ref_scale: SHRScale,
    pub(crate) div: f32,
    pub(crate) window: u32,
    pub(crate) attenuation: i32,
    pub(crate) gain: i32,
    pub(crate) detector: i32,
    pub(crate) processing_units: i32,
    pub(crate) window_bandwidth: f64,
    pub(crate) decimation_type: SHRDecimationType,
    pub(crate) decimation_detector: SHRDecimationDetector,
    pub(crate) decimation_count: i32,
    pub(crate) decimation_time_ms: i32,
    pub(crate) channelize_enabled: i32,
    pub(crate) channel_output_units: i32,
    pub(crate) channel_center_hz: f64,
    pub(crate) channel_width_hz: f64,
    pub(crate) reserved2: [u32; 16],
}

/// Struct representing the header of a sweep within an SHR file.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SHRSweepHeader {
    pub timestamp: u64,
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
    pub(crate) adc_overflow: u8,
    pub(crate) reserved: [u8; 15],
}

/// A frequency/amplitude pair from one sweep bin.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct SHRSweepPoint {
    /// Bin frequency in Hz.
    pub frequency_hz: f64,
    /// Unmodified amplitude in the file's reference scale (dBm or mV).
    pub amplitude: f32,
}

/// Struct representing a sweep within an SHR file.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SHRSweep {
    pub number: i32,
    pub frequency: f64,
    pub amplitude: f64,
    pub header: SHRSweepHeader,
    /// All bins in file order, independent of the selected summary metric.
    /// Defaults to empty when deserializing data written before this field existed.
    #[serde(default)]
    pub data: Vec<SHRSweepPoint>,
}

/// Struct representing the entire SHR file, including its header and sweeps.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SHRFile {
    pub file_header: SHRFileHeader,
    pub sweeps: Vec<SHRSweep>,
}

/// Enumeration representing the scale used in the SHR file.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) enum SHRScale {
    #[serde(rename = "dBm")]
    Dbm = 0,
    #[serde(rename = "mV")]
    MV = 1,
}

/// Enumeration representing the type of decimation used in the SHR file.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) enum SHRDecimationType {
    Time = 0,
    Count = 1,
}

/// Enumeration representing the detector used for decimation in the SHR file.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) enum SHRDecimationDetector {
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

impl TryFrom<i32> for SHRScale {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Dbm),
            1 => Ok(Self::MV),
            _ => Err("Invalid value for SHRScale"),
        }
    }
}

impl TryFrom<i32> for SHRDecimationDetector {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Average),
            1 => Ok(Self::Max),
            _ => Err("Invalid value for SHRDecimationDetector"),
        }
    }
}

impl TryFrom<i32> for SHRDecimationType {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Time),
            1 => Ok(Self::Count),
            _ => Err("Invalid value for SHRDecimationType"),
        }
    }
}

impl TryFrom<i32> for SHRParsingType {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Peak),
            1 => Ok(Self::Mean),
            2 => Ok(Self::Low),
            _ => Err("Invalid value for SHRParsingType"),
        }
    }
}

impl fmt::Display for SHRFileError {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::IOError(err) => write!(formatter, "IO error: {}", err),
            Self::InvalidSignature => write!(formatter, "Invalid file signature"),
            Self::InvalidVersion => write!(formatter, "Invalid file version"),
            Self::InvalidFile => write!(formatter, "Invalid file"),
        }
    }
}

impl From<io::Error> for SHRFileError {
    fn from(err: io::Error) -> Self {
        Self::IOError(err)
    }
}
