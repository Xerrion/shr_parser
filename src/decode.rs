use std::fs;
use std::io::{self, Read};
use std::path::Path;

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use rayon::prelude::*;

use crate::types::{
    SHRDecimationDetector, SHRDecimationType, SHRFile, SHRFileError, SHRFileHeader, SHRParsingType,
    SHRScale, SHRSweep, SHRSweepHeader, SHRSweepPoint,
};

const FILE_HEADER_SIZE: usize = 472;
const SWEEP_HEADER_SIZE: usize = 48;
const BIN_SIZE: usize = 4;

pub(crate) fn parse_file(
    path: &Path,
    parsing_type: SHRParsingType,
) -> Result<SHRFile, SHRFileError> {
    // Own the bytes so concurrent changes to the source cannot invalidate readers.
    let bytes = fs::read(path)?;
    let buffer = bytes.as_slice();
    let file_header = read_file_header(byte_range(buffer, 0, FILE_HEADER_SIZE)?)?;
    file_header.validate()?;
    let sweeps = parse_sweeps(buffer, &file_header, parsing_type)?;

    Ok(SHRFile {
        file_header,
        sweeps,
    })
}

fn read_file_header(mut reader: &[u8]) -> io::Result<SHRFileHeader> {
    Ok(SHRFileHeader {
        signature: reader.read_u16::<LittleEndian>()?,
        version: reader.read_u16::<LittleEndian>()?,
        reserved1: reader.read_u32::<LittleEndian>()?,
        data_offset: reader.read_u64::<LittleEndian>()?,
        sweep_count: reader.read_u32::<LittleEndian>()?,
        sweep_length: reader.read_u32::<LittleEndian>()?,
        first_bin_freq_hz: reader.read_f64::<LittleEndian>()?,
        bin_size_hz: reader.read_f64::<LittleEndian>()?,
        title: {
            let mut title = Vec::with_capacity(128);
            for _ in 0..128 {
                title.push(reader.read_u16::<LittleEndian>()?);
            }
            title
        },
        center_freq_hz: reader.read_f64::<LittleEndian>()?,
        span_hz: reader.read_f64::<LittleEndian>()?,
        rbw_hz: reader.read_f64::<LittleEndian>()?,
        vbw_hz: reader.read_f64::<LittleEndian>()?,
        ref_level: reader.read_f32::<LittleEndian>()?,
        ref_scale: SHRScale::try_from(reader.read_i32::<LittleEndian>()?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
        div: reader.read_f32::<LittleEndian>()?,
        window: reader.read_u32::<LittleEndian>()?,
        attenuation: reader.read_i32::<LittleEndian>()?,
        gain: reader.read_i32::<LittleEndian>()?,
        detector: reader.read_i32::<LittleEndian>()?,
        processing_units: reader.read_i32::<LittleEndian>()?,
        window_bandwidth: reader.read_f64::<LittleEndian>()?,
        decimation_type: SHRDecimationType::try_from(reader.read_i32::<LittleEndian>()?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
        decimation_detector: SHRDecimationDetector::try_from(reader.read_i32::<LittleEndian>()?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
        decimation_count: reader.read_i32::<LittleEndian>()?,
        decimation_time_ms: reader.read_i32::<LittleEndian>()?,
        channelize_enabled: reader.read_i32::<LittleEndian>()?,
        channel_output_units: reader.read_i32::<LittleEndian>()?,
        channel_center_hz: reader.read_f64::<LittleEndian>()?,
        channel_width_hz: reader.read_f64::<LittleEndian>()?,
        reserved2: {
            let mut reserved = [0; 16];
            for value in &mut reserved {
                *value = reader.read_u32::<LittleEndian>()?;
            }
            reserved
        },
    })
}

fn parse_sweeps(
    buffer: &[u8],
    file_header: &SHRFileHeader,
    parsing_type: SHRParsingType,
) -> Result<Vec<SHRSweep>, SHRFileError> {
    let sweep_count =
        usize::try_from(file_header.sweep_count).map_err(|_| SHRFileError::InvalidFile)?;
    if sweep_count == 0 {
        return Ok(Vec::new());
    }

    let sweep_length =
        usize::try_from(file_header.sweep_length).map_err(|_| SHRFileError::InvalidFile)?;
    let data_size = sweep_length
        .checked_mul(BIN_SIZE)
        .ok_or(SHRFileError::InvalidFile)?;
    let sweep_size = SWEEP_HEADER_SIZE
        .checked_add(data_size)
        .ok_or(SHRFileError::InvalidFile)?;
    let data_offset =
        usize::try_from(file_header.data_offset).map_err(|_| SHRFileError::InvalidFile)?;
    if data_offset < FILE_HEADER_SIZE {
        return Err(SHRFileError::InvalidFile);
    }
    let total_size = sweep_count
        .checked_mul(sweep_size)
        .ok_or(SHRFileError::InvalidFile)?;
    let sweep_bytes = byte_range(buffer, data_offset, total_size)?;

    // The checked range contains exactly sweep_count complete, fixed-size sweeps.
    // Parallel chunks are indexed, so collection retains the file's sweep order.
    sweep_bytes
        .par_chunks(sweep_size)
        .enumerate()
        .map(|(number, bytes)| parse_sweep(bytes, number, file_header, parsing_type))
        .collect()
}

fn parse_sweep(
    mut reader: &[u8],
    number: usize,
    file_header: &SHRFileHeader,
    parsing_type: SHRParsingType,
) -> Result<SHRSweep, SHRFileError> {
    let header = read_sweep_header(&mut reader)?;
    let (bins, _) = reader.as_chunks::<BIN_SIZE>();
    let points = bins
        .iter()
        .enumerate()
        .map(|(index, bytes)| SHRSweepPoint {
            frequency_hz: file_header.first_bin_freq_hz + index as f64 * file_header.bin_size_hz,
            amplitude: LittleEndian::read_f32(bytes),
        })
        .collect();

    Ok(SHRSweep::from_points(
        number as i32,
        header,
        points,
        parsing_type,
        file_header.first_bin_freq_hz,
        file_header.bin_size_hz,
    ))
}

fn read_sweep_header(reader: &mut &[u8]) -> io::Result<SHRSweepHeader> {
    Ok(SHRSweepHeader {
        timestamp: reader.read_u64::<LittleEndian>()?,
        latitude: reader.read_f64::<LittleEndian>()?,
        longitude: reader.read_f64::<LittleEndian>()?,
        altitude: reader.read_f64::<LittleEndian>()?,
        adc_overflow: reader.read_u8()?,
        reserved: {
            let mut reserved = [0; 15];
            reader.read_exact(&mut reserved)?;
            reserved
        },
    })
}

fn byte_range(buffer: &[u8], start: usize, length: usize) -> Result<&[u8], SHRFileError> {
    let end = start.checked_add(length).ok_or(SHRFileError::InvalidFile)?;
    buffer
        .get(start..end)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "SHR file is truncated").into())
}
