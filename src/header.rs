use std::io::{self, Write};
use std::path::Path;

use crate::types::{
    BasicHeaderInfo, SHRDecimationDetector, SHRDecimationType, SHRFileError, SHRFileHeader,
    SHRScale,
};

impl SHRFileHeader {
    const EXPECTED_SIGNATURE: u16 = 0xAA10;
    const EXPECTED_VERSION_1: u16 = 1;
    const EXPECTED_VERSION_2: u16 = 2;

    pub(crate) fn validate(&self) -> Result<(), SHRFileError> {
        if self.signature == Self::EXPECTED_SIGNATURE && self.has_supported_version() {
            Ok(())
        } else {
            Err(SHRFileError::InvalidFile)
        }
    }

    pub(crate) fn write_csv_preamble<W: Write>(
        &self,
        writer: &mut W,
        path: &Path,
    ) -> io::Result<()> {
        // The existing CSV describes signature validity using the version, not the signature.
        let validity = if self.has_supported_version() {
            "valid"
        } else {
            "not valid"
        };
        writeln!(
            writer,
            "#File signature is {}; Found file signature {:#X}, expected: {:#X}",
            validity,
            self.signature,
            Self::EXPECTED_SIGNATURE
        )?;
        writeln!(
            writer,
            "#File version is {}; Found file version {:#X}, expected: {:#X} or {:#X}",
            validity,
            self.version,
            Self::EXPECTED_VERSION_1,
            Self::EXPECTED_VERSION_2
        )?;
        let filename = path.to_string_lossy();
        if filename
            .bytes()
            .any(|byte| matches!(byte, b',' | b'"' | b'\r' | b'\n'))
        {
            writer.write_all(b"\"#File name: ")?;
            let mut start = 0;
            for (index, byte) in filename.bytes().enumerate() {
                let escaped: &[u8] = match byte {
                    b'"' => b"\"\"",
                    b'\r' => b"\\r",
                    b'\n' => b"\\n",
                    _ => continue,
                };
                writer.write_all(&filename.as_bytes()[start..index])?;
                writer.write_all(escaped)?;
                start = index + 1;
            }
            writer.write_all(&filename.as_bytes()[start..])?;
            writer.write_all(b"\"\n")?;
        } else {
            writeln!(writer, "#File name: {}", filename)?;
        }

        let info = self.get_basic_header_info();
        write!(
            writer,
            "#Sweep count: {}\n\
             #Sweep size: {}\n\
             #Sweep Start Freq: {}\n\
             #Sweep Bin Size: {}\n\
             #Sweep Center Freq: {}\n\
             #Sweep Span Freq: {}\n\
             #Sweep Freq Range: {} MHz to {} MHz\n\
             #RBW: {} kHz\n\
             #Reference Level: {} {}\n",
            info.sweep_count,
            info.sweep_length,
            info.first_bin_freq_hz,
            info.bin_size_hz,
            info.center_freq_hz,
            info.span_hz,
            info.min_freq,
            info.max_freq,
            info.rbw_hz,
            info.ref_level,
            info.ref_scale
        )?;

        match (&self.decimation_type, &self.decimation_detector) {
            (SHRDecimationType::Count, SHRDecimationDetector::Average) => writeln!(
                writer,
                "#Averaged {} traces(s) per output trace",
                self.decimation_count
            )?,
            (SHRDecimationType::Count, SHRDecimationDetector::Max) => writeln!(
                writer,
                "#Max held {} traces(s) per output trace",
                self.decimation_count
            )?,
            (SHRDecimationType::Time, SHRDecimationDetector::Average) => writeln!(
                writer,
                "#Averaged traces(s) for {} seconds per output trace",
                self.decimation_time_ms / 1000
            )?,
            (SHRDecimationType::Time, SHRDecimationDetector::Max) => writeln!(
                writer,
                "#Max held trace(s) for {} seconds per output trace",
                self.decimation_time_ms / 1000
            )?,
        }

        let channelized = if self.channelize_enabled != 0 {
            "channelized"
        } else {
            "not channelized"
        };
        writeln!(writer, "#Was {}", channelized)
    }

    fn has_supported_version(&self) -> bool {
        matches!(
            self.version,
            Self::EXPECTED_VERSION_1 | Self::EXPECTED_VERSION_2
        )
    }

    fn get_basic_header_info(&self) -> BasicHeaderInfo {
        BasicHeaderInfo {
            min_freq: (self.center_freq_hz - self.span_hz / 2.0) * 1e-6,
            max_freq: (self.center_freq_hz + self.span_hz / 2.0) * 1e-6,
            ref_scale: match self.ref_scale {
                SHRScale::Dbm => "dBm",
                SHRScale::MV => "mV",
            },
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
}
