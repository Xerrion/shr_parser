use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use shr_parser::{SHRFileError, SHRParser, SHRParsingType};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/example_file/sweep.shr")
}

struct TemporaryFile(PathBuf);

impl TemporaryFile {
    fn new(contents: &[u8]) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "shr_parser-test-{}-{}.shr",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let temporary_file = Self(path);
        std::io::Write::write_all(&mut file, contents).unwrap();
        temporary_file
    }
}

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[test]
fn sweep_details_preserve_every_bin_for_all_summary_types() {
    let bytes = fs::read(fixture_path()).unwrap();
    let data_offset = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let sweep_count = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    let sweep_length = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
    let first_bin_hz = f64::from_le_bytes(bytes[24..32].try_into().unwrap());
    let bin_size_hz = f64::from_le_bytes(bytes[32..40].try_into().unwrap());
    assert_eq!(sweep_count, 417);
    assert_eq!(sweep_length, 16_386);

    for parsing_type in [
        SHRParsingType::Peak,
        SHRParsingType::Mean,
        SHRParsingType::Low,
    ] {
        let parser = SHRParser::new(fixture_path(), parsing_type).unwrap();
        assert_eq!(parser.get_file_path(), fixture_path());
        assert_eq!(parser.shr_file.sweeps.len(), sweep_count);
        for (index, sweep) in parser.shr_file.sweeps.iter().enumerate() {
            let offset = data_offset + index * (48 + sweep_length * 4);
            let timestamp = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            assert_eq!(sweep.number, index as i32);
            assert_eq!(sweep.get_header().timestamp, timestamp);
            assert_eq!(sweep.get_data().len(), sweep_length);
            let raw_amplitudes = &bytes[offset + 48..offset + 48 + sweep_length * 4];
            for (bin, (point, raw)) in sweep
                .get_data()
                .iter()
                .zip(raw_amplitudes.as_chunks::<4>().0)
                .enumerate()
            {
                assert_eq!(point.frequency_hz, first_bin_hz + bin as f64 * bin_size_hz);
                assert_eq!(point.amplitude.to_bits(), u32::from_le_bytes(*raw));
            }

            let points = sweep.get_data();
            let (expected_frequency, expected_amplitude) = match parsing_type {
                SHRParsingType::Peak | SHRParsingType::Low => {
                    let compare =
                        |a: &&shr_parser::SHRSweepPoint, b: &&shr_parser::SHRSweepPoint| {
                            a.amplitude
                                .partial_cmp(&b.amplitude)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        };
                    let point = if parsing_type == SHRParsingType::Peak {
                        points.iter().max_by(compare).unwrap()
                    } else {
                        points.iter().min_by(compare).unwrap()
                    };
                    (point.frequency_hz / 1.0e6, point.amplitude as f64)
                }
                SHRParsingType::Mean => (
                    (first_bin_hz + bin_size_hz * sweep_length as f64 / 2.0) / 1.0e6,
                    (points.iter().map(|point| point.amplitude).sum::<f32>() / sweep_length as f32)
                        as f64,
                ),
            };
            assert_eq!(sweep.frequency, expected_frequency);
            assert_eq!(sweep.amplitude, expected_amplitude);
        }
    }
}

#[test]
fn sweep_details_handle_empty_and_single_bin_sweeps() {
    let bytes = fs::read(fixture_path()).unwrap();
    let data_offset = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    for amplitudes in [vec![], vec![-42.5f32]] {
        let mut modified = bytes[..data_offset + 48].to_vec();
        modified[16..20].copy_from_slice(&1u32.to_le_bytes());
        modified[20..24].copy_from_slice(&(amplitudes.len() as u32).to_le_bytes());
        modified[24..32].copy_from_slice(&123.5f64.to_le_bytes());
        modified[32..40].copy_from_slice(&0.25f64.to_le_bytes());
        for amplitude in &amplitudes {
            modified.extend_from_slice(&amplitude.to_le_bytes());
        }
        let file = TemporaryFile::new(&modified);
        let parser = SHRParser::new(file.0.clone(), SHRParsingType::Peak).unwrap();
        let sweeps = parser.get_sweeps();
        assert_eq!(sweeps.len(), 1);
        assert_eq!(sweeps[0].get_data().len(), amplitudes.len());
        if let Some(point) = sweeps[0].get_data().first() {
            assert_eq!(point.frequency_hz, 123.5);
            assert_eq!(point.amplitude, -42.5);
        }
    }
}

#[test]
fn sweeps_return_empty_for_shr_file_with_no_sweeps() {
    let mut bytes = fs::read(fixture_path()).unwrap();
    let data_offset = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    bytes.truncate(data_offset);
    bytes[16..20].copy_from_slice(&0u32.to_le_bytes());
    let file = TemporaryFile::new(&bytes);
    let parser = SHRParser::new(file.0.clone(), SHRParsingType::Peak).unwrap();
    assert!(parser.get_sweeps().is_empty());
}

#[test]
fn sweeps_return_error_for_invalid_shr_file() {
    let mut bytes = fs::read(fixture_path()).unwrap();
    bytes[0..2].copy_from_slice(&0u16.to_le_bytes());
    let file = TemporaryFile::new(&bytes);
    assert!(matches!(
        SHRParser::new(file.0.clone(), SHRParsingType::Peak),
        Err(SHRFileError::InvalidFile)
    ));
}

#[test]
fn parser_rejects_truncated_header() {
    let bytes = fs::read(fixture_path()).unwrap();
    let file = TemporaryFile::new(&bytes[..20]);
    match SHRParser::new(file.0.clone(), SHRParsingType::Peak) {
        Err(SHRFileError::IOError(error)) => assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof),
        result => panic!("Expected an unexpected EOF error, got {result:?}"),
    }
}

#[test]
fn sweeps_return_error_for_nonexistent_shr_file() {
    let path = fixture_path().with_file_name("nonexistent.shr");
    match SHRParser::new(path, SHRParsingType::Peak) {
        Err(SHRFileError::IOError(error)) => assert_eq!(error.kind(), io::ErrorKind::NotFound),
        result => panic!("Expected a missing file error, got {result:?}"),
    }
}

#[test]
fn parser_exports_summary_csv() {
    let parser = SHRParser::new(fixture_path(), SHRParsingType::Peak).unwrap();
    let output = TemporaryFile::new(&[]);
    parser.to_csv(output.0.clone()).unwrap();
    let csv = fs::read_to_string(&output.0).unwrap();
    assert_eq!(csv, parser.to_str());
    let (_, rows) = csv
        .split_once("Sweep,Timestamp,Peak Freq MHz,Peak Ampl dBM,Latitude,Longitude\n")
        .unwrap();
    assert_eq!(rows.lines().count(), parser.shr_file.sweeps.len());
}

fn single_sweep_bytes(amplitudes: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0; 472 + 48];
    bytes[..2].copy_from_slice(&0xAA10u16.to_le_bytes());
    bytes[2..4].copy_from_slice(&1u16.to_le_bytes());
    bytes[8..16].copy_from_slice(&472u64.to_le_bytes());
    bytes[16..20].copy_from_slice(&1u32.to_le_bytes());
    bytes[20..24].copy_from_slice(&(amplitudes.len() as u32).to_le_bytes());
    bytes[24..32].copy_from_slice(&1_000_000f64.to_le_bytes());
    bytes[32..40].copy_from_slice(&1_000f64.to_le_bytes());
    for amplitude in amplitudes {
        bytes.extend_from_slice(&amplitude.to_le_bytes());
    }
    bytes
}

#[test]
fn parser_rejects_sweep_offsets_inside_the_file_header() {
    for offset in [0u64, 1, 464, 471] {
        let mut bytes = single_sweep_bytes(&[-42.5]);
        bytes[8..16].copy_from_slice(&offset.to_le_bytes());
        let file = TemporaryFile::new(&bytes);
        assert!(
            matches!(
                SHRParser::new(file.0.clone(), SHRParsingType::Peak),
                Err(SHRFileError::InvalidFile)
            ),
            "accepted overlapping offset {offset}"
        );
    }
}

#[test]
fn parser_accepts_sweep_offsets_at_or_after_the_file_header() {
    for padding in [0usize, 16] {
        let mut bytes = single_sweep_bytes(&[-42.5]);
        bytes.splice(472..472, std::iter::repeat_n(0, padding));
        bytes[8..16].copy_from_slice(&(472u64 + padding as u64).to_le_bytes());
        let file = TemporaryFile::new(&bytes);
        let parser = SHRParser::new(file.0.clone(), SHRParsingType::Peak).unwrap();
        assert_eq!(parser.shr_file.sweeps[0].amplitude, -42.5);
    }
}

#[test]
fn csv_filename_stays_in_one_non_formula_field() {
    let file = TemporaryFile::new(&single_sweep_bytes(&[-42.5]));
    let mut parser = SHRParser::new(file.0.clone(), SHRParsingType::Peak).unwrap();
    let original_line_count = parser.to_str().lines().count();
    for (name, expected) in [
        ("recording.shr", "#File name: recording.shr"),
        ("recording,=1+1.shr", "\"#File name: recording,=1+1.shr\""),
        ("recording\".shr", "\"#File name: recording\"\".shr\""),
        (
            "recording\r\n=1+1,attacker_cell\n\"quoted\".shr",
            "\"#File name: recording\\r\\n=1+1,attacker_cell\\n\"\"quoted\"\".shr\"",
        ),
    ] {
        parser.file_path = PathBuf::from(name);
        let csv = parser.to_str();
        assert_eq!(csv.lines().count(), original_line_count);
        assert_eq!(csv.lines().nth(2), Some(expected));
        let output = TemporaryFile::new(&[]);
        parser.to_csv(output.0.clone()).unwrap();
        assert_eq!(fs::read_to_string(&output.0).unwrap(), csv);
    }
}

#[test]
fn summaries_ignore_nan_without_changing_raw_bins() {
    for amplitudes in [
        vec![100.0, f32::NAN, -99.0],
        vec![f32::NAN, 100.0, -99.0],
        vec![100.0, -99.0, f32::NAN],
    ] {
        let file = TemporaryFile::new(&single_sweep_bytes(&amplitudes));
        for (parsing_type, expected) in [
            (SHRParsingType::Peak, 100.0),
            (SHRParsingType::Mean, 0.5),
            (SHRParsingType::Low, -99.0),
        ] {
            let parser = SHRParser::new(file.0.clone(), parsing_type).unwrap();
            let sweep = &parser.shr_file.sweeps[0];
            assert_eq!(sweep.amplitude, expected);
            if parsing_type != SHRParsingType::Mean {
                let index = amplitudes
                    .iter()
                    .position(|&value| value as f64 == expected)
                    .unwrap();
                assert_eq!(sweep.frequency, 1.0 + index as f64 * 0.001);
            }
            for (point, raw) in sweep.data.iter().zip(&amplitudes) {
                assert_eq!(point.amplitude.to_bits(), raw.to_bits());
            }
        }
    }
}

#[test]
fn summaries_mark_all_nan_sweeps_as_unknown() {
    let file = TemporaryFile::new(&single_sweep_bytes(&[f32::NAN, f32::NAN]));
    for parsing_type in [
        SHRParsingType::Peak,
        SHRParsingType::Mean,
        SHRParsingType::Low,
    ] {
        let parser = SHRParser::new(file.0.clone(), parsing_type).unwrap();
        let sweep = &parser.shr_file.sweeps[0];
        assert!(sweep.amplitude.is_nan());
        assert!(sweep.frequency.is_nan());
    }
}

#[test]
fn parser_survives_concurrent_truncation() {
    const CHILD_ENV: &str = "SHR_PARSER_TRUNCATION_TEST_CHILD";
    if std::env::var_os(CHILD_ENV).is_none() {
        // A memory-map regression must fail this child, rather than kill the test runner.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "parser_survives_concurrent_truncation",
                "--nocapture",
            ])
            .env(CHILD_ENV, "1")
            .output()
            .unwrap();
        assert!(output.status.success(), "child failed: {output:?}");
        return;
    }

    let mut bytes = single_sweep_bytes(&[]);
    let bins = 4_194_304u32;
    bytes[20..24].copy_from_slice(&bins.to_le_bytes());
    bytes.resize(472 + 48 + bins as usize * 4, 0);
    let file = TemporaryFile::new(&bytes);
    let path = file.0.clone();
    let truncater = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(30));
        fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_len(0)
            .unwrap();
    });
    let result = SHRParser::new(file.0.clone(), SHRParsingType::Peak);
    truncater.join().unwrap();
    match result {
        Ok(parser) => assert_eq!(parser.shr_file.sweeps[0].data.len(), bins as usize),
        Err(SHRFileError::IOError(error)) => assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof),
        result => panic!("unexpected result: {result:?}"),
    }
}
