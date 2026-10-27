use std::cmp::Ordering;

use crate::types::{SHRParsingType, SHRSweep, SHRSweepHeader, SHRSweepPoint};

impl SHRSweep {
    /// Constructs a new `SHRSweep` instance.
    ///
    /// NaN amplitudes are retained in `data` but excluded from summary metrics.
    /// A nonempty sweep containing only NaN amplitudes has NaN summary values.
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
        let data = sweep_data_raw
            .into_iter()
            .enumerate()
            .map(|(index, amplitude)| SHRSweepPoint {
                frequency_hz: first_bin_freq_hz + index as f64 * bin_size_hz,
                amplitude,
            })
            .collect();

        Self::from_points(
            sweep_number,
            sweep_header,
            data,
            parsing_type,
            first_bin_freq_hz,
            bin_size_hz,
        )
    }

    pub(crate) fn from_points(
        number: i32,
        header: SHRSweepHeader,
        data: Vec<SHRSweepPoint>,
        parsing_type: SHRParsingType,
        first_bin_freq_hz: f64,
        bin_size_hz: f64,
    ) -> Self {
        let (frequency, amplitude) =
            Self::calculate_sweep_metrics(&data, parsing_type, first_bin_freq_hz, bin_size_hz);

        Self {
            number,
            frequency,
            amplitude,
            header,
            data,
        }
    }

    /// Borrows all frequency/amplitude pairs without cloning the sweep.
    /// Frequencies are in Hz; the summary `frequency` field remains in MHz.
    pub fn get_data(&self) -> &[SHRSweepPoint] {
        &self.data
    }

    pub fn get_header(&self) -> &SHRSweepHeader {
        &self.header
    }

    fn calculate_sweep_metrics(
        data: &[SHRSweepPoint],
        parsing_type: SHRParsingType,
        first_bin_freq_hz: f64,
        bin_size_hz: f64,
    ) -> (f64, f64) {
        let mut samples = data
            .iter()
            .enumerate()
            .filter(|(_, point)| !point.amplitude.is_nan())
            .peekable();
        if !data.is_empty() && samples.peek().is_none() {
            return (f64::NAN, f64::NAN);
        }

        match parsing_type {
            SHRParsingType::Peak => {
                // max_by retains the last equal value.
                let (index, power) = samples
                    .max_by(|(_, a), (_, b)| {
                        a.amplitude
                            .partial_cmp(&b.amplitude)
                            .unwrap_or(Ordering::Equal)
                    })
                    .map(|(index, point)| (index, point.amplitude))
                    .unwrap_or((0, 0.0));

                let frequency = (first_bin_freq_hz + index as f64 * bin_size_hz) / 1.0e6;
                (frequency, power as f64)
            }
            SHRParsingType::Mean => {
                // Accumulation and division stay in f32, in file order.
                let (sum, count) = samples.fold((0.0f32, 0usize), |(sum, count), (_, point)| {
                    (sum + point.amplitude, count + 1)
                });
                let power = if count != 0 { sum / count as f32 } else { 0.0 };

                let frequency = (first_bin_freq_hz + bin_size_hz * data.len() as f64 / 2.0) / 1.0e6;
                (frequency, power as f64)
            }
            SHRParsingType::Low => {
                // min_by retains the first equal value, unlike max_by above.
                let (index, power) = samples
                    .min_by(|(_, a), (_, b)| {
                        a.amplitude
                            .partial_cmp(&b.amplitude)
                            .unwrap_or(Ordering::Equal)
                    })
                    .map(|(index, point)| (index, point.amplitude))
                    .unwrap_or((0, 0.0));

                let frequency = (first_bin_freq_hz + index as f64 * bin_size_hz) / 1.0e6;
                (frequency, power as f64)
            }
        }
    }
}
