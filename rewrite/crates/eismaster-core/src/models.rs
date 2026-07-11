use chrono::NaiveDateTime;
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;

/// Metadata about an imported EIS spectrum file.
#[derive(Debug, Clone, Serialize)]
pub struct SpectrumMetadata {
    /// Original file path (display only; not used for logic).
    pub file_path: PathBuf,
    /// Technique label, e.g. "A.C. Impedance".
    pub technique: String,
    /// Instrument model string, e.g. "CHI660F".
    pub instrument_model: String,
    /// Acquisition timestamp if available.
    pub acquired_at: Option<NaiveDateTime>,
    /// Free-form note from the file header.
    pub note: String,
    /// All key-value header fields from the file.
    pub header: HashMap<String, String>,
    /// Source format identifier: "txt", "csv", "bin".
    pub source_format: String,
}

/// Numeric arrays for one EIS spectrum.
#[derive(Debug, Clone, Serialize)]
pub struct SpectrumData {
    pub metadata: SpectrumMetadata,
    /// Frequency in Hz.
    pub freq_hz: Vec<f64>,
    /// Real part of impedance in Ohm.
    pub z_real_ohm: Vec<f64>,
    /// Imaginary part of impedance in Ohm.
    pub z_imag_ohm: Vec<f64>,
    /// Impedance magnitude |Z| in Ohm.
    pub z_mod_ohm: Vec<f64>,
    /// Phase in degrees.
    pub phase_deg: Vec<f64>,
}

impl SpectrumData {
    pub fn n_points(&self) -> usize {
        self.freq_hz.len()
    }

    pub fn display_name(&self) -> &str {
        self.metadata
            .file_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
    }
}
