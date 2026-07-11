/** Types matching the Rust eismaster-core models (serde Serialize). */

export interface SpectrumMetadata {
  file_path: string;
  technique: string;
  instrument_model: string;
  acquired_at: string | null;
  note: string;
  header: Record<string, string>;
  source_format: string;
}

export interface SpectrumData {
  metadata: SpectrumMetadata;
  freq_hz: number[];
  z_real_ohm: number[];
  z_imag_ohm: number[];
  z_mod_ohm: number[];
  phase_deg: number[];
}

export interface DrtExportRequest {
  spectra_dir: string;
  drt_dir: string;
  output_path: string;
  line_x: string;
  logtau_breaks: number[];
}

export interface MatlabDrtRequest {
  spectra_paths: string[];
  output_dir: string;
  matlab_exe: string;
  drttools_dir: string;
  matlab_bridge_dir: string;
  method: string;
  drt_type: number;
  lambda_value: number;
  coeff_value: number;
  inductance_mode: number;
  derivative_order: string;
  data_used: string;
  shape_control: string;
  line_x_axis: string;
  logtau_breaks: number[] | null;
}

export interface MatlabDrtResult {
  command: string;
  return_code: number | null;
  stdout: string;
  stderr: string;
  staging_dir: string;
  output_dir: string;
  output_files: string[];
  workbook_path: string | null;
}

export interface DevResourcePaths {
  matlab_bridge_dir: string | null;
  drttools_dir: string | null;
}

export interface QualityIssue {
  severity: string;
  message: string;
}

export interface QualityReport {
  status: string;
  issues: QualityIssue[];
  kk_status: string;
  kk_message: string;
}

export interface SegmentDetection {
  requested_mode: string;
  resolved_mode: string;
  peak_indices: number[];
  split_indices: number[];
}

export interface TemplateInfo {
  key: string;
  label: string;
  parameter_names: string[];
}

export interface InspectResult {
  file: string;
  n_points: number;
  quality: QualityReport;
  segmentation: SegmentDetection;
  circuit_templates: TemplateInfo[];
}

export interface FitOutcome {
  model_key: string;
  model_label: string;
  status: string;
  message: string;
  parameters: Record<string, number>;
  statistics: Record<string, number>;
  predicted_real_ohm: number[] | null;
  predicted_imag_ohm: number[] | null;
  masked_points: number;
  preprocess_actions: string[];
  fallback_from: string | null;
  diagnosis_type: string;
  diagnosis_severity: string;
  diagnosis_explanation: string;
  diagnosis_suggestions: string[];
  confidence_intervals: Record<string, [number, number]>;
  correlation_matrix_max: number;
}

export interface BatchItemResult {
  file: string;
  label: string;
  quality: QualityReport | null;
  fit: FitOutcome | null;
  error: string | null;
}

export interface BatchSummary {
  model_key: string;
  model_label: string;
  items: BatchItemResult[];
  n_total: number;
  n_ok: number;
  n_warn: number;
  n_failed: number;
}
