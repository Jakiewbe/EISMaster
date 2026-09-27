# Feature Matrix

Each feature must be preserved unless explicitly removed later.

| Area | Current Feature | Rewrite Target | Validation |
| --- | --- | --- | --- |
| Import | CHI TXT parsing | Rust parser | Same frequency, Zreal, Zimag arrays as Python |
| Plot | Nyquist plot | Frontend chart | Same x/y data arrays |
| Plot | Bode magnitude/phase | Frontend chart | Same x/y data arrays |
| Quality | Basic EIS quality checks | Rust implementation | Same status and issue list on fixtures |
| Fitting | Equivalent circuit fitting | Rust fitting engine | Same parameter names and close numeric values |
| Fitting | Batch fitting | Rust batch runner | Same summary table shape and labels |
| Fitting | DRT-guided initial guess | Rust or staged migration | Same initial parameter guidance where feasible |
| Export | Raw plot matrix | Rust export | Same sheet/file layout |
| Export | Rs/Rct table | Rust export | Same columns |
| Export | Fit overlay matrix | Rust export | Same sheet/file layout |
| Export | Fit report | Rust export | Same sections and keys |
| MATLAB DRT | Stage input TXT files | Rust staging | Same file names and numeric columns |
| MATLAB DRT | Call DRTtools via MATLAB | Rust process runner | Same command semantics |
| MATLAB DRT | Include bundled DRTtools | Tauri bundle resources | `DRTtools.m` present in packaged app |
| DRT Export | `drt_line_plot` sheet | Rust xlsx export | Supports `logtau/gamma` and `tau/s/gamma` |
| DRT Export | `drt_cloud_density` sheet | Rust xlsx export | Same `logtau + gamma_tau...` matrix |
| DRT Export | `drt_quant_area` sheet | Rust xlsx export | Same configurable logtau-bin trend table |
| UI | MATLAB DRT config | Tauri UI | Same fields and defaults |
| UI | Custom logtau integration bins | Tauri UI | Same comma-separated parsing |

