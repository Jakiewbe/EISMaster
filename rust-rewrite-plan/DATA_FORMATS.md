# Data Formats to Preserve

## DRT Workbook

The MATLAB DRT export workbook is named `drt_matrix.xlsx`.

Required sheets:

### `drt_line_plot`

For `logtau / gamma`:

| row | meaning |
| --- | --- |
| 1 | sample labels |
| 2 | `logtau`, then repeated `gamma_tau` |
| 3+ | x values and gamma values |

For `tau/s / gamma`:

| row | meaning |
| --- | --- |
| 1 | sample labels |
| 2 | `tau/s`, then repeated `gamma_tau` |
| 3+ | tau values in seconds and gamma values |

### `drt_cloud_density`

Matrix for display heatmaps:

- first column: `logtau`
- sample columns: `gamma_tau`

### `drt_quant_area`

Trend table for quantitative contribution plots:

- `sample`
- `time_min`
- `total_area`
- one column per user-defined logtau region

Default logtau breaks:

```text
-3, 0
```

Example custom breaks:

```text
1, 3
```

This produces:

```text
logtau_lt_1
logtau_1_to_3
logtau_ge_3
```

## MATLAB DRT Files

Input staging files are plain TXT with numeric columns expected by DRTtools.

Output files are named:

```text
<input_stem>_DRT.txt
```

The parser must support:

- continuous DRT exports with `tau` or `freq` headers
- peak-fit exports with peak height, position, and width

