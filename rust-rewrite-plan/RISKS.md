# Risks and Decisions

## Highest Risk Areas

1. Equivalent-circuit fitting parity
2. DRT-guided initial guesses
3. MATLAB path handling in packaged builds
4. Excel layout compatibility
5. Chinese and long Windows paths

## Decisions

### Do Not Rewrite Everything at Once

Start with CLI-level parity. GUI comes after data and export correctness.

### Keep MATLAB DRT External

Do not rewrite DRTtools in Rust initially. Preserve MATLAB integration first.

### Keep Python as Oracle

The Python version remains the source of truth until the Rust version passes fixture comparisons.

## Size Expectations

Rough target after Rust/Tauri rewrite:

- without MATLAB runtime: tens of MB
- bundled DRTtools: about 1-2 MB extra
- no Python/scipy/PySide6 stack

