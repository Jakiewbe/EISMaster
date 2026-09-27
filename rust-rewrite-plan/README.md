# EISMaster Rewrite Plan

This folder records the plan for a full rewrite while keeping the current Python/PySide6 version as the reference implementation.

## Goal

Rewrite EISMaster into a smaller, maintainable desktop application while preserving all current user-facing features and output formats.

## Current Reference Version

Use the current Python application as the behavior oracle:

- Source: repository root Python implementation
- Latest packaged reference: `dist_rebuild4/EISMaster/EISMaster.exe`
- Important packaged fix: `matlab-DRTtools-local` must be included in release builds

Do not change behavior during migration unless a difference is explicitly documented and approved.

## Proposed Target Stack

- Desktop shell: Tauri
- UI: TypeScript frontend
- Core engine: Rust library
- MATLAB DRT integration: Rust process runner calling MATLAB `-batch`
- Excel export: Rust xlsx writer
- Plotting: frontend charting for interactive plots, Rust for data preparation

Rust is preferred over Go because EIS fitting, numerical code, typed data models, and native desktop packaging are better aligned with the project goals.

## Migration Rule

Every new module must be validated against the current Python output on fixed sample files before replacing the old implementation.

