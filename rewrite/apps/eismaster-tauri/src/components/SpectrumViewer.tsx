import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { SpectrumData, InspectResult, FitOutcome } from "../types";
import { NyquistPlot } from "./NyquistPlot";
import { BodePlot } from "./BodePlot";

export function SpectrumViewer({
  spectrum,
  showPlot,
  fitResult,
}: {
  spectrum: SpectrumData | null;
  showPlot: boolean;
  fitResult: FitOutcome | null;
}) {
  const [inspectResult, setInspectResult] = useState<InspectResult | null>(null);
  const [tableCollapsed, setTableCollapsed] = useState(false);

  useEffect(() => {
    if (!spectrum) { setInspectResult(null); return; }
    let cancelled = false;
    invoke<InspectResult>("inspect_spectrum", { path: spectrum.metadata.file_path })
      .then((r) => { if (!cancelled) setInspectResult(r); })
      .catch(() => { if (!cancelled) setInspectResult(null); });
    return () => { cancelled = true; };
  }, [spectrum?.metadata.file_path]);

  if (!spectrum) {
    return (
      <div className="panel panel-center" role="region" aria-label="Spectrum viewer empty state">
        <div className="empty-hint">
          <span className="empty-hint-icon" role="img" aria-label="No selection">📈</span>
          <span>No spectrum selected</span>
          <span style={{ fontSize: "10px", marginTop: "4px", color: "var(--text-muted)" }}>
            Select an item from the sidebar to visualize plot and data.
          </span>
        </div>
      </div>
    );
  }

  const fileName = spectrum.metadata.file_path.split("\\").pop()?.split("/").pop() ?? spectrum.metadata.file_path;
  const quality = inspectResult?.quality;

  return (
    <div className="panel panel-center" role="region" aria-label={`Spectrum: ${fileName}`}
      style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* Header */}
      <div className="data-header" style={{ flexShrink: 0 }}>
        <strong>{fileName}</strong>
        <span className="data-info">
          {spectrum.metadata.source_format.toUpperCase()} | {spectrum.metadata.technique || "EIS"} | {spectrum.freq_hz.length} pt
          {spectrum.metadata.instrument_model && spectrum.metadata.instrument_model !== "Unknown" && <> | {spectrum.metadata.instrument_model}</>}
          {quality && <span className={`quality-badge quality-${quality.status}`} style={{ marginLeft: "8px" }}>{quality.status}</span>}
        </span>
      </div>

      {/* Quality issues banner */}
      {quality && quality.issues.length > 0 && (
        <div style={{ flexShrink: 0, padding: "2px 12px", fontSize: "10px", color: "var(--text-muted)", background: "var(--bg-surface-subtle)", borderBottom: "1px solid var(--border-color)" }}>
          {quality.issues.map((issue, i) => (
            <span key={i} style={{ marginRight: "12px" }}>[{issue.severity}] {issue.message}</span>
          ))}
        </div>
      )}

      {/* Charts area — fills remaining space */}
      {showPlot && (
        <div style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 280px", gridTemplateRows: "1fr 1fr", gap: "6px", padding: "6px", minHeight: 0, overflow: "hidden" }}>
          {/* Nyquist: main chart, spans both rows */}
          <div style={{ gridRow: "1 / 3", gridColumn: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
            <div style={{ fontSize: "10px", color: "var(--text-muted)", padding: "0 4px 2px", flexShrink: 0 }}>Nyquist 图</div>
            <div style={{ flex: 1, minHeight: 0 }}>
              <NyquistPlot spectrum={spectrum} fitReal={fitResult?.predicted_real_ohm ?? null} fitImag={fitResult?.predicted_imag_ohm ?? null} />
            </div>
          </div>
          {/* Bode magnitude */}
          <div style={{ gridRow: 1, gridColumn: 2, minHeight: 0, display: "flex", flexDirection: "column" }}>
            <div style={{ fontSize: "10px", color: "var(--text-muted)", padding: "0 4px 2px", flexShrink: 0 }}>Bode 幅频图</div>
            <div style={{ flex: 1, minHeight: 0 }}>
              <BodePlot freq={spectrum.freq_hz} values={spectrum.z_mod_ohm} yLabel="|Z| (Ω)" logY={true} color="#81D4FA" />
            </div>
          </div>
          {/* Bode phase */}
          <div style={{ gridRow: 2, gridColumn: 2, minHeight: 0, display: "flex", flexDirection: "column" }}>
            <div style={{ fontSize: "10px", color: "var(--text-muted)", padding: "0 4px 2px", flexShrink: 0 }}>Bode 相位图</div>
            <div style={{ flex: 1, minHeight: 0 }}>
              <BodePlot freq={spectrum.freq_hz} values={spectrum.phase_deg} yLabel="Phase (°)" logY={false} color="#81D4FA" />
            </div>
          </div>
        </div>
      )}

      {/* Data table — collapsible, fixed bottom area */}
      <div style={{ flexShrink: 0, borderTop: "1px solid var(--border-color)" }}>
        <div
          onClick={() => setTableCollapsed(!tableCollapsed)}
          style={{ cursor: "pointer", padding: "4px 12px", fontSize: "11px", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: "6px", userSelect: "none" }}
        >
          <span style={{ fontSize: "10px" }}>{tableCollapsed ? "▶" : "▼"}</span>
          数据表格 ({spectrum.freq_hz.length} 行)
        </div>
        {!tableCollapsed && (
          <div className="table-scroll" style={{ maxHeight: "160px", overflowY: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">#</th>
                  <th scope="col">Freq (Hz)</th>
                  <th scope="col">Z' (Ω)</th>
                  <th scope="col">-Z'' (Ω)</th>
                  <th scope="col">|Z| (Ω)</th>
                  <th scope="col">Phase (°)</th>
                </tr>
              </thead>
              <tbody>
                {spectrum.freq_hz.map((f, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>{f.toExponential(4)}</td>
                    <td>{spectrum.z_real_ohm[i].toFixed(4)}</td>
                    <td>{(-spectrum.z_imag_ohm[i]).toFixed(4)}</td>
                    <td>{spectrum.z_mod_ohm[i].toFixed(4)}</td>
                    <td>{spectrum.phase_deg[i].toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
