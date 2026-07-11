import type { SpectrumData } from "../types";

export function SpectrumList({
  spectra,
  selectedIdx,
  onSelect,
}: {
  spectra: SpectrumData[];
  selectedIdx: number | null;
  onSelect: (idx: number) => void;
}) {
  const formatFreq = (hz: number) => {
    if (hz >= 1e6) return `${(hz / 1e6).toFixed(1)} MHz`;
    if (hz >= 1e3) return `${(hz / 1e3).toFixed(1)} kHz`;
    if (hz >= 1) return `${hz.toFixed(1)} Hz`;
    if (hz > 0) return `${(hz * 1000).toFixed(0)} mHz`;
    return `${hz} Hz`;
  };

  return (
    <div className="panel panel-left" role="navigation" aria-label="Spectra browser">
      <h3>Spectra ({spectra.length})</h3>
      <div className="spectrum-list">
        {spectra.map((s, i) => {
          const fileName = s.metadata.file_path.split("\\").pop()?.split("/").pop() ?? "Unknown";
          const minFreq = s.freq_hz.length > 0 ? Math.min(...s.freq_hz) : 0;
          const maxFreq = s.freq_hz.length > 0 ? Math.max(...s.freq_hz) : 0;
          const freqRangeStr = s.freq_hz.length > 0 ? `${formatFreq(minFreq)} – ${formatFreq(maxFreq)}` : "";

          return (
            <div
              key={i}
              className={`spectrum-item ${i === selectedIdx ? "selected" : ""}`}
              onClick={() => onSelect(i)}
              role="button"
              tabIndex={0}
              aria-pressed={i === selectedIdx}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  onSelect(i);
                }
              }}
            >
              <div className="spectrum-name" title={s.metadata.file_path}>
                {fileName}
              </div>
              <div className="spectrum-meta">
                <span>{s.freq_hz.length} pts</span>
                {freqRangeStr && <span style={{ opacity: 0.8 }} title="Frequency range">{freqRangeStr}</span>}
              </div>
            </div>
          );
        })}
        {spectra.length === 0 && (
          <div className="empty-hint">
            <span className="empty-hint-icon" role="img" aria-label="No data">📊</span>
            <span>No spectra loaded</span>
            <span style={{ fontSize: "10px", marginTop: "4px", color: "var(--text-muted)" }}>
              Import file(s) or folder to begin.
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
