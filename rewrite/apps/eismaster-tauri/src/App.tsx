import { useState, useCallback } from "react";
import type { SpectrumData, FitOutcome, BatchSummary } from "./types";
import { Toolbar } from "./components/Toolbar";
import { SpectrumList } from "./components/SpectrumList";
import { SpectrumViewer } from "./components/SpectrumViewer";
import { RightPanel } from "./components/RightPanel";

export default function App() {
  const [spectra, setSpectra] = useState<SpectrumData[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [statusMsg, setStatusMsg] = useState("就绪");
  const [showPlot, setShowPlot] = useState(true);
  const [isOperationRunning, setIsOperationRunning] = useState(false);
  // Per-spectrum fit state — keyed by file_path
  const [fitResultsByPath, setFitResultsByPath] = useState<Record<string, FitOutcome>>({});
  // Per-spectrum inspect state
  const [batchSummary, setBatchSummary] = useState<BatchSummary | null>(null);
  // Force remount of child tabs on clear
  const [clearKey, setClearKey] = useState(0);

  const selected = selectedIdx !== null ? spectra[selectedIdx] : null;
  // Derive current fit from per-spectrum map — null if no fit for current spectrum
  const currentFit = selected ? (fitResultsByPath[selected.metadata.file_path] ?? null) : null;

  const handleImported = useCallback((newSpectra: SpectrumData[], firstNewIdx: number) => {
    setSpectra((prev) => [...prev, ...newSpectra]);
    setSelectedIdx(firstNewIdx);
  }, []);

  const handleClearSpectra = useCallback(() => {
    setSpectra([]);
    setSelectedIdx(null);
    setFitResultsByPath({});
    setBatchSummary(null);
    setStatusMsg("已清空");
    setClearKey((k) => k + 1);
  }, []);

  // Called by RightPanel when a fit completes — stores fit per spectrum path
  const handleFitResult = useCallback((spectrumPath: string, fit: FitOutcome | null) => {
    if (fit) {
      setFitResultsByPath((prev) => ({ ...prev, [spectrumPath]: fit }));
    } else {
      setFitResultsByPath((prev) => {
        const next = { ...prev };
        delete next[spectrumPath];
        return next;
      });
    }
  }, []);

  return (
    <div className="app">
      <Toolbar
        spectraCount={spectra.length}
        showPlot={showPlot}
        onShowPlotChange={setShowPlot}
        onImported={handleImported}
        onStatusChange={setStatusMsg}
        isOperationRunning={isOperationRunning}
        onIsOperationRunningChange={setIsOperationRunning}
        onClearSpectra={handleClearSpectra}
      />
      <div className="main-area">
        <SpectrumList spectra={spectra} selectedIdx={selectedIdx} onSelect={setSelectedIdx} />
        <SpectrumViewer
          spectrum={selected}
          showPlot={showPlot}
          fitResult={currentFit}
        />
        <RightPanel
          key={clearKey}
          spectra={spectra}
          selected={selected}
          currentFit={currentFit}
          onFitResult={handleFitResult}
          batchSummary={batchSummary}
          onBatchSummary={setBatchSummary}
          onStatusChange={setStatusMsg}
          isOperationRunning={isOperationRunning}
          onIsOperationRunningChange={setIsOperationRunning}
        />
      </div>
      <div className="status-bar">
        <span>{statusMsg}</span>
        <span style={{ fontSize: "10px", color: "var(--text-muted)" }}>
          {isOperationRunning ? "◉ 处理中" : "○ 就绪"}
          {spectra.length > 0 && ` · ${spectra.length} 谱图`}
          {selected && ` · ${selected.metadata.instrument_model || "?"} · ${selected.metadata.source_format.toUpperCase()}`}
        </span>
      </div>
    </div>
  );
}
