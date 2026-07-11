import { useState, useEffect, useRef, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import type {
  SpectrumData,
  FitOutcome,
  BatchSummary,
  DrtExportRequest,
  MatlabDrtRequest,
  MatlabDrtResult,
  DevResourcePaths,
} from "../types";
import { BatchFitPlot } from "./NyquistPlot";

type Tab = "analyze" | "batch" | "matlab" | "drt";

export function RightPanel({
  spectra, selected, currentFit, onFitResult,
  batchSummary, onBatchSummary, onStatusChange,
  isOperationRunning, onIsOperationRunningChange,
}: {
  spectra: SpectrumData[];
  selected: SpectrumData | null;
  currentFit: FitOutcome | null;
  onFitResult: (spectrumPath: string, fit: FitOutcome | null) => void;
  batchSummary: BatchSummary | null;
  onBatchSummary: (s: BatchSummary | null) => void;
  onStatusChange: (msg: string) => void;
  isOperationRunning: boolean;
  onIsOperationRunningChange: (v: boolean) => void;
}) {
  const [activeTab, setActiveTab] = useState<Tab>("analyze");
  return (
    <div className="panel panel-right">
      <div className="tab-bar" role="tablist" aria-label="Analysis tools">
        {(["analyze", "batch", "matlab", "drt"] as Tab[]).map((tab) => (
          <button key={tab} role="tab" aria-selected={activeTab === tab} aria-controls={`tabpanel-${tab}`} id={`tab-${tab}`}
            className={`tab-btn ${activeTab === tab ? "active" : ""}`} onClick={() => setActiveTab(tab)}>
            {tab === "analyze" ? "拟合" : tab === "batch" ? "批量拟合" : tab === "matlab" ? "MATLAB DRT" : "DRT 导出"}
          </button>
        ))}
      </div>
      <div className="tab-content" role="tabpanel" id={`tabpanel-${activeTab}`} aria-labelledby={`tab-${activeTab}`}>
        {activeTab === "analyze" && <AnalyzeTab selected={selected} currentFit={currentFit} onFitResult={onFitResult} onStatusChange={onStatusChange} isOperationRunning={isOperationRunning} onIsOperationRunningChange={onIsOperationRunningChange} />}
        {activeTab === "batch" && <BatchTab spectra={spectra} batchSummary={batchSummary} onBatchSummary={onBatchSummary} onStatusChange={onStatusChange} isOperationRunning={isOperationRunning} onIsOperationRunningChange={onIsOperationRunningChange} />}
        {activeTab === "matlab" && <MatlabTab spectra={spectra} onStatusChange={onStatusChange} isOperationRunning={isOperationRunning} onIsOperationRunningChange={onIsOperationRunningChange} batchMode={false} />}
        {activeTab === "drt" && <DrtExportTab onStatusChange={onStatusChange} isOperationRunning={isOperationRunning} onIsOperationRunningChange={onIsOperationRunningChange} />}
      </div>
    </div>
  );
}

// =========================================================================
// Analyze tab — per-spectrum fit
// =========================================================================

function AnalyzeTab({
  selected, currentFit, onFitResult, onStatusChange, isOperationRunning, onIsOperationRunningChange,
}: {
  selected: SpectrumData | null; currentFit: FitOutcome | null;
  onFitResult: (path: string, f: FitOutcome | null) => void;
  onStatusChange: (msg: string) => void; isOperationRunning: boolean; onIsOperationRunningChange: (v: boolean) => void;
}) {
  const [fitLoading, setFitLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState("zview_segmented_rq_rwo");

  const handleFit = async () => {
    if (!selected) return;
    setFitLoading(true); onIsOperationRunningChange(true);
    const path = selected.metadata.file_path;
    try {
      const result = await invoke<FitOutcome>("fit_spectrum", { path, modelKey: selectedModel });
      onFitResult(path, result);
      onStatusChange(`拟合${result.status === "ok" ? "完成" : result.status === "warn" ? "完成(警告)" : "失败"}`);
    } catch (err) { onStatusChange(`拟合失败: ${err}`); }
    finally { setFitLoading(false); onIsOperationRunningChange(false); }
  };

  const handleExport = async () => {
    if (!selected || !currentFit) return;
    const savePath = await open({ directory: false, multiple: false, filters: [{ name: "Excel", extensions: ["xlsx"] }] });
    if (!savePath) return;
    try {
      onIsOperationRunningChange(true);
      await invoke("export_single_fit", { outputPath: savePath, spectrumPath: selected.metadata.file_path, modelKey: currentFit.model_key });
      onStatusChange(`已导出: ${(savePath as string).split("\\").pop()}`);
    } catch (err) { onStatusChange(`导出失败: ${err}`); }
    finally { onIsOperationRunningChange(false); }
  };

  return (
    <>
      <h3>等效电路拟合</h3>
      <div className="form-section">
        <div className="form-group">
          <label htmlFor="model-select">拟合模型</label>
          <select id="model-select" value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} disabled={fitLoading || isOperationRunning}>
            <option value="zview_segmented_rq_rwo">Single-arc R(QRWo)</option>
            <option value="zview_double_rq_qrwo">Double-arc R(QR)(Q(RWo))</option>
          </select>
        </div>
        <div className="form-actions">
          <button className="run-btn" onClick={handleFit} disabled={!selected || fitLoading || isOperationRunning}>
            {fitLoading ? "拟合中..." : "开始拟合"}
          </button>
          <button className="export-btn" onClick={handleExport} disabled={!selected || !currentFit || isOperationRunning} style={{ marginLeft: "8px" }}>
            导出数据
          </button>
        </div>

        {!selected ? (
          <div className="empty-hint" style={{ padding: "16px" }}>选择谱图后进行拟合。</div>
        ) : !currentFit ? (
          <div className="empty-hint" style={{ padding: "16px" }}>当前谱图尚未拟合。</div>
        ) : (
          <FitResultDisplay fit={currentFit} selected={selected} />
        )}
      </div>
    </>
  );
}

// =========================================================================
// Shared fit result display
// =========================================================================

function FitResultDisplay({ fit, selected }: { fit: FitOutcome; selected: SpectrumData }) {
  return (
    <div className="inspect-results">
      <div className="inspect-section">
        <strong>拟合状态:</strong>{" "}
        <span className={`quality-badge quality-${fit.status === "ok" ? "pass" : fit.status}`}>
          {fit.status === "ok" ? "正常" : fit.status === "warn" ? "警告" : "失败"}
        </span>
        <div style={{ fontSize: "10px", marginTop: "4px", color: "var(--text-muted)" }}>{fit.model_label}</div>
      </div>

      {fit.masked_points > 0 && (
        <div className="inspect-section">
          <strong>预处理:</strong> 已排除 {fit.masked_points} 个点
          {fit.preprocess_actions.map((a, i) => <div key={i} className="issue-info" style={{ fontSize: "10px", marginTop: "2px" }}>• {a}</div>)}
        </div>
      )}

      <div className="inspect-section">
        <strong>拟合参数:</strong>
        <table className="param-table"><tbody>
          {Object.entries(fit.parameters).map(([key, val]) => (
            <tr key={key}><td style={{ fontWeight: ["Rs","Rct","Rsei"].includes(key) ? 600 : "normal" }}>{key}</td><td style={{ fontFamily: "var(--font-mono)" }}>{Number(val).toPrecision(6)}</td></tr>
          ))}
        </tbody></table>
      </div>

      <div className="inspect-section">
        <strong>拟合统计:</strong>
        <table className="param-table"><tbody>
          {[["chi2_reduced","Chi² reduced"],["rss","RSS"],["aic","AIC"],["aicc","AICc"],["bic","BIC"]].map(([key,label]) =>
            fit.statistics[key] !== undefined ? <tr key={key}><td>{label}</td><td style={{ fontFamily: "var(--font-mono)" }}>{Number(fit.statistics[key]) < 0.01 ? Number(fit.statistics[key]).toExponential(4) : Number(fit.statistics[key]).toPrecision(6)}</td></tr> : null
          )}
        </tbody></table>
      </div>

      {fit.diagnosis_type && fit.diagnosis_type !== "none" && (
        <div className="inspect-section">
          <strong>诊断:</strong>{" "}
          <span style={{ color: fit.diagnosis_severity === "error" ? "var(--danger-color)" : "var(--warning-color)" }}>{fit.diagnosis_explanation}</span>
          {fit.diagnosis_suggestions.length > 0 && <ul style={{ fontSize: "10px", marginTop: "4px", paddingLeft: "16px" }}>{fit.diagnosis_suggestions.map((s,i) => <li key={i}>{s}</li>)}</ul>}
        </div>
      )}

      {fit.predicted_real_ohm && fit.predicted_imag_ohm && (
        <div className="inspect-section">
          <strong>拟合残差:</strong>
          <ResidualPlot freq={selected.freq_hz} zRealExp={selected.z_real_ohm} zImagExp={selected.z_imag_ohm} zRealFit={fit.predicted_real_ohm} zImagFit={fit.predicted_imag_ohm} />
        </div>
      )}
    </div>
  );
}

// =========================================================================
// Residual plot
// =========================================================================

function ResidualPlot({ freq, zRealExp, zImagExp, zRealFit, zImagFit }: {
  freq: number[]; zRealExp: number[]; zImagExp: number[]; zRealFit: number[]; zImagFit: number[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dim, setDim] = useState({ w: 260, h: 100 });
  useEffect(() => {
    const c = containerRef.current; if (!c) return;
    const obs = new ResizeObserver((e) => { if (e[0]) setDim({ w: Math.max(e[0].contentRect.width, 200), h: 100 }); });
    obs.observe(c); return () => obs.disconnect();
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = dim.w * dpr; canvas.height = dim.h * dpr; ctx.scale(dpr, dpr);
    const pad = { t: 10, r: 10, b: 22, l: 34 };
    ctx.fillStyle = "#141416"; ctx.fillRect(0, 0, dim.w, dim.h);

    const n = Math.min(freq.length, zRealFit.length);
    if (n === 0) return;
    const res: number[] = [];
    for (let i = 0; i < n; i++) { const ze = Math.hypot(zRealExp[i], zImagExp[i]); res.push(ze > 1e-9 ? ((zRealFit[i] - ze) / ze) * 100 : 0); }
    const logF = freq.map((f) => (f > 0 ? Math.log10(f) : NaN));
    const pts = logF.map((lf, i) => ({ x: lf, y: res[i] })).filter((p) => isFinite(p.x) && isFinite(p.y));
    if (pts.length < 2) return;

    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yAbs = Math.max(...ys.map(Math.abs), 0.5);
    const pw = dim.w - pad.l - pad.r, ph = dim.h - pad.t - pad.b;
    const tx = (v: number) => pad.l + ((v - xMin) / (xMax - xMin || 1)) * pw;
    const ty = (v: number) => dim.h - pad.b - ((v + yAbs) / (2 * yAbs)) * ph;
    const y0 = ty(0);

    ctx.strokeStyle = "rgba(140,140,150,0.2)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, y0); ctx.lineTo(dim.w - pad.r, y0); ctx.stroke();

    const barW = Math.max(1, pw / pts.length - 1);
    for (const p of pts) {
      const x = tx(p.x);
      ctx.fillStyle = p.y > 0 ? "#f43f5e" : "#3b82f6";
      ctx.fillRect(x - barW / 2, Math.min(y0, ty(p.y)), barW, Math.abs(ty(p.y) - y0));
    }
    ctx.fillStyle = "#8c8c96"; ctx.font = "8px ui-monospace,monospace"; ctx.textAlign = "center"; ctx.fillText("Freq (Hz)", dim.w / 2, dim.h - 4);
  }, [freq, zRealExp, zImagExp, zRealFit, zImagFit, dim]);

  useEffect(() => { const raf = requestAnimationFrame(draw); return () => cancelAnimationFrame(raf); }, [draw]);
  return <div ref={containerRef} style={{ width: "100%", marginTop: "4px" }}><canvas ref={canvasRef} style={{ width: "100%", height: "100px", display: "block", borderRadius: "4px", border: "1px solid #27272b" }} /></div>;
}

// =========================================================================
// Batch tab — batch fit + trend + batch MATLAB DRT
// =========================================================================

function BatchTab({
  spectra, batchSummary, onBatchSummary, onStatusChange, isOperationRunning, onIsOperationRunningChange,
}: {
  spectra: SpectrumData[]; batchSummary: BatchSummary | null; onBatchSummary: (s: BatchSummary | null) => void;
  onStatusChange: (msg: string) => void; isOperationRunning: boolean; onIsOperationRunningChange: (v: boolean) => void;
}) {
  const [batchModel, setBatchModel] = useState("zview_segmented_rq_rwo");
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchSelectedIdx, setBatchSelectedIdx] = useState<number | null>(null);
  const [batchExportPath, setBatchExportPath] = useState("");
  const [showRs, setShowRs] = useState(true);
  const [showRsei, setShowRsei] = useState(true);
  const [showRct, setShowRct] = useState(true);

  // ----- MATLAB DRT batch state -----
  const [matlabExe, setMatlabExe] = useState("");
  const [drttoolsDir, setDrttoolsDir] = useState("");
  const [matlabBridgeDir, setMatlabBridgeDir] = useState("");
  const [matlabOutputDir, setMatlabOutputDir] = useState("");
  const [matlabMethod, setMatlabMethod] = useState("simple");
  const [matlabDrtType, setMatlabDrtType] = useState(2);
  const [matlabLambda, setMatlabLambda] = useState("0.001");
  const [matlabCoeff, setMatlabCoeff] = useState("0.5");
  const [matlabInductance, setMatlabInductance] = useState(1);
  const [matlabLineX, setMatlabLineX] = useState("logtau");
  const [matlabLogtauBreaks, setMatlabLogtauBreaks] = useState("-3,0");
  const [matlabRunning, setMatlabRunning] = useState(false);
  const [matlabLog, setMatlabLog] = useState("");

  useEffect(() => {
    invoke<DevResourcePaths>("get_dev_resource_paths").then((p) => {
      if (p.matlab_bridge_dir) setMatlabBridgeDir(p.matlab_bridge_dir);
      if (p.drttools_dir) setDrttoolsDir(p.drttools_dir);
    }).catch(() => {});
  }, []);

  const handleBatchFit = async () => {
    if (spectra.length === 0) { onStatusChange("请先导入谱图"); return; }
    setBatchLoading(true); onIsOperationRunningChange(true); onBatchSummary(null); setBatchSelectedIdx(null);
    try {
      const paths = spectra.map((s) => s.metadata.file_path);
      const result = await invoke<BatchSummary>("fit_batch_paths", { paths, modelKey: batchModel });
      onBatchSummary(result);
      onStatusChange(`批量拟合完成: ${result.n_total} 文件, 正常: ${result.n_ok}, 警告: ${result.n_warn}, 失败: ${result.n_failed}`);
    } catch (err) { onStatusChange(`批量拟合失败: ${err}`); }
    finally { setBatchLoading(false); onIsOperationRunningChange(false); }
  };

  const handleBatchExport = async () => {
    if (!batchExportPath) { onStatusChange("请设置导出路径"); return; }
    if (spectra.length === 0) { onStatusChange("请先导入谱图"); return; }
    onIsOperationRunningChange(true);
    try {
      const paths = spectra.map((s) => s.metadata.file_path);
      const result = await invoke<string>("export_batch_workbook_from_paths", { outputPath: batchExportPath, paths, modelKey: batchModel });
      onStatusChange(result);
    } catch (err) { onStatusChange(`导出失败: ${err}`); }
    finally { onIsOperationRunningChange(false); }
  };

  const pickDir = (s: (v: string) => void) => open({ directory: true, multiple: false }).then((p) => { if (p) s(p); });
  const pickFile = (s: (v: string) => void) => open({ multiple: false, filters: [{ name: "Excel", extensions: ["xlsx"] }] }).then((p) => { if (p) s(p); });
  const pickExe = (s: (v: string) => void) => open({ multiple: false, filters: [{ name: "Executable", extensions: ["exe"] }] }).then((p) => { if (p) s(p); });

  // ----- Run batch MATLAB DRT -----
  const handleBatchMatlabDrt = async () => {
    if (spectra.length === 0) { onStatusChange("请先导入谱图。"); return; }
    if (!matlabExe) { onStatusChange("请设置 MATLAB 可执行文件路径。"); return; }
    if (!drttoolsDir) { onStatusChange("请设置 DRTtools 目录。"); return; }
    if (!matlabBridgeDir) { onStatusChange("请设置 MATLAB bridge 目录。"); return; }
    if (!matlabOutputDir) { onStatusChange("请设置输出目录。"); return; }

    const spectraPaths = spectra.map((s) => s.metadata.file_path);
    const breaks = matlabLogtauBreaks.split(",").map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
    const req: MatlabDrtRequest = {
      spectra_paths: spectraPaths, output_dir: matlabOutputDir,
      matlab_exe: matlabExe, drttools_dir: drttoolsDir, matlab_bridge_dir: matlabBridgeDir,
      method: matlabMethod, drt_type: matlabDrtType,
      lambda_value: parseFloat(matlabLambda) || 0.001, coeff_value: parseFloat(matlabCoeff) || 0.5,
      inductance_mode: matlabInductance, derivative_order: "1st-order",
      data_used: "Combined Re-Im Data", shape_control: "FWHM Coefficient",
      line_x_axis: matlabLineX, logtau_breaks: breaks,
    };

    setMatlabRunning(true); onIsOperationRunningChange(true); setMatlabLog(""); onStatusChange("正在后台运行 MATLAB DRT...");
    try {
      const result = await invoke<MatlabDrtResult>("run_matlab_drt", { req });
      let log = `命令:\n${result.command}\n\n退出码: ${result.return_code ?? "null"}\n\n`;
      log += `--- stdout ---\n${result.stdout}\n\n--- stderr ---\n${result.stderr}\n\n`;
      log += `输出文件 (${result.output_files.length}):\n`;
      for (const f of result.output_files) log += `  ${f.split("\\").pop()?.split("/").pop()}\n`;
      if (result.workbook_path) log += `\nExcel Workbook: ${result.workbook_path}\n`;
      setMatlabLog(log);
      onStatusChange(result.return_code === 0 ? `MATLAB DRT 完成: ${result.output_files.length} 文件` : `MATLAB 执行失败，退出码 ${result.return_code}`);
    } catch (err) { setMatlabLog(`错误: ${err}`); onStatusChange(`MATLAB DRT 失败: ${err}`); }
    finally { setMatlabRunning(false); onIsOperationRunningChange(false); }
  };

  return (
    <>
      <h3>批量拟合</h3>
      <div className="form-section">
        <div className="form-group">
          <label htmlFor="batch-model">拟合模型</label>
          <select id="batch-model" value={batchModel} onChange={(e) => setBatchModel(e.target.value)} disabled={batchLoading || isOperationRunning}>
            <option value="zview_segmented_rq_rwo">Single-arc R(QRWo)</option>
            <option value="zview_double_rq_qrwo">Double-arc R(QR)(Q(RWo))</option>
          </select>
        </div>
        <div className="form-actions">
          <button className="run-btn" onClick={handleBatchFit} disabled={batchLoading || spectra.length === 0 || isOperationRunning}>
            {batchLoading ? "处理中..." : "批量拟合"}
          </button>
        </div>

        {batchSummary ? (
          <div className="inspect-results" style={{ marginTop: "14px" }}>
            <div className="inspect-section">
              <strong>批量结果摘要:</strong>
              <div style={{ fontSize: "10px", color: "var(--text-muted)", marginTop: "2px" }}>模型: {batchSummary.model_label}</div>
              <div style={{ marginTop: "4px", fontSize: "11px" }}>
                共 {batchSummary.n_total} 个:{" "}
                <span style={{ color: "var(--success-color)", fontWeight: 600 }}>{batchSummary.n_ok} 正常</span> |{" "}
                <span style={{ color: "var(--warning-color)", fontWeight: 600 }}>{batchSummary.n_warn} 警告</span> |{" "}
                <span style={{ color: "var(--danger-color)", fontWeight: 600 }}>{batchSummary.n_failed} 失败</span>
              </div>
            </div>

            {batchSummary.items.length > 1 && (
              <div className="inspect-section">
                <strong>参数趋势:</strong>
                <div style={{ display: "flex", gap: "12px", marginBottom: "4px" }}>
                  {["Rs","Rsei","Rct"].map((label) => (
                    <label key={label} style={{ fontSize: "10px", cursor: "pointer", display: "flex", alignItems: "center", gap: "4px" }}>
                      <input type="checkbox" checked={label === "Rs" ? showRs : label === "Rsei" ? showRsei : showRct}
                        onChange={(e) => { label === "Rs" ? setShowRs(e.target.checked) : label === "Rsei" ? setShowRsei(e.target.checked) : setShowRct(e.target.checked); }} />
                      {label}
                    </label>
                  ))}
                </div>
                <TrendPlot items={batchSummary.items} showRs={showRs} showRsei={showRsei} showRct={showRct} />
              </div>
            )}

            <div className="inspect-section" style={{ maxHeight: "220px", overflowY: "auto", overflowX: "auto", padding: "0" }}>
              <table className="data-table" style={{ fontSize: "10px", whiteSpace: "nowrap" }}>
                <thead><tr>
                  <th scope="col" style={{ textAlign: "center", width: "30px" }}>#</th>
                  <th scope="col" style={{ textAlign: "left", paddingLeft: "6px", minWidth: "70px" }}>样品</th>
                  <th scope="col" style={{ textAlign: "center", width: "40px" }}>状态</th>
                  <th scope="col" style={{ textAlign: "right", paddingRight: "6px", width: "55px" }}>Rs</th>
                  <th scope="col" style={{ textAlign: "right", paddingRight: "6px", width: "55px" }}>Rsei</th>
                  <th scope="col" style={{ textAlign: "right", paddingRight: "6px", width: "55px" }}>Rct</th>
                  <th scope="col" style={{ textAlign: "right", paddingRight: "6px", width: "60px" }}>χ²ᵣ</th>
                </tr></thead>
                <tbody>
                  {batchSummary.items.map((item, i) => (
                    <tr key={i} onClick={() => setBatchSelectedIdx(i)}
                      style={{ cursor: "pointer", background: i === batchSelectedIdx ? "var(--accent-muted)" : undefined }}>
                      <td style={{ textAlign: "center", color: "var(--text-muted)" }}>{i + 1}</td>
                      <td style={{ textAlign: "left", paddingLeft: "6px", overflow: "hidden", textOverflow: "ellipsis", maxWidth: "90px" }}
                        title={item.file}>{item.label}</td>
                      <td style={{ textAlign: "center" }}>
                        {item.error ? <span style={{ color: "var(--danger-color)" }}>ERR</span>
                          : item.fit ? <span style={{ color: item.fit.status === "ok" ? "var(--success-color)" : item.fit.status === "warn" ? "var(--warning-color)" : "var(--danger-color)", fontWeight: 600 }}>{item.fit.status === "ok" ? "OK" : item.fit.status === "warn" ? "WARN" : "FAIL"}</span>
                            : "-"}
                      </td>
                      <td style={{ textAlign: "right", paddingRight: "6px", fontFamily: "var(--font-mono)" }}>
                        {item.fit?.parameters?.["Rs"] !== undefined ? Number(item.fit.parameters["Rs"]).toPrecision(4) : "-"}
                      </td>
                      <td style={{ textAlign: "right", paddingRight: "6px", fontFamily: "var(--font-mono)" }}>
                        {item.fit?.parameters?.["Rsei"] !== undefined ? Number(item.fit.parameters["Rsei"]).toPrecision(4) : "-"}
                      </td>
                      <td style={{ textAlign: "right", paddingRight: "6px", fontFamily: "var(--font-mono)" }}>
                        {item.fit?.parameters?.["Rct"] !== undefined ? Number(item.fit.parameters["Rct"]).toPrecision(4) : "-"}
                      </td>
                      <td style={{ textAlign: "right", paddingRight: "6px", fontFamily: "var(--font-mono)" }}>
                        {item.fit?.statistics?.chi2_reduced !== undefined ? Number(item.fit.statistics.chi2_reduced).toExponential(2) : item.error ? "Failed" : "-"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {batchSelectedIdx !== null && batchSummary.items[batchSelectedIdx]?.fit?.predicted_real_ohm && (
              <div className="inspect-section">
                <strong style={{ fontSize: "10px" }}>拟合预览: {batchSummary.items[batchSelectedIdx].label}</strong>
                <BatchFitPlot real={batchSummary.items[batchSelectedIdx].fit!.predicted_real_ohm!}
                  imag={batchSummary.items[batchSelectedIdx].fit!.predicted_imag_ohm!} label={batchSummary.items[batchSelectedIdx].label} />
              </div>
            )}

            <div className="inspect-section">
              <div className="form-group" style={{ padding: "0" }}>
                <label htmlFor="batch-export-path">导出路径 (.xlsx)</label>
                <div className="input-with-btn">
                  <input id="batch-export-path" type="text" value={batchExportPath} onChange={(e) => setBatchExportPath(e.target.value)} placeholder="e.g. C:\data\batch_results.xlsx" disabled={isOperationRunning} />
                  <button className="pick-btn" onClick={() => pickFile(setBatchExportPath)} disabled={isOperationRunning}>Browse</button>
                </div>
              </div>
              <button className="export-btn" onClick={handleBatchExport} disabled={!batchExportPath || isOperationRunning} style={{ width: "100%", margin: "10px 0 0 0" }}>
                导出批量数据
              </button>
            </div>
          </div>
        ) : (
          <div className="empty-hint" style={{ padding: "16px 16px" }}>
            {spectra.length > 0 ? "点击「批量拟合」开始处理。" : "请先导入谱图。"}
          </div>
        )}
      </div>

      {/* ── Batch MATLAB DRT section ── */}
      <h3 style={{ marginTop: "16px" }}>批量 MATLAB DRT</h3>
      <div className="form-section">
        <div className="form-group">
          <label>MATLAB 可执行文件</label>
          <div className="input-with-btn">
            <input type="text" value={matlabExe} onChange={(e) => setMatlabExe(e.target.value)} placeholder="D:\Matlabs\bin\matlab.exe" disabled={matlabRunning || isOperationRunning} />
            <button className="pick-btn" onClick={() => pickExe(setMatlabExe)} disabled={matlabRunning || isOperationRunning}>Browse</button>
          </div>
        </div>
        <div className="form-group">
          <label>DRTtools 目录</label>
          <div className="input-with-btn">
            <input type="text" value={drttoolsDir} onChange={(e) => setDrttoolsDir(e.target.value)} placeholder="DRTtools 所在目录" disabled={matlabRunning || isOperationRunning} />
            <button className="pick-btn" onClick={() => pickDir(setDrttoolsDir)} disabled={matlabRunning || isOperationRunning}>Browse</button>
          </div>
        </div>
        <div className="form-group">
          <label>Bridge 目录</label>
          <input type="text" value={matlabBridgeDir} onChange={(e) => setMatlabBridgeDir(e.target.value)} placeholder="matlab_bridge 目录" disabled={matlabRunning || isOperationRunning} />
        </div>
        <div className="form-group">
          <label>输出目录</label>
          <div className="input-with-btn">
            <input type="text" value={matlabOutputDir} onChange={(e) => setMatlabOutputDir(e.target.value)} placeholder="e.g. C:\data\drt_output" disabled={matlabRunning || isOperationRunning} />
            <button className="pick-btn" onClick={() => pickDir(setMatlabOutputDir)} disabled={matlabRunning || isOperationRunning}>Browse</button>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group flex-1" style={{ padding: "0" }}>
            <label>计算方式</label>
            <select value={matlabMethod} onChange={(e) => setMatlabMethod(e.target.value)} disabled={matlabRunning || isOperationRunning}>
              <option value="simple">标准法 (Tikhonov)</option>
              <option value="credit">贝叶斯置信区间</option>
              <option value="bht">BHT (贝叶斯分层)</option>
              <option value="peak">峰拟合分析</option>
            </select>
          </div>
          <div className="form-group flex-1" style={{ padding: "0 0 0 8px" }}>
            <label>DRT 类型</label>
            <select value={matlabDrtType} onChange={(e) => setMatlabDrtType(Number(e.target.value))} disabled={matlabRunning || isOperationRunning}>
              <option value={1}>tau / gamma</option><option value={2}>freq / gamma</option><option value={3}>tau / g</option><option value={4}>freq / g</option>
            </select>
          </div>
        </div>

        <div className="form-row" style={{ marginTop: "10px" }}>
          <div className="form-group flex-1" style={{ padding: "0" }}>
            <label>Lambda (λ)</label>
            <input type="text" value={matlabLambda} onChange={(e) => setMatlabLambda(e.target.value)} placeholder="0.001" disabled={matlabRunning || isOperationRunning} />
          </div>
          <div className="form-group flex-1" style={{ padding: "0 0 0 8px" }}>
            <label>FWHM Coeff</label>
            <input type="text" value={matlabCoeff} onChange={(e) => setMatlabCoeff(e.target.value)} placeholder="0.5" disabled={matlabRunning || isOperationRunning} />
          </div>
        </div>

        <div className="form-row" style={{ marginTop: "10px" }}>
          <div className="form-group flex-1" style={{ padding: "0" }}>
            <label>电感处理</label>
            <select value={matlabInductance} onChange={(e) => setMatlabInductance(Number(e.target.value))} disabled={matlabRunning || isOperationRunning}>
              <option value={1}>保留电感</option><option value={2}>忽略电感</option><option value={3}>去除电感</option>
            </select>
          </div>
          <div className="form-group flex-1" style={{ padding: "0 0 0 8px" }}>
            <label>线图 X 轴</label>
            <select value={matlabLineX} onChange={(e) => setMatlabLineX(e.target.value)} disabled={matlabRunning || isOperationRunning}>
              <option value="logtau">log(tau)</option><option value="tau">tau / s</option>
            </select>
          </div>
        </div>

        <div className="form-group" style={{ marginTop: "10px" }}>
          <label>积分分区 logτ</label>
          <input type="text" value={matlabLogtauBreaks} onChange={(e) => setMatlabLogtauBreaks(e.target.value)} placeholder="-3, 0" disabled={matlabRunning || isOperationRunning} />
        </div>

        <div className="form-actions" style={{ marginTop: "12px" }}>
          <button className="run-btn" onClick={handleBatchMatlabDrt} disabled={matlabRunning || spectra.length === 0 || isOperationRunning}>
            {matlabRunning ? "执行中..." : "运行批量 MATLAB DRT"}
          </button>
        </div>

        {matlabLog && (
          <div className="matlab-log-section" style={{ marginTop: "12px" }}>
            <h3>执行日志</h3>
            <pre className="matlab-log">{matlabLog}</pre>
          </div>
        )}
      </div>
    </>
  );
}

// =========================================================================
// Trend plot
// =========================================================================

function TrendPlot({ items, showRs, showRsei, showRct }: {
  items: { label: string; fit: FitOutcome | null }[]; showRs: boolean; showRsei: boolean; showRct: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null); const containerRef = useRef<HTMLDivElement>(null);
  const [dim, setDim] = useState({ w: 260, h: 130 });
  useEffect(() => {
    const c = containerRef.current; if (!c) return;
    const obs = new ResizeObserver((e) => { if (e[0]) setDim({ w: Math.max(e[0].contentRect.width, 200), h: 130 }); });
    obs.observe(c); return () => obs.disconnect();
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = dim.w * dpr; canvas.height = dim.h * dpr; ctx.scale(dpr, dpr);
    const pad = { t: 12, r: 10, b: 24, l: 34 };
    ctx.fillStyle = "#141416"; ctx.fillRect(0, 0, dim.w, dim.h);

    const series: { name: string; color: string; data: (number | null)[] }[] = [];
    if (showRs) series.push({ name: "Rs", color: "#42A5F5", data: items.map((it) => it.fit?.parameters?.["Rs"] ?? null) });
    if (showRsei) series.push({ name: "Rsei", color: "#FFA726", data: items.map((it) => it.fit?.parameters?.["Rsei"] ?? null) });
    if (showRct) series.push({ name: "Rct", color: "#66BB6A", data: items.map((it) => it.fit?.parameters?.["Rct"] ?? null) });
    if (series.length === 0) return;
    const allY = series.flatMap((s) => s.data.filter((v): v is number => v !== null));
    if (allY.length === 0) return;

    const yMax = Math.max(...allY) * 1.1; const xMax = items.length - 1;
    const pw = dim.w - pad.l - pad.r, ph = dim.h - pad.t - pad.b;
    const tx = (v: number) => pad.l + (v / (xMax || 1)) * pw;
    const ty = (v: number) => dim.h - pad.b - (v / (yMax || 1)) * ph;

    ctx.strokeStyle = "rgba(140,140,150,0.06)"; ctx.setLineDash([3,3]); ctx.lineWidth = 1;
    [0, 0.5, 1].forEach((f) => { const y = ty(yMax * f); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(dim.w - pad.r, y); ctx.stroke(); });
    ctx.setLineDash([]);

    for (const s of series) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.beginPath();
      let started = false;
      for (let i = 0; i < s.data.length; i++) { const v = s.data[i]; if (v === null) { started = false; continue; } if (!started) { ctx.moveTo(tx(i), ty(v)); started = true; } else ctx.lineTo(tx(i), ty(v)); }
      ctx.stroke();
      ctx.fillStyle = s.color;
      for (let i = 0; i < s.data.length; i++) { if (s.data[i] !== null) { ctx.beginPath(); ctx.arc(tx(i), ty(s.data[i]!), 2.5, 0, Math.PI * 2); ctx.fill(); } }
    }

    ctx.fillStyle = "#8c8c96"; ctx.font = "8px ui-monospace,monospace"; ctx.textAlign = "center"; ctx.fillText("样品序号", dim.w / 2, dim.h - 4);
    ctx.save(); ctx.translate(8, dim.h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("阻值 (Ω)", 0, 0); ctx.restore();

    ctx.font = "9px ui-monospace,monospace"; let lx = pad.l;
    for (const s of series) { ctx.fillStyle = s.color; ctx.fillText(s.name, lx, pad.t - 2); lx += ctx.measureText(s.name).width + 14; }
  }, [items, showRs, showRsei, showRct, dim]);

  useEffect(() => { const raf = requestAnimationFrame(draw); return () => cancelAnimationFrame(raf); }, [draw]);
  return <div ref={containerRef} style={{ width: "100%", marginTop: "4px" }}><canvas ref={canvasRef} style={{ width: "100%", height: "130px", display: "block", borderRadius: "4px", border: "1px solid #27272b" }} /></div>;
}

// =========================================================================
// MATLAB DRT tab (standalone — for single spectrum or explicit config)
// =========================================================================

function MatlabTab({
  spectra, onStatusChange, isOperationRunning, onIsOperationRunningChange, batchMode,
}: {
  spectra: SpectrumData[]; onStatusChange: (msg: string) => void; isOperationRunning: boolean; onIsOperationRunningChange: (v: boolean) => void; batchMode: boolean;
}) {
  const [matlabExe, setMatlabExe] = useState("");
  const [drttoolsDir, setDrttoolsDir] = useState("");
  const [matlabBridgeDir, setMatlabBridgeDir] = useState("");
  const [matlabOutputDir, setMatlabOutputDir] = useState("");
  const [matlabMethod, setMatlabMethod] = useState("simple");
  const [matlabDrtType, setMatlabDrtType] = useState(2);
  const [matlabLambda, setMatlabLambda] = useState("0.001"); const [matlabCoeff, setMatlabCoeff] = useState("0.5");
  const [matlabInductance, setMatlabInductance] = useState(1); const [matlabLineX, setMatlabLineX] = useState("logtau");
  const [matlabLogtauBreaks, setMatlabLogtauBreaks] = useState("-3,0");
  const [matlabRunning, setMatlabRunning] = useState(false); const [matlabLog, setMatlabLog] = useState("");

  useEffect(() => {
    invoke<DevResourcePaths>("get_dev_resource_paths").then((p) => {
      if (p.matlab_bridge_dir) setMatlabBridgeDir(p.matlab_bridge_dir);
      if (p.drttools_dir) setDrttoolsDir(p.drttools_dir);
    }).catch(() => {});
  }, []);

  const pickDir = (s: (v: string) => void) => open({ directory: true, multiple: false }).then((p) => { if (p) s(p); });
  const pickExe = (s: (v: string) => void) => open({ multiple: false, filters: [{ name: "Executable", extensions: ["exe"] }] }).then((p) => { if (p) s(p); });

  const handleRun = async () => {
    if (spectra.length === 0) { onStatusChange("请先导入谱图。"); return; }
    if (!matlabExe) { onStatusChange("请设置 MATLAB 可执行文件路径。"); return; }
    if (!drttoolsDir) { onStatusChange("请设置 DRTtools 目录。"); return; }
    if (!matlabBridgeDir) { onStatusChange("请设置 MATLAB bridge 目录。"); return; }
    if (!matlabOutputDir) { onStatusChange("请设置输出目录。"); return; }
    const breaks = matlabLogtauBreaks.split(",").map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
    const req: MatlabDrtRequest = {
      spectra_paths: spectra.map((s) => s.metadata.file_path), output_dir: matlabOutputDir,
      matlab_exe: matlabExe, drttools_dir: drttoolsDir, matlab_bridge_dir: matlabBridgeDir,
      method: matlabMethod, drt_type: matlabDrtType,
      lambda_value: parseFloat(matlabLambda) || 0.001, coeff_value: parseFloat(matlabCoeff) || 0.5,
      inductance_mode: matlabInductance, derivative_order: "1st-order",
      data_used: "Combined Re-Im Data", shape_control: "FWHM Coefficient",
      line_x_axis: matlabLineX, logtau_breaks: breaks,
    };
    setMatlabRunning(true); onIsOperationRunningChange(true); setMatlabLog(""); onStatusChange("正在后台运行 MATLAB DRT...");
    try {
      const result = await invoke<MatlabDrtResult>("run_matlab_drt", { req });
      let log = `命令:\n${result.command}\n\n退出码: ${result.return_code ?? "null"}\n\n--- stdout ---\n${result.stdout}\n\n--- stderr ---\n${result.stderr}\n\n`;
      log += `输出文件 (${result.output_files.length}):\n`;
      for (const f of result.output_files) log += `  ${f.split("\\").pop()?.split("/").pop()}\n`;
      if (result.workbook_path) log += `\nExcel Workbook: ${result.workbook_path}\n`;
      setMatlabLog(log);
      onStatusChange(result.return_code === 0 ? `MATLAB DRT 完成: ${result.output_files.length} 文件` : `MATLAB 执行失败，退出码 ${result.return_code}`);
    } catch (err) { setMatlabLog(`错误: ${err}`); onStatusChange(`MATLAB DRT 失败: ${err}`); }
    finally { setMatlabRunning(false); onIsOperationRunningChange(false); }
  };

  return (
    <>
      <h3>MATLAB DRT 分析</h3>
      <div className="form-section">
        <div className="form-group"><label>MATLAB 可执行文件</label><div className="input-with-btn"><input type="text" value={matlabExe} onChange={(e) => setMatlabExe(e.target.value)} placeholder="D:\Matlabs\bin\matlab.exe" disabled={matlabRunning || isOperationRunning} /><button className="pick-btn" onClick={() => pickExe(setMatlabExe)} disabled={matlabRunning || isOperationRunning}>Browse</button></div></div>
        <div className="form-group"><label>DRTtools 目录</label><div className="input-with-btn"><input type="text" value={drttoolsDir} onChange={(e) => setDrttoolsDir(e.target.value)} placeholder="DRTtools" disabled={matlabRunning || isOperationRunning} /><button className="pick-btn" onClick={() => pickDir(setDrttoolsDir)} disabled={matlabRunning || isOperationRunning}>Browse</button></div></div>
        <div className="form-group"><label>Bridge 目录</label><input type="text" value={matlabBridgeDir} onChange={(e) => setMatlabBridgeDir(e.target.value)} placeholder="matlab_bridge" disabled={matlabRunning || isOperationRunning} /></div>
        <div className="form-group"><label>输出目录</label><div className="input-with-btn"><input type="text" value={matlabOutputDir} onChange={(e) => setMatlabOutputDir(e.target.value)} placeholder="e.g. C:\data\drt_output" disabled={matlabRunning || isOperationRunning} /><button className="pick-btn" onClick={() => pickDir(setMatlabOutputDir)} disabled={matlabRunning || isOperationRunning}>Browse</button></div></div>
        <div className="form-row">
          <div className="form-group flex-1" style={{ padding: "0" }}><label>计算方式</label><select value={matlabMethod} onChange={(e) => setMatlabMethod(e.target.value)} disabled={matlabRunning || isOperationRunning}><option value="simple">标准法</option><option value="credit">贝叶斯置信区间</option><option value="bht">BHT</option><option value="peak">峰拟合</option></select></div>
          <div className="form-group flex-1" style={{ padding: "0 0 0 8px" }}><label>DRT 类型</label><select value={matlabDrtType} onChange={(e) => setMatlabDrtType(Number(e.target.value))} disabled={matlabRunning || isOperationRunning}><option value={1}>tau/gamma</option><option value={2}>freq/gamma</option><option value={3}>tau/g</option><option value={4}>freq/g</option></select></div>
        </div>
        <div className="form-row" style={{ marginTop: "10px" }}>
          <div className="form-group flex-1" style={{ padding: "0" }}><label>Lambda</label><input type="text" value={matlabLambda} onChange={(e) => setMatlabLambda(e.target.value)} disabled={matlabRunning || isOperationRunning} /></div>
          <div className="form-group flex-1" style={{ padding: "0 0 0 8px" }}><label>Coeff</label><input type="text" value={matlabCoeff} onChange={(e) => setMatlabCoeff(e.target.value)} disabled={matlabRunning || isOperationRunning} /></div>
        </div>
        <div className="form-row" style={{ marginTop: "10px" }}>
          <div className="form-group flex-1" style={{ padding: "0" }}><label>电感处理</label><select value={matlabInductance} onChange={(e) => setMatlabInductance(Number(e.target.value))} disabled={matlabRunning || isOperationRunning}><option value={1}>保留</option><option value={2}>忽略</option><option value={3}>去除</option></select></div>
          <div className="form-group flex-1" style={{ padding: "0 0 0 8px" }}><label>X 轴</label><select value={matlabLineX} onChange={(e) => setMatlabLineX(e.target.value)} disabled={matlabRunning || isOperationRunning}><option value="logtau">log(tau)</option><option value="tau">tau/s</option></select></div>
        </div>
        <div className="form-group" style={{ marginTop: "10px" }}><label>积分分区 logτ</label><input type="text" value={matlabLogtauBreaks} onChange={(e) => setMatlabLogtauBreaks(e.target.value)} disabled={matlabRunning || isOperationRunning} /></div>
        <div className="form-actions"><button className="run-btn" onClick={handleRun} disabled={matlabRunning || spectra.length === 0 || isOperationRunning}>{matlabRunning ? "执行中..." : "运行 MATLAB DRT"}</button><button className="clear-btn" onClick={() => setMatlabLog("")} disabled={matlabRunning || isOperationRunning}>清除日志</button></div>
        {matlabLog && <div className="matlab-log-section"><h3>执行日志</h3><pre className="matlab-log">{matlabLog}</pre></div>}
      </div>
    </>
  );
}

// =========================================================================
// DRT Export tab
// =========================================================================

function DrtExportTab({ onStatusChange, isOperationRunning, onIsOperationRunningChange }: {
  onStatusChange: (msg: string) => void; isOperationRunning: boolean; onIsOperationRunningChange: (v: boolean) => void;
}) {
  const [drtSpectraDir, setDrtSpectraDir] = useState(""); const [drtDir, setDrtDir] = useState("");
  const [drtOutput, setDrtOutput] = useState(""); const [lineX, setLineX] = useState("logtau");
  const [logtauBreaks, setLogtauBreaks] = useState("-3,0");

  const pickDir = (s: (v: string) => void) => open({ directory: true, multiple: false }).then((p) => { if (p) s(p); });
  const pickFile = (s: (v: string) => void) => open({ directory: false, multiple: false, filters: [{ name: "Excel", extensions: ["xlsx"] }] }).then((p) => { if (p) s(p); });

  const handleExport = async () => {
    if (!drtSpectraDir || !drtDir || !drtOutput) { onStatusChange("请填写所有 DRT 导出路径。"); return; }
    const breaks = logtauBreaks.split(",").map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
    const req: DrtExportRequest = { spectra_dir: drtSpectraDir, drt_dir: drtDir, output_path: drtOutput, line_x: lineX, logtau_breaks: breaks };
    onIsOperationRunningChange(true);
    try { const r = await invoke<string>("export_drt_workbook", { req }); onStatusChange(r); }
    catch (err) { onStatusChange(`导出失败: ${err}`); }
    finally { onIsOperationRunningChange(false); }
  };

  return (
    <>
      <h3>手动 DRT 导出</h3>
      <div className="form-section">
        <div className="form-group"><label>谱图目录</label><div className="input-with-btn"><input type="text" value={drtSpectraDir} onChange={(e) => setDrtSpectraDir(e.target.value)} placeholder="e.g. C:\data\spectra" disabled={isOperationRunning} /><button className="pick-btn" onClick={() => pickDir(setDrtSpectraDir)} disabled={isOperationRunning}>Browse</button></div></div>
        <div className="form-group"><label>DRT 结果目录</label><div className="input-with-btn"><input type="text" value={drtDir} onChange={(e) => setDrtDir(e.target.value)} placeholder="e.g. C:\data\drt_results" disabled={isOperationRunning} /><button className="pick-btn" onClick={() => pickDir(setDrtDir)} disabled={isOperationRunning}>Browse</button></div></div>
        <div className="form-group"><label>输出 XLSX 路径</label><div className="input-with-btn"><input type="text" value={drtOutput} onChange={(e) => setDrtOutput(e.target.value)} placeholder="e.g. C:\data\drt_matrix.xlsx" disabled={isOperationRunning} /><button className="pick-btn" onClick={() => pickFile(setDrtOutput)} disabled={isOperationRunning}>Browse</button></div></div>
        <div className="form-row">
          <div className="form-group flex-1" style={{ padding: "0" }}><label>线图 X 轴</label><select value={lineX} onChange={(e) => setLineX(e.target.value)} disabled={isOperationRunning}><option value="logtau">log(tau)</option><option value="tau">tau (s)</option></select></div>
          <div className="form-group flex-1" style={{ padding: "0 0 0 8px" }}><label>log(tau) 分界</label><input type="text" value={logtauBreaks} onChange={(e) => setLogtauBreaks(e.target.value)} placeholder="-3,0" disabled={isOperationRunning} /></div>
        </div>
        <button className="export-btn" onClick={handleExport} disabled={!drtSpectraDir || !drtDir || !drtOutput || isOperationRunning}>导出 DRT Workbook</button>
      </div>
    </>
  );
}
