import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import type { SpectrumData } from "../types";

export function Toolbar({
  spectraCount,
  showPlot,
  onShowPlotChange,
  onImported,
  onStatusChange,
  isOperationRunning,
  onIsOperationRunningChange,
  onClearSpectra,
}: {
  spectraCount: number;
  showPlot: boolean;
  onShowPlotChange: (v: boolean) => void;
  onImported: (spectra: SpectrumData[], firstNewIdx: number) => void;
  onStatusChange: (msg: string) => void;
  isOperationRunning: boolean;
  onIsOperationRunningChange: (v: boolean) => void;
  onClearSpectra: () => void;
}) {
  const handleImportFiles = async () => {
    const selected = await open({
      multiple: true,
      filters: [{ name: "EIS Data", extensions: ["txt", "csv", "bin"] }],
    });
    if (!selected) return;

    const paths = Array.isArray(selected) ? selected : [selected];
    onStatusChange(`正在导入 ${paths.length} 个文件...`);
    onIsOperationRunningChange(true);

    const results: SpectrumData[] = [];
    for (const p of paths) {
      try {
        const data = await invoke<SpectrumData>("parse_file", { path: p });
        results.push(data);
      } catch (err) {
        onStatusChange(`导入失败 ${p.split("\\").pop()}: ${err}`);
      }
    }

    onIsOperationRunningChange(false);
    if (results.length > 0) {
      onImported(results, spectraCount);
      onStatusChange(`已导入 ${results.length} 个谱图`);
    } else {
      onStatusChange("未找到有效谱图数据");
    }
  };

  const handleImportFolder = async () => {
    const selected = await open({
      directory: true,
      multiple: false,
    });
    if (!selected) return;

    onStatusChange(`正在导入文件夹: ${selected.split("\\").pop()}...`);
    onIsOperationRunningChange(true);
    try {
      const results = await invoke<SpectrumData[]>("parse_folder", {
        dir: selected,
      });
      onImported(results, spectraCount);
      onStatusChange(`已从文件夹导入 ${results.length} 个谱图`);
    } catch (err) {
      onStatusChange(`导入失败: ${err}`);
    } finally {
      onIsOperationRunningChange(false);
    }
  };

  return (
    <div className="toolbar" role="toolbar">
      <div className="toolbar-brand">
        <span className="toolbar-title">EISMaster Pro</span>
        <span className="toolbar-tagline">电化学阻抗谱分析</span>
      </div>
      <span className="toolbar-sep" />
      <button onClick={handleImportFiles} disabled={isOperationRunning}>
        导入文件
      </button>
      <button onClick={handleImportFolder} disabled={isOperationRunning}>
        导入文件夹
      </button>
      {spectraCount > 0 && (
        <>
          <button
            onClick={onClearSpectra}
            disabled={isOperationRunning}
            className="clear-btn"
          >
            清空全部
          </button>
        </>
      )}
      <span className="toolbar-sep" />
      <label className="toolbar-checkbox-label">
        <input
          type="checkbox"
          checked={showPlot}
          onChange={(e) => onShowPlotChange(e.target.checked)}
        />
        显示图表
      </label>
      <div className="toolbar-info">
        {spectraCount > 0 && <span>{spectraCount} 个谱图</span>}
      </div>
    </div>
  );
}
