import { useRef, useEffect, useCallback, useState } from "react";

/** Responsive Bode magnitude or phase plot — fills container. */
export function BodePlot({
  freq, values, yLabel, logY, color,
}: {
  freq: number[]; values: number[]; yLabel: string; logY: boolean; color: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 260, h: 150 });

  useEffect(() => {
    const c = containerRef.current;
    if (!c) return;
    const obs = new ResizeObserver((entries) => {
      if (!entries.length) return;
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setSize({ w: width, h: height });
    });
    obs.observe(c);
    return () => obs.disconnect();
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const W = size.w, H = size.h;
    if (W <= 0 || H <= 0) return;
    canvas.width = W * dpr; canvas.height = H * dpr; ctx.scale(dpr, dpr);

    const pad = { t: 12, r: 12, b: 32, l: 44 };
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = "#141416"; ctx.fillRect(0, 0, W, H);
    if (freq.length === 0) return;

    const valid: { x: number; y: number }[] = [];
    for (let i = 0; i < freq.length; i++) {
      if (freq[i] > 0 && isFinite(freq[i]) && isFinite(values[i])) {
        const logF = Math.log10(freq[i]);
        const v = logY ? (values[i] > 0 ? Math.log10(values[i]) : NaN) : values[i];
        if (isFinite(v)) valid.push({ x: logF, y: v });
      }
    }
    if (valid.length < 2) return;

    const xs = valid.map((p) => p.x), ys = valid.map((p) => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const tx = (v: number) => pad.l + ((v - xMin) / (xMax - xMin || 1)) * pw;
    const ty = (v: number) => H - pad.b - ((v - yMin) / (yMax - yMin || 0.1)) * ph;

    const xMid = (xMin + xMax) / 2, yMid = (yMin + yMax) / 2;
    ctx.strokeStyle = "rgba(140,140,150,0.06)"; ctx.setLineDash([4,4]); ctx.lineWidth = 1;
    [xMin, xMid, xMax].forEach((v) => { const x = tx(v); ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke(); });
    [yMin, yMid, yMax].forEach((v) => { const y = ty(v); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); });
    ctx.setLineDash([]);

    ctx.strokeStyle = "#27272b"; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.moveTo(pad.l, H - pad.b); ctx.lineTo(pad.l, pad.t); ctx.stroke();

    ctx.fillStyle = "#8c8c96"; ctx.font = "9px ui-monospace,monospace"; ctx.textAlign = "center";
    ctx.fillText("Freq (Hz)", W / 2, H - 6);
    ctx.save(); ctx.translate(10, H / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yLabel, 0, 0); ctx.restore();

    // Data points
    ctx.fillStyle = color;
    for (const p of valid) {
      ctx.beginPath(); ctx.arc(tx(p.x), ty(p.y), 2.2, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#141416"; ctx.lineWidth = 0.5; ctx.stroke();
    }
  }, [freq, values, yLabel, logY, color, size]);

  useEffect(() => { const raf = requestAnimationFrame(draw); return () => cancelAnimationFrame(raf); }, [draw]);

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%", minHeight: "120px" }}>
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block", borderRadius: "4px", border: "1px solid #27272b" }} />
    </div>
  );
}
