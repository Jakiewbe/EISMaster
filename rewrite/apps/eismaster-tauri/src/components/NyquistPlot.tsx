import { useRef, useEffect, useCallback, useState } from "react";

/** Responsive Nyquist scatter plot — fills container, no fixed height. */
export function NyquistPlot({
  spectrum,
  fitReal,
  fitImag,
}: {
  spectrum: { z_real_ohm: number[]; z_imag_ohm: number[] };
  fitReal?: number[] | null;
  fitImag?: number[] | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 400, h: 300 });

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
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    const pad = { t: 14, r: 14, b: 38, l: 50 };
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#141416"; ctx.fillRect(0, 0, W, H);

    const zr = spectrum.z_real_ohm, zi = spectrum.z_imag_ohm;
    if (zr.length === 0) return;

    const allX = [...zr]; const allY = zi.map((v) => -v);
    if (fitReal?.length) { allX.push(...fitReal); allY.push(...fitImag!.map((v) => -v)); }
    const xMin = Math.min(...allX), xMax = Math.max(...allX);
    const yMin = Math.min(...allY), yMax = Math.max(...allY);
    const xR = xMax - xMin || 1, yR = yMax - yMin || 1;
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const sc = Math.min(pw / xR, ph / yR);
    const tx = (v: number) => pad.l + (v - xMin) * sc;
    const ty = (v: number) => H - pad.b - (v - yMin) * sc;
    const xMid = (xMin + xMax) / 2, yMid = (yMin + yMax) / 2;

    // Grid
    ctx.strokeStyle = "rgba(140,140,150,0.06)"; ctx.setLineDash([4,4]); ctx.lineWidth = 1;
    [xMin, xMid, xMax].forEach((v) => { const x = tx(v); ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke(); });
    [yMin, yMid, yMax].forEach((v) => { const y = ty(v); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); });
    ctx.setLineDash([]);

    // Axes
    ctx.strokeStyle = "#27272b"; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b);
    ctx.moveTo(pad.l, H - pad.b); ctx.lineTo(pad.l, pad.t); ctx.stroke();

    // Axis labels
    ctx.fillStyle = "#8c8c96"; ctx.font = "10px ui-monospace,monospace"; ctx.textAlign = "center";
    ctx.fillText("Z' (Ω)", W / 2, H - 8);
    ctx.save(); ctx.translate(12, H / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("-Z'' (Ω)", 0, 0); ctx.restore();

    // Tick labels
    ctx.font = "9px ui-monospace,monospace";
    [xMin, xMid, xMax].forEach((v) => { const x = tx(v); ctx.textAlign = "center"; ctx.fillText(v.toPrecision(3), x, H - pad.b + 14); });
    ctx.textAlign = "right";
    [yMin, yMid, yMax].forEach((v) => { ctx.fillText(v.toPrecision(3), pad.l - 6, ty(v) + 3); });

    // Fit overlay
    if (fitReal && fitImag && fitReal.length > 1) {
      ctx.strokeStyle = "#f43f5e"; ctx.lineWidth = 1.8; ctx.beginPath();
      ctx.moveTo(tx(fitReal[0]), ty(-fitImag[0]));
      for (let i = 1; i < fitReal.length; i++) ctx.lineTo(tx(fitReal[i]), ty(-fitImag[i]));
      ctx.stroke();
    }

    // Data points
    ctx.fillStyle = "#3b82f6";
    for (let i = 0; i < zr.length; i++) {
      ctx.beginPath(); ctx.arc(tx(zr[i]), ty(-zi[i]), 2.8, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#141416"; ctx.lineWidth = 0.5; ctx.stroke();
    }
  }, [spectrum, fitReal, fitImag, size]);

  useEffect(() => { const raf = requestAnimationFrame(draw); return () => cancelAnimationFrame(raf); }, [draw]);

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%", minHeight: "200px" }}>
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block", borderRadius: "4px", border: "1px solid #27272b" }} />
    </div>
  );
}

/** Small Nyquist fit-preview for batch item rows. */
export function BatchFitPlot({
  real, imag, label,
}: { real: number[]; imag: number[]; label: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 300, h: 160 });

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
    canvas.width = W * dpr; canvas.height = H * dpr; ctx.scale(dpr, dpr);
    const pad = { t: 16, r: 10, b: 30, l: 36 };
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = "#141416"; ctx.fillRect(0, 0, W, H);
    if (real.length === 0) return;

    const negImag = imag.map((v) => -v);
    const allX = [...real], allY = [...negImag];
    const xMin = Math.min(...allX), xMax = Math.max(...allX);
    const yMin = Math.min(...allY), yMax = Math.max(...allY);
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const sc = Math.min(pw / (xMax - xMin || 1), ph / (yMax - yMin || 1));
    const tx = (v: number) => pad.l + (v - xMin) * sc;
    const ty = (v: number) => H - pad.b - (v - yMin) * sc;

    ctx.strokeStyle = "rgba(140,140,150,0.04)"; ctx.setLineDash([3,3]); ctx.lineWidth = 1;
    const xMid = (xMin + xMax) / 2, yMid = (yMin + yMax) / 2;
    [xMin, xMid, xMax].forEach((v) => { const x = tx(v); ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke(); });
    [yMin, yMid, yMax].forEach((v) => { const y = ty(v); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); });
    ctx.setLineDash([]);
    ctx.strokeStyle = "#27272b"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.moveTo(pad.l, H - pad.b); ctx.lineTo(pad.l, pad.t); ctx.stroke();

    ctx.strokeStyle = "#f43f5e"; ctx.lineWidth = 1.5; ctx.beginPath();
    ctx.moveTo(tx(real[0]), ty(negImag[0]));
    for (let i = 1; i < real.length; i++) ctx.lineTo(tx(real[i]), ty(negImag[i]));
    ctx.stroke();

    ctx.fillStyle = "#e3e3e6"; ctx.font = "bold 8px ui-monospace,monospace"; ctx.textAlign = "left"; ctx.fillText(label, pad.l + 4, pad.t - 2);
  }, [real, imag, label, size]);

  useEffect(() => { const raf = requestAnimationFrame(draw); return () => cancelAnimationFrame(raf); }, [draw]);

  return (
    <div ref={containerRef} style={{ width: "100%", height: "160px", marginTop: "8px" }}>
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block", borderRadius: "4px", border: "1px solid #27272b" }} />
    </div>
  );
}
