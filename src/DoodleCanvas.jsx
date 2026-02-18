import React, { useEffect, useRef, useState } from "react";

export default function DoodleCanvas({ onCanvasReady, onInteractingChange }) {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);

  const drawingRef = useRef(false);
  const lastRef = useRef({ x: 0, y: 0 });

  const [brush, setBrush] = useState(18);
  const [isErasing, setIsErasing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    canvas.width = 280;
    canvas.height = 280;

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    ctxRef.current = ctx;
    onCanvasReady?.(canvas);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);


  const applyBrush = () => {
    const ctx = ctxRef.current;
    if (!ctx) return;

    ctx.lineWidth = brush;
    ctx.strokeStyle = isErasing ? "black" : "white";
    ctx.globalCompositeOperation = "source-over"; // IMPORTANT
  };


  useEffect(() => {
    applyBrush();
  }, [brush, isErasing]);

  const getPos = (clientX, clientY) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height),
    };
  };

  const startStroke = (clientX, clientY) => {
    drawingRef.current = true;
    onInteractingChange?.(true);
    applyBrush();
    lastRef.current = getPos(clientX, clientY);
  };

  const moveStroke = (clientX, clientY) => {
    if (!drawingRef.current) return;
    const ctx = ctxRef.current;
    if (!ctx) return;

    applyBrush();

    const p = getPos(clientX, clientY);
    const last = lastRef.current;

    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();

    lastRef.current = p;
  };

  const endStroke = () => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    onInteractingChange?.(false);
  };

  const clear = () => {
    const canvas = canvasRef.current;
    const ctx = ctxRef.current;
    if (!canvas || !ctx) return;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  return (
    <div style={{ display: "grid", gap: 14 }}>
      <canvas
        ref={canvasRef}
        style={{
          width: 360,
          height: 360,
          borderRadius: 16,
          border: "1px solid rgba(255,255,255,0.15)",
          background: "black",
          display: "block",
          cursor: "crosshair",
          WebkitUserSelect: "none",
          userSelect: "none",
          // important for Safari/trackpad so it doesn't scroll/zoom instead
          touchAction: "none",
        }}
        // MOUSE (trackpad behaves like mouse)
        onMouseDown={(e) => {
          if (e.button !== 0) return;
          e.preventDefault();
          startStroke(e.clientX, e.clientY);
        }}
        onMouseMove={(e) => {
          // Safari trackpad sometimes won’t report buttons reliably; rely on our ref flag
          moveStroke(e.clientX, e.clientY);
        }}
        onMouseUp={(e) => {
          e.preventDefault();
          endStroke();
        }}
        onMouseLeave={() => endStroke()}
        // TOUCH (for phones/ipads)
        onTouchStart={(e) => {
          e.preventDefault();
          const t = e.touches[0];
          startStroke(t.clientX, t.clientY);
        }}
        onTouchMove={(e) => {
          e.preventDefault();
          const t = e.touches[0];
          moveStroke(t.clientX, t.clientY);
        }}
        onTouchEnd={(e) => {
          e.preventDefault();
          endStroke();
        }}
        onTouchCancel={(e) => {
          e.preventDefault();
          endStroke();
        }}
      />

      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <button
          onClick={clear}
          style={{ padding: "10px 12px", borderRadius: 12, border: 0, cursor: "pointer" }}
        >
          Clear
        </button>

        <button
          onClick={() => setIsErasing((v) => !v)}
          style={{ padding: "10px 12px", borderRadius: 12, border: 0, cursor: "pointer" }}
        >
          {isErasing ? "Draw mode" : "Eraser"}
        </button>
      </div>

      <div style={{ display: "grid", gap: 6, maxWidth: 360 }}>
        <div style={{ display: "flex", justifyContent: "space-between", opacity: 0.85 }}>
          <span>Brush</span>
          <span>{brush}</span>
        </div>

        <input
          type="range"
          min="6"
          max="34"
          value={brush}
          onChange={(e) => setBrush(Number(e.target.value))}
          onMouseDown={() => onInteractingChange?.(true)}
          onMouseUp={() => onInteractingChange?.(false)}
          style={{ width: "100%", height: 28, cursor: "pointer" }}
        />
      </div>
    </div>
  );
}
