import React, { useEffect, useRef } from "react";

export default function PcaPlot({ points = [], highlight = null, width = 420, height = 280 }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    // background
    ctx.fillStyle = "rgba(255,255,255,0.02)";
    ctx.fillRect(0, 0, width, height);

    if (!points.length) {
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.font = "12px system-ui";
      ctx.fillText("No PCA points yet", 12, 20);
      return;
    }

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [x, y] of points) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    const pad = 14;
    const sx = (x) => pad + ((x - minX) / (maxX - minX + 1e-9)) * (width - pad * 2);
    const sy = (y) => pad + ((y - minY) / (maxY - minY + 1e-9)) * (height - pad * 2);

    // points
    ctx.fillStyle = "rgba(255,255,255,0.65)";
    for (const [x, y] of points) {
      ctx.beginPath();
      ctx.arc(sx(x), sy(y), 2.4, 0, Math.PI * 2);
      ctx.fill();
    }

    // highlight current
    if (highlight) {
      ctx.strokeStyle = "white";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx(highlight[0]), sy(highlight[1]), 6, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [points, highlight, width, height]);

  return (
    <canvas
      ref={ref}
      width={width}
      height={height}
      style={{
        width,
        height,
        borderRadius: 14,
        border: "1px solid rgba(255,255,255,0.12)",
        display: "block",
      }}
    />
  );
}
