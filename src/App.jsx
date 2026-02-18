import React, { useCallback, useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import DoodleCanvas from "./DoodleCanvas";

export default function App() {
  const canvasRef = useRef(null);
  const modelRef = useRef(null);
  const rafRef = useRef(null);

  const [modelReady, setModelReady] = useState(false);
  const [isInteracting, setIsInteracting] = useState(false);
  const [predictions, setPredictions] = useState([]);

  const handleCanvasReady = useCallback((c) => {
    canvasRef.current = c;
  }, []);

  // Load model
  useEffect(() => {
    const load = async () => {
      await tf.ready();
      const modelUrl = `${import.meta.env.BASE_URL}model/model.json`;
      const model = await tf.loadGraphModel(modelUrl);
      modelRef.current = model;
      setModelReady(true);
    };
    load();
  }, []);

  // Preprocess canvas → tensor
  const preprocessCanvas = (canvas) => {
    const off = document.createElement("canvas");
    off.width = 28;
    off.height = 28;
    const ctx = off.getContext("2d");

    ctx.drawImage(canvas, 0, 0, 28, 28);

    const img = ctx.getImageData(0, 0, 28, 28).data;
    const data = new Float32Array(28 * 28);

    for (let i = 0; i < 28 * 28; i++) {
      const r = img[i * 4];
      const g = img[i * 4 + 1];
      const b = img[i * 4 + 2];
      data[i] = (r + g + b) / (3 * 255);
    }

    return tf.tensor4d(data, [1, 28, 28, 1]);
  };

  const predict = useCallback(async () => {
    if (!modelRef.current || !canvasRef.current) return;

    const probs = await tf.tidy(async () => {
      const input = preprocessCanvas(canvasRef.current);
      const output = modelRef.current.predict(input);
      return Array.from(await output.data());
    });

    const mapped = probs
      .map((p, i) => ({ label: i, prob: p }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 3);

    setPredictions(mapped);
  }, []);

  // Live prediction loop
  useEffect(() => {
    if (!modelReady) return;

    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    if (!isInteracting) {
      predict();
      return;
    }

    const loop = () => {
      predict();
      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [isInteracting, modelReady, predict]);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0b0b10",
        color: "white",
        padding: 24,
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          maxWidth: 1200,
          margin: "0 auto",
          display: "grid",
          gap: 24,
        }}
      >
        <h1 style={{ margin: 0, fontSize: 32 }}>
          Doodle Digit Classifier
        </h1>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(320px, 520px) 1fr",
            gap: 24,
          }}
        >
          {/* Canvas */}
          <div
            style={{
              padding: 16,
              borderRadius: 18,
              border: "1px solid rgba(255,255,255,0.12)",
              background: "rgba(255,255,255,0.03)",
            }}
          >
            <DoodleCanvas
              onCanvasReady={handleCanvasReady}
              onInteractingChange={setIsInteracting}
            />
          </div>

          {/* Predictions */}
          <div
            style={{
              padding: 16,
              borderRadius: 18,
              border: "1px solid rgba(255,255,255,0.12)",
              background: "rgba(255,255,255,0.03)",
              minHeight: 360,
              display: "grid",
              alignContent: "start",
              gap: 16,
            }}
          >
            <h2 style={{ margin: 0 }}>Predictions</h2>

            {!modelReady ? (
              <div style={{ opacity: 0.7 }}>Loading model…</div>
            ) : predictions.length === 0 ? (
              <div style={{ opacity: 0.7 }}>Draw a digit (0–9)</div>
            ) : (
              <ol style={{ margin: 0, paddingLeft: 18, display: "grid", gap: 14 }}>
                {predictions.map((p) => (
                  <li key={p.label}>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <strong style={{ fontSize: 20 }}>{p.label}</strong>
                      <span>{(p.prob * 100).toFixed(1)}%</span>
                    </div>
                    <div
                      style={{
                        height: 10,
                        borderRadius: 999,
                        background: "rgba(255,255,255,0.12)",
                        overflow: "hidden",
                        marginTop: 6,
                      }}
                    >
                      <div
                        style={{
                          height: "100%",
                          width: `${Math.max(1, Math.round(p.prob * 100))}%`,
                          background: "white",
                        }}
                      />
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
