import React, { useCallback, useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import DoodleCanvas from "./DoodleCanvas";

export default function App() {
  const canvasRef = useRef(null);
  const modelRef = useRef(null);
  const rafRef = useRef(null);

  const [status, setStatus] = useState("Loading model...");
  const [isInteracting, setIsInteracting] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [predictions, setPredictions] = useState([]); // top 3: [{label, prob}]

  const handleCanvasReady = useCallback((c) => {
    canvasRef.current = c;
  }, []);

  // Load TFJS GraphModel (converted from SavedModel)
  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        setStatus("Loading model...");
        await tf.ready();
        const modelUrl = `${import.meta.env.BASE_URL}model/model.json`;
        const m = await tf.loadGraphModel(modelUrl);
        modelRef.current = m; // 

        if (!cancelled) {
          setModelReady(true);
          setStatus("Model ready ✅ Draw a digit (0–9).");
        }
      } catch (e) {
        console.error(e);
        if (!cancelled) setStatus("Model failed to load ❌ (check console)");
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  // Canvas -> tensor [1,28,28,1]
  const preprocessCanvas = (canvas) => {
    const off = document.createElement("canvas");
    off.width = 28;
    off.height = 28;
    const ctx = off.getContext("2d", { willReadFrequently: true });

    // scale down
    ctx.drawImage(canvas, 0, 0, 28, 28);

    const img = ctx.getImageData(0, 0, 28, 28).data;
    const data = new Float32Array(28 * 28);

    // grayscale from rgb, normalize to [0,1]
    for (let i = 0; i < 28 * 28; i++) {
      const r = img[i * 4 + 0];
      const g = img[i * 4 + 1];
      const b = img[i * 4 + 2];
      const gray = (r + g + b) / (3 * 255); // white stroke => ~1, black bg => ~0
      data[i] = gray;
    }

    return tf.tensor4d(data, [1, 28, 28, 1]);
  };

  const predictOnce = useCallback(async () => {
    const model = modelRef.current;
    const canvas = canvasRef.current;
    if (!model || !canvas) return;

    // Do the whole prediction in a tidy to avoid leaks
    const probs = await tf.tidy(async () => {
      const input = preprocessCanvas(canvas);
      const out = modelRef.current.execute(input); // GraphModel supports predict() alias
      const data = await out.data();
      return Array.from(data);
    });

    const mapped = probs.map((p, i) => ({ label: i, prob: p }));
    mapped.sort((a, b) => b.prob - a.prob);
    setPredictions(mapped.slice(0, 3));

    if (mapped[0]) {
      setStatus(`Prediction: ${mapped[0].label} (${(mapped[0].prob * 100).toFixed(1)}%)`);
    }
  }, []);

  // Predict continuously while drawing for "live" feel; once more on release.
  useEffect(() => {
    if (!modelReady) return;

    // Stop any previous loop
    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    if (!isInteracting) {
      // Predict once when the user stops
      predictOnce();
      return;
    }

    const loop = () => {
      predictOnce();
      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [isInteracting, modelReady, predictOnce]);

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
          width: "100%",
          maxWidth: 1400,
          margin: "0 auto",
          display: "grid",
          gap: 18,
        }}
      >
        <div style={{ display: "grid", gap: 8 }}>
          <h1 style={{ margin: 0, fontSize: 32 }}>Doodle Digit Classifier</h1>
          <p style={{ margin: 0, opacity: 0.8 }}>{status}</p>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(320px, 520px) 1fr",
            gap: 18,
            alignItems: "start",
          }}
        >
          {/* LEFT: canvas */}
          <div
            style={{
              padding: 16,
              borderRadius: 18,
              border: "1px solid rgba(255,255,255,0.12)",
              background: "rgba(255,255,255,0.03)",
              backdropFilter: "blur(6px)",
            }}
          >
            <DoodleCanvas onCanvasReady={handleCanvasReady} onInteractingChange={setIsInteracting} />

          </div>

          {/* RIGHT: predictions */}
          <div
            style={{
              padding: 16,
              borderRadius: 18,
              border: "1px solid rgba(255,255,255,0.12)",
              background: "rgba(255,255,255,0.03)",
              backdropFilter: "blur(6px)",
              display: "grid",
              alignContent: "start",
              gap: 12,
              minHeight: 360,
            }}
          >
            <h2 style={{ margin: 0, fontSize: 18 }}>Predictions</h2>

            {!modelReady ? (
              <div style={{ opacity: 0.8, lineHeight: 1.6 }}>
                Model is loading…
                <div style={{ opacity: 0.7, fontSize: 13, marginTop: 8 }}>
                  If this hangs, open the browser console and tell me what it says.
                </div>
              </div>
            ) : predictions.length === 0 ? (
              <div style={{ opacity: 0.8, lineHeight: 1.6 }}>
                Draw a digit (0–9). When you release, you’ll see the top predictions here.
                <div style={{ opacity: 0.7, fontSize: 13, marginTop: 8 }}>
                  Tip: draw big and centered for best accuracy.
                </div>
              </div>
            ) : (
              <ol style={{ margin: 0, paddingLeft: 18, display: "grid", gap: 12 }}>
                {predictions.map((p) => (
                  <li key={p.label}>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <strong style={{ fontSize: 18 }}>{p.label}</strong>
                      <span style={{ opacity: 0.85 }}>{(p.prob * 100).toFixed(1)}%</span>
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

            <div style={{ opacity: 0.65, fontSize: 12, marginTop: 8 }}>
              Model runs fully in your browser using TensorFlow.js (no server).
            </div>
          </div>
        </div>

        
      </div>
    </div>
  );
}
