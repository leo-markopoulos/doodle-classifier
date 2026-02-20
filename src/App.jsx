import React, { useCallback, useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import DoodleCanvas from "./DoodleCanvas";
import PcaPlot from "./components/PcaPlot";
import { runPCA2D, projectWithPCA } from "./ml/pca";
import { resolveEmbeddingOutputName, extractCnnEmbedding } from "./ml/cnnEmbedding";

function baseUrlJoin(path) {
  const base = import.meta.env.BASE_URL || "/";
  const b = base.endsWith("/") ? base : base + "/";
  const p = path.startsWith("/") ? path.slice(1) : path;
  return b + p;
}

function looksLikeHtml(text) {
  const t = (text || "").trimStart().toLowerCase();
  return t.startsWith("<!doctype") || t.startsWith("<html") || t.startsWith("<");
}

export default function App() {
  const canvasRef = useRef(null);
  const cnnRef = useRef(null);
  const logregRef = useRef(null);

  const lastEmbeddingRef = useRef(null);

  const [modelReady, setModelReady] = useState(false);
  const [selectedModel, setSelectedModel] = useState("pretrained-cnn");
  const [embeddingSource, setEmbeddingSource] = useState("raw"); // raw | cnn-penultimate

  const [predictions, setPredictions] = useState([]);
  const [modelError, setModelError] = useState("");
  const [modelWarning, setModelWarning] = useState("");
  const [lastTriedUrl, setLastTriedUrl] = useState("");

  const [samples, setSamples] = useState([]);
  const [pcaState, setPcaState] = useState({ points2d: [], mean: [], components: [] });
  const [currentPoint, setCurrentPoint] = useState(null);
  const [pcaRunning, setPcaRunning] = useState(false);
  const [pcaError, setPcaError] = useState("");

  const [cnnEmbeddingOutName, setCnnEmbeddingOutName] = useState(null);

  const handleCanvasReady = useCallback((canvas) => {
    canvasRef.current = canvas;
  }, []);

  const preprocessCanvas = (canvas) => {
    const off = document.createElement("canvas");
    off.width = 28;
    off.height = 28;
    const ctx = off.getContext("2d");
    ctx.drawImage(canvas, 0, 0, 28, 28);

    const img = ctx.getImageData(0, 0, 28, 28).data;
    const data = new Float32Array(784);

    for (let i = 0; i < 784; i++) {
      const r = img[i * 4];
      const g = img[i * 4 + 1];
      const b = img[i * 4 + 2];
      data[i] = (r + g + b) / (3 * 255);
    }
    return data;
  };

  const topK = (probs, k = 3) =>
    probs
      .map((p, i) => ({ label: i, prob: p }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, k);

  // Clear PCA state when embedding source changes (prevents mixed dims)
  useEffect(() => {
    setSamples([]);
    setPcaState({ points2d: [], mean: [], components: [] });
    setCurrentPoint(null);
    setPcaError("");
  }, [embeddingSource]);

  // Load model
  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        setModelReady(false);
        setPredictions([]);
        setModelError("");
        setModelWarning("");
        setLastTriedUrl("");
        setCnnEmbeddingOutName(null);

        await tf.ready();

        if (selectedModel === "pretrained-cnn") {
          const url = baseUrlJoin("model/pretrained-cnn/model.json");
          setLastTriedUrl(url);

          const model = await tf.loadGraphModel(url);
          if (cancelled) return;

          cnnRef.current?.dispose?.();
          cnnRef.current = model;

          const outName = resolveEmbeddingOutputName(model);
          setCnnEmbeddingOutName(outName);
          console.log("CNN embedding output:", outName);

          setModelReady(true);
          return;
        }

        if (selectedModel === "logreg") {
          const url = baseUrlJoin("model/logreg/logreg.json");
          setLastTriedUrl(url);

          const res = await fetch(url, { cache: "no-store" });
          if (!res.ok) throw new Error(`Failed to fetch logreg.json (${res.status})`);

          const text = await res.text();
          if (looksLikeHtml(text)) throw new Error("Got HTML instead of JSON (path wrong / fallback)");

          const json = JSON.parse(text);

          const inDim = json.inDim ?? 784;
          const outDim = json.outDim ?? 10;

          const expected = inDim * outDim;
          let Warr = Array.isArray(json.W) ? json.W : [];
          let barr = Array.isArray(json.b) ? json.b : [];

          if (Warr.length !== expected) {
            setModelWarning(
              `LogReg weights length is ${Warr.length} but expected ${expected}. Using zeros until you export real weights.`
            );
            Warr = new Array(expected).fill(0);
          }

          if (barr.length !== outDim) {
            setModelWarning(
              `LogReg bias length is ${barr.length} but expected ${outDim}. Using zeros until you export real weights.`
            );
            barr = new Array(outDim).fill(0);
          }

          if (logregRef.current?.W) logregRef.current.W.dispose();
          if (logregRef.current?.b) logregRef.current.b.dispose();

          logregRef.current = {
            W: tf.tensor2d(Warr, [inDim, outDim], "float32"),
            b: tf.tensor1d(barr, "float32"),
          };

          setModelReady(true);
          return;
        }
      } catch (e) {
        console.error(e);
        if (!cancelled) setModelError(e?.message ?? String(e));
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [selectedModel]);

  // Node finder (safe for both nodes-as-array and nodes-as-object)
  const dumpCnnNodes = async () => {
    try {
      if (!cnnRef.current) return;

      const graph = cnnRef.current.executor?.graph;
      const nodes = graph?.nodes;

      const names = Array.isArray(nodes) ? nodes.map((n) => n.name) : Object.keys(nodes || {});

      const interesting = names.filter((n) =>
        /conv|dense|relu|matmul|bias|reshape|flatten|pool/i.test(n)
      );

      console.log("=== CNN Graph Nodes (all) ===", names);
      console.log("=== CNN Graph Nodes (filtered) ===", interesting);

      alert("Printed node names to console (DevTools).");
    } catch (e) {
      console.error(e);
      alert("Failed to dump nodes. Check console.");
    }
  };

  const predict = useCallback(async () => {
    try {
      if (!modelReady) return;
      if (!canvasRef.current) return;

      const raw = preprocessCanvas(canvasRef.current);

      // Default embedding: raw pixels
      lastEmbeddingRef.current = Array.from(raw);

      // ✅ IMPORTANT: call the async IIFE with () so probs is an array, not a Promise
      const probs = await (async () => {
        if (selectedModel === "pretrained-cnn") {
          if (!cnnRef.current) return Array(10).fill(0);

          const input = tf.tensor4d(raw, [1, 28, 28, 1], "float32");
          const outT = cnnRef.current.predict(input);
          const out = Array.isArray(outT) ? outT[0] : outT;

          const arr = Array.from(await out.data());

          out.dispose?.();
          input.dispose();

          return arr;
        }

        if (selectedModel === "logreg") {
          if (!logregRef.current) return Array(10).fill(0);

          const x = tf.tensor2d(raw, [1, 784], "float32");
          const logits = x.matMul(logregRef.current.W).add(logregRef.current.b);
          const p = tf.softmax(logits);

          const arr = Array.from(await p.data());

          // cleanup
          p.dispose();
          logits.dispose();
          x.dispose();

          return arr;
        }

        return Array(10).fill(0);
      })();

      // Optional: compute CNN embedding for PCA when selected
      if (
        embeddingSource === "cnn-penultimate" &&
        selectedModel === "pretrained-cnn" &&
        cnnRef.current &&
        cnnEmbeddingOutName
      ) {
        const input4d = tf.tensor4d(raw, [1, 28, 28, 1], "float32");
        try {
          const emb = await extractCnnEmbedding(cnnRef.current, input4d, cnnEmbeddingOutName);
          lastEmbeddingRef.current = Array.from(emb);
          console.log("CNN emb len:", emb.length); // remove later if you want
        } finally {
          input4d.dispose();
        }
      }

      setPredictions(topK(probs, 3));

      const emb = lastEmbeddingRef.current;
      if (
        emb &&
        pcaState.components?.length === 2 &&
        pcaState.mean?.length &&
        emb.length === pcaState.mean.length
      ) {
        setCurrentPoint(projectWithPCA(emb, pcaState.mean, pcaState.components));
      } else {
        setCurrentPoint(null);
      }
    } catch (e) {
      console.error(e);
      setModelError(e?.message ?? String(e));
    }
  }, [modelReady, selectedModel, embeddingSource, cnnEmbeddingOutName, pcaState]);

  const addSample = () => {
    const emb = lastEmbeddingRef.current;
    if (!emb) return;

    setSamples((prev) => {
      const next = prev.length >= 300 ? prev.slice(1) : prev.slice();
      next.push({ embedding: emb });
      return next;
    });
  };

  const runPca = async () => {
    try {
      setPcaError("");
      if (samples.length < 2) return;

      const dim = samples[0]?.embedding?.length ?? 0;
      if (!dim || samples.some((s) => s.embedding.length !== dim)) {
        throw new Error("Samples have mixed embedding sizes. Clear samples and re-add using one embedding source.");
      }

      setPcaRunning(true);
      await new Promise((r) => setTimeout(r, 0));

      const embs = samples.map((s) => s.embedding);
      const res = await runPCA2D(embs);

      setPcaState(res);

      const emb = lastEmbeddingRef.current;
      if (emb && emb.length === res.mean.length) {
        setCurrentPoint(projectWithPCA(emb, res.mean, res.components));
      } else {
        setCurrentPoint(null);
      }
    } catch (e) {
      console.error(e);
      setPcaError(e?.message ?? String(e));
    } finally {
      setPcaRunning(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0b0b10", color: "white", padding: 24, boxSizing: "border-box" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto", display: "grid", gap: 24 }}>
        <h1 style={{ margin: 0, fontSize: 32 }}>Doodle Digit Classifier</h1>

        <div style={{ display: "grid", gridTemplateColumns: "minmax(320px, 520px) 1fr", gap: 24, alignItems: "start" }}>
          <div
            style={{
              padding: 16,
              borderRadius: 18,
              border: "1px solid rgba(255,255,255,0.12)",
              background: "rgba(255,255,255,0.03)",
              display: "grid",
              gap: 12,
            }}
          >
            <DoodleCanvas onCanvasReady={handleCanvasReady} onStrokeEnd={predict} />

            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <button onClick={addSample} style={{ padding: "10px 12px", borderRadius: 12, border: 0, cursor: "pointer" }}>
                Add sample
              </button>

              <button
                onClick={runPca}
                disabled={samples.length < 2 || pcaRunning}
                style={{
                  padding: "10px 12px",
                  borderRadius: 12,
                  border: 0,
                  cursor: samples.length < 2 || pcaRunning ? "not-allowed" : "pointer",
                  opacity: samples.length < 2 || pcaRunning ? 0.6 : 1,
                }}
              >
                {pcaRunning ? "Running…" : "Run PCA"}
              </button>

              <div style={{ alignSelf: "center", fontSize: 12, opacity: 0.8 }}>Samples: {samples.length}</div>

              <button
                onClick={dumpCnnNodes}
                disabled={selectedModel !== "pretrained-cnn" || !modelReady}
                style={{
                  padding: "10px 12px",
                  borderRadius: 12,
                  border: 0,
                  cursor: selectedModel !== "pretrained-cnn" || !modelReady ? "not-allowed" : "pointer",
                  opacity: selectedModel !== "pretrained-cnn" || !modelReady ? 0.6 : 1,
                }}
              >
                Find CNN embedding node
              </button>
            </div>

            {pcaError ? <div style={{ color: "#ff6b6b", fontSize: 13 }}>{pcaError}</div> : null}
          </div>

          <div style={{ display: "grid", gap: 24 }}>
            <div
              style={{
                padding: 16,
                borderRadius: 18,
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(255,255,255,0.03)",
                display: "grid",
                alignContent: "start",
                gap: 16,
              }}
            >
              <h2 style={{ margin: 0 }}>Predictions</h2>

              <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                <div style={{ display: "grid", gap: 6, minWidth: 240 }}>
                  <label style={{ fontSize: 12, opacity: 0.75 }}>Model</label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    style={{
                      padding: "8px 10px",
                      borderRadius: 10,
                      border: "1px solid rgba(255,255,255,0.18)",
                      background: "rgba(255,255,255,0.05)",
                      color: "white",
                      outline: "none",
                    }}
                  >
                    <option value="pretrained-cnn">Pretrained CNN</option>
                    <option value="logreg">Logistic Regression</option>
                  </select>
                </div>

                <div style={{ display: "grid", gap: 6, minWidth: 240 }}>
                  <label style={{ fontSize: 12, opacity: 0.75 }}>Embedding source</label>
                  <select
                    value={embeddingSource}
                    onChange={(e) => setEmbeddingSource(e.target.value)}
                    style={{
                      padding: "8px 10px",
                      borderRadius: 10,
                      border: "1px solid rgba(255,255,255,0.18)",
                      background: "rgba(255,255,255,0.05)",
                      color: "white",
                      outline: "none",
                    }}
                  >
                    <option value="raw">raw pixels</option>
                    <option value="cnn-penultimate" disabled={selectedModel !== "pretrained-cnn" || !cnnEmbeddingOutName}>
                      CNN penultimate layer
                    </option>
                  </select>

                  {selectedModel === "pretrained-cnn" && modelReady && !cnnEmbeddingOutName ? (
                    <div style={{ fontSize: 12, opacity: 0.75 }}>(CNN embedding node not found — use “Find CNN embedding node”)</div>
                  ) : null}
                </div>
              </div>

              {modelError ? (
                <div style={{ color: "#ff6b6b", fontSize: 13, lineHeight: 1.35 }}>
                  {modelError}
                  {lastTriedUrl ? (
                    <div style={{ marginTop: 8, opacity: 0.85 }}>
                      Tried:{" "}
                      <a href={lastTriedUrl} target="_blank" rel="noreferrer" style={{ color: "white" }}>
                        {lastTriedUrl}
                      </a>
                    </div>
                  ) : null}
                </div>
              ) : !modelReady ? (
                <div style={{ opacity: 0.7 }}>Loading model…</div>
              ) : modelWarning ? (
                <div style={{ color: "#ffd166", fontSize: 13 }}>{modelWarning}</div>
              ) : null}

              {!modelError && modelReady && predictions.length > 0 ? (
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
              ) : !modelError && modelReady ? (
                <div style={{ opacity: 0.7 }}>Draw a digit (0–9)</div>
              ) : null}
            </div>

            <div
              style={{
                padding: 16,
                borderRadius: 18,
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(255,255,255,0.03)",
                display: "grid",
                gap: 12,
              }}
            >
              <h2 style={{ margin: 0 }}>PCA</h2>
              <PcaPlot points={pcaState.points2d} highlight={currentPoint} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}