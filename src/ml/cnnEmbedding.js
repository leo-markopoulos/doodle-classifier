// src/ml/cnnEmbedding.js
import * as tf from "@tensorflow/tfjs";

export const DEFAULT_EMBEDDING_NODE =
  "StatefulPartitionedCall/sequential_1/dense_1/Relu";

export function resolveEmbeddingOutputName(graphModel) {
  const preferred = "StatefulPartitionedCall/sequential_1/dense_1/Relu";

  const nodes = graphModel?.executor?.graph?.nodes;
  if (!nodes) return null;

  // Case A: nodes is an ARRAY
  if (Array.isArray(nodes)) {
    if (nodes.some((n) => n.name === preferred)) return preferred;
    const fb = nodes.find((n) => n.name?.includes("dense_1/Relu"))?.name;
    return fb ?? null;
  }

  // Case B: nodes is an OBJECT keyed by node name
  // Fast path: exact key exists
  if (nodes[preferred]) return preferred;

  // Fallback: find a key that contains dense_1/Relu
  const keys = Object.keys(nodes);
  const fb = keys.find((k) => k.includes("dense_1/Relu"));
  return fb ?? null;
}

/**
 * Extract a 1D embedding vector from a GraphModel.
 * @param {tf.GraphModel} graphModel
 * @param {tf.Tensor4D} input4d  shape [1, H, W, C] (usually [1,28,28,1])
 * @param {string} outputName   node name for embedding output
 * @returns {Promise<Float32Array>}
 */
export async function extractCnnEmbedding(graphModel, input4d, outputName) {
  if (!graphModel) throw new Error("CNN model not loaded.");
  if (!outputName) throw new Error("CNN embedding node not resolved.");

  let y; // Tensor or Tensor[]
  let t; // Tensor
  let squeezed; // Tensor

  try {
    y = await graphModel.executeAsync(input4d, outputName);
    t = Array.isArray(y) ? y[0] : y;

    // Flatten [1, D] -> [D]
    squeezed = t.squeeze();

    // Pull data to CPU
    const data = await squeezed.data();
    return data; // Float32Array
  } finally {
    // Always dispose in finally, even if an error happens
    if (squeezed) squeezed.dispose();

    // If y was an array, dispose each; otherwise dispose single tensor
    if (Array.isArray(y)) y.forEach((tt) => tt?.dispose?.());
    else y?.dispose?.();
  }
}