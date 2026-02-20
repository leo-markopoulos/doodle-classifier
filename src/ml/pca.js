// PCA via Gram matrix (X X^T) so it's fast in-browser when N (samples) is small.
// embeddings: Array<Array<number>> length N, each length D
// returns:
//   points2d: Array<[number, number]> length N
//   mean: Array<number> length D
//   components: Array<Array<number>> shape [2][D]  (so you can project new points)
// Notes:
//   - This is deterministic-ish but power iteration can vary slightly.
//   - Works great for N up to a few hundred and D=784.

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function norm(v) {
  return Math.sqrt(dot(v, v)) + 1e-12;
}

function normalize(v) {
  const n = norm(v);
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

function matVecMul(A, v) {
  const n = A.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const row = A[i];
    for (let j = 0; j < n; j++) s += row[j] * v[j];
    out[i] = s;
  }
  return out;
}

function powerIterationTop1(A, iters = 80) {
  const n = A.length;
  let v = new Float32Array(n);
  // init
  for (let i = 0; i < n; i++) v[i] = (Math.random() - 0.5);
  normalize(v);

  for (let t = 0; t < iters; t++) {
    const Av = matVecMul(A, v);
    v = Av;
    normalize(v);
  }

  // Rayleigh quotient for eigenvalue
  const Av = matVecMul(A, v);
  const lambda = dot(v, Av);

  return { v: Array.from(v), lambda };
}

function deflate(A, v, lambda) {
  const n = A.length;
  // A <- A - lambda * v v^T
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      A[i][j] -= lambda * v[i] * v[j];
    }
  }
}

// Compute mean and centered matrix (as arrays)
function centerEmbeddings(embeddings) {
  const N = embeddings.length;
  const D = embeddings[0].length;

  const mean = new Float32Array(D);
  for (let i = 0; i < N; i++) {
    const x = embeddings[i];
    for (let d = 0; d < D; d++) mean[d] += x[d];
  }
  for (let d = 0; d < D; d++) mean[d] /= N;

  const Xc = new Array(N);
  for (let i = 0; i < N; i++) {
    const x = embeddings[i];
    const row = new Float32Array(D);
    for (let d = 0; d < D; d++) row[d] = x[d] - mean[d];
    Xc[i] = row;
  }

  return { mean: Array.from(mean), Xc };
}

// Gram matrix G = Xc Xc^T (N x N)
function gramMatrix(Xc) {
  const N = Xc.length;
  const G = new Array(N);
  for (let i = 0; i < N; i++) {
    G[i] = new Float32Array(N);
  }

  for (let i = 0; i < N; i++) {
    G[i][i] = dot(Xc[i], Xc[i]);
    for (let j = i + 1; j < N; j++) {
      const v = dot(Xc[i], Xc[j]);
      G[i][j] = v;
      G[j][i] = v;
    }
  }

  // convert to arrays for easier deflation
  const A = new Array(N);
  for (let i = 0; i < N; i++) A[i] = Array.from(G[i]);
  return A;
}

// components (2 x D) = (U^T Xc) / S
function computeComponents(Xc, U2, S2) {
  const N = Xc.length;
  const D = Xc[0].length;
  const comps = [new Float32Array(D), new Float32Array(D)];

  for (let k = 0; k < 2; k++) {
    const invS = 1.0 / (S2[k] + 1e-12);
    for (let d = 0; d < D; d++) {
      let s = 0;
      for (let i = 0; i < N; i++) {
        s += U2[k][i] * Xc[i][d];
      }
      comps[k][d] = s * invS;
    }
  }

  return [Array.from(comps[0]), Array.from(comps[1])];
}

export async function runPCA2D(embeddings) {
  if (!embeddings || embeddings.length < 2) {
    return { points2d: [], mean: [], components: [] };
  }

  const N = embeddings.length;
  const D = embeddings[0].length;

  // Keep N reasonable so it stays snappy
  if (N > 400) {
    embeddings = embeddings.slice(N - 400);
  }

  const { mean, Xc } = centerEmbeddings(embeddings);
  const A = gramMatrix(Xc);

  // top1
  const e1 = powerIterationTop1(A, 90);
  deflate(A, e1.v, e1.lambda);

  // top2
  const e2 = powerIterationTop1(A, 90);

  const lambda1 = Math.max(0, e1.lambda);
  const lambda2 = Math.max(0, e2.lambda);
  const s1 = Math.sqrt(lambda1);
  const s2 = Math.sqrt(lambda2);

  const U1 = e1.v;
  const U2 = e2.v;

  // Points Z = U * S
  const points2d = [];
  for (let i = 0; i < embeddings.length; i++) {
    points2d.push([U1[i] * s1, U2[i] * s2]);
  }

  const components = computeComponents(Xc, [U1, U2], [s1, s2]);

  return {
    points2d,
    mean,
    components, // [2][D]
  };
}

export function projectWithPCA(embedding, mean, components) {
  if (!embedding || !mean || !components || components.length !== 2) return null;

  const D = embedding.length;
  let z0 = 0;
  let z1 = 0;

  for (let i = 0; i < D; i++) {
    const x = embedding[i] - (mean[i] ?? 0);
    z0 += x * (components[0][i] ?? 0);
    z1 += x * (components[1][i] ?? 0);
  }

  return [z0, z1];
}
