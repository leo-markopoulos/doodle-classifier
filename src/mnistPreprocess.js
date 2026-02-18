export function canvasToMnistTensor(canvas, tf) {
  const srcSize = 280;

  const tmp = document.createElement("canvas");
  tmp.width = srcSize;
  tmp.height = srcSize;
  const tctx = tmp.getContext("2d");

  // black bg + copy drawing
  tctx.fillStyle = "black";
  tctx.fillRect(0, 0, srcSize, srcSize);
  tctx.drawImage(canvas, 0, 0, srcSize, srcSize);

  const img = tctx.getImageData(0, 0, srcSize, srcSize);
  const data = img.data;

  // find bbox of ink
  let minX = srcSize, minY = srcSize, maxX = 0, maxY = 0;
  const threshold = 20;

  for (let y = 0; y < srcSize; y++) {
    for (let x = 0; x < srcSize; x++) {
      const idx = (y * srcSize + x) * 4;
      const r = data[idx], g = data[idx + 1], b = data[idx + 2];
      const brightness = (r + g + b) / 3;
      if (brightness > threshold) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (minX > maxX || minY > maxY) return null;

  // pad
  const pad = 20;
  minX = Math.max(0, minX - pad);
  minY = Math.max(0, minY - pad);
  maxX = Math.min(srcSize - 1, maxX + pad);
  maxY = Math.min(srcSize - 1, maxY + pad);

  const w = maxX - minX + 1;
  const h = maxY - minY + 1;

  // crop
  const crop = document.createElement("canvas");
  crop.width = w;
  crop.height = h;
  const cctx = crop.getContext("2d");
  cctx.putImageData(tctx.getImageData(minX, minY, w, h), 0, 0);

  // resize into 28x28 (fit into 20x20 then center)
  const out = document.createElement("canvas");
  out.width = 28;
  out.height = 28;
  const octx = out.getContext("2d");

  octx.fillStyle = "black";
  octx.fillRect(0, 0, 28, 28);

  const target = 20;
  const scale = Math.min(target / w, target / h);
  const newW = Math.max(1, Math.round(w * scale));
  const newH = Math.max(1, Math.round(h * scale));
  const dx = Math.floor((28 - newW) / 2);
  const dy = Math.floor((28 - newH) / 2);

  octx.drawImage(crop, 0, 0, w, h, dx, dy, newW, newH);

  // to tensor
  const outImg = octx.getImageData(0, 0, 28, 28).data;
  const arr = new Float32Array(28 * 28);

  for (let i = 0; i < 28 * 28; i++) {
    const r = outImg[i * 4];
    const g = outImg[i * 4 + 1];
    const b = outImg[i * 4 + 2];
    arr[i] = ((r + g + b) / 3) / 255.0; // 0..1
  }

  const tensor = tf.tensor4d(arr, [1, 28, 28, 1]);
  return { tensor, previewCanvas: out };
}
