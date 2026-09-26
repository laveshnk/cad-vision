/**
 * Copies the MediaPipe WASM runtime from node_modules into public/ and downloads
 * the hand_landmarker.task model so the app can run fully offline / self-hosted.
 *
 * Resilient by design: failures print a warning (runtime falls back to CDN URLs)
 * and never fail the surrounding npm script.
 */
import { cp, mkdir, stat, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, '..');
const WASM_SRC = path.join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const WASM_DST = path.join(root, 'public', 'mediapipe', 'wasm');
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const MODEL_DST = path.join(root, 'public', 'models', 'hand_landmarker.task');

async function copyWasm() {
  if (!existsSync(WASM_SRC)) {
    console.warn(`[assets] WASM source not found (${WASM_SRC}) — skipping copy.`);
    return;
  }
  await mkdir(path.dirname(WASM_DST), { recursive: true });
  await cp(WASM_SRC, WASM_DST, { recursive: true });
  console.log(`[assets] Copied MediaPipe WASM runtime -> ${path.relative(root, WASM_DST)}`);
}

async function downloadModel() {
  try {
    const existing = await stat(MODEL_DST);
    if (existing.size > 1024) {
      console.log(`[assets] Model already present -> ${path.relative(root, MODEL_DST)}`);
      return;
    }
  } catch {
    // not present — download below
  }
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await mkdir(path.dirname(MODEL_DST), { recursive: true });
  await writeFile(MODEL_DST, buf);
  console.log(`[assets] Downloaded hand_landmarker.task (${(buf.length / 1024).toFixed(0)} kB)`);
}

try {
  await copyWasm();
} catch (err) {
  console.warn(`[assets] WASM copy failed: ${err instanceof Error ? err.message : err}`);
}
try {
  await downloadModel();
} catch (err) {
  console.warn(
    `[assets] Model download failed (${err instanceof Error ? err.message : err}). ` +
      'The app will fall back to the CDN model URL at runtime.'
  );
}
