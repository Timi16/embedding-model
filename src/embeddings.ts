import { pipeline, type Tensor } from "@xenova/transformers";

/**
 * Minimal embedders with instruction-aware prefixes (E5/BGE style)
 * Defaults:
 *  - MODEL_ID: Xenova/bge-small-en-v1.5  (good English baseline)
 *  - INSTRUCTION: "e5" (adds "query:" / "passage:" prefixes)
 */
const MODEL_ID = process.env.MODEL_ID ?? "Xenova/bge-small-en-v1.5";

type Mode = "query" | "passage";
type Instruction = "none" | "e5";

let extractorPromise: Promise<any> | null = null;
let modelDim: number | null = null;

function withPrefix(texts: string[], mode: Mode, instruction: Instruction) {
  if (instruction === "e5") {
    const prefix = mode === "query" ? "query: " : "passage: ";
    return texts.map((t) => prefix + t);
  }
  return texts;
}

/**
 * Loads (and memoizes) the feature-extraction pipeline.
 * We rely on built-in mean pooling + normalize options.
 */
async function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = pipeline("feature-extraction", MODEL_ID);
  }
  return extractorPromise;
}

/**
 * Convert the pipeline output to number[][] for a batch.
 * Handles single/tensor/batch outputs robustly.
 */
function toMatrix(output: any): number[][] {
  // Newer versions may return a Tensor with `data` (Float32Array) and `dims`
  const maybeTensor = output as Tensor & { data?: Float32Array; dims?: number[] };

  if (maybeTensor && maybeTensor.data && Array.isArray(maybeTensor.dims)) {
    const [n, d] = maybeTensor.dims.length === 2 ? maybeTensor.dims : [1, maybeTensor.dims.at(-1) ?? maybeTensor.data.length];
    modelDim = d ?? modelDim;
    const arr = Array.from(maybeTensor.data);
    const rows: number[][] = [];
    for (let i = 0; i < n; i++) rows.push(arr.slice(i * (d ?? 0), (i + 1) * (d ?? 0)));
    return rows;
  }

  // Single vector as Float32Array
  if (output instanceof Float32Array) {
    modelDim = output.length;
    return [Array.from(output)];
  }

  // Array of vectors (number[] or Float32Array)
  if (Array.isArray(output)) {
    const rows = output.map((v: any) => Array.from(v?.data ?? v));
    modelDim = rows[0]?.length ?? modelDim;
    return rows;
  }

  throw new Error("Unexpected embedding output shape");
}

export async function embedBatch(
  texts: string[],
  opts?: { mode?: Mode; instruction?: Instruction; normalize?: boolean }
): Promise<number[][]> {
  if (texts.length === 0) return [];
  const mode = opts?.mode ?? "passage";
  const instruction = opts?.instruction ?? "e5";
  const normalize = opts?.normalize ?? true;

  const extractor = await getExtractor();
  const prefixed = withPrefix(texts, mode, instruction);

  // Use built-in pooling & normalization from transformers.js
  const out = await extractor(prefixed, { pooling: "mean", normalize });
  return toMatrix(out);
}

export async function getModelInfo() {
  // Warm once to know dimension
  if (modelDim == null) {
    await embedBatch(["warmup"], { mode: "passage", instruction: "e5" });
  }
  return { model: MODEL_ID, dim: modelDim ?? null };
}

/** Utility: convert number[] -> base64(Float32Array) */
export function float32ToBase64(vec: number[]): string {
  const f = new Float32Array(vec);
  // Node Buffer view. Ensure we copy the underlying bytes correctly.
  return Buffer.from(f.buffer).toString("base64");
}
