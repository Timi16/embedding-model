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
  // Treat as a possible Tensor from transformers.js
  const maybe = output as Partial<Tensor> & { data?: Float32Array; dims?: number[] };

  // Case 1: Tensor with dims + data (common path)
  if (maybe && maybe.data && Array.isArray(maybe.dims)) {
    const dims = maybe.dims;
    // avoid .at(): use classic indexing
    const n = Math.max(1, dims[0] ?? 1);
    const d = (dims.length > 1 ? (dims[dims.length - 1] ?? maybe.data.length) : maybe.data.length);
    modelDim = d;

    const flat = Array.from(maybe.data);
    const rows: number[][] = new Array(n);
    for (let i = 0; i < n; i++) {
      rows[i] = flat.slice(i * d, (i + 1) * d);
    }
    return rows;
  }

  // Case 2: Single vector as Float32Array
  if (output instanceof Float32Array) {
    modelDim = output.length;
    return [Array.from(output)];
  }

  // Case 3: Array of vectors or objects with .data
  if (Array.isArray(output)) {
    const rows: number[][] = (output as any[]).map((v: any) => {
      if (v instanceof Float32Array) return Array.from(v);
      if (v?.data instanceof Float32Array) return Array.from(v.data);
      // assume number[]-like
      return (v as number[]).map((x) => Number(x));
    });
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
