import Fastify from "fastify";
import cors from "@fastify/cors";
import { z } from "zod";
import { embedBatch, getModelInfo, float32ToBase64 } from "./embeddings";

const PORT = Number(process.env.PORT ?? 8080);

const EmbeddingsRequest = z.object({
  // OpenAI-style fields
  model: z.string().optional(), // ignored; we use env MODEL_ID but we echo it back
  input: z.union([z.string(), z.array(z.string())]),

  // Extras for RAG ergonomics
  encoding_format: z.enum(["float", "base64"]).default("float"),
  mode: z.enum(["query", "passage"]).default("passage"),
  instruction: z.enum(["none", "e5"]).default("e5"),
  normalize: z.boolean().default(true)
});

type EmbeddingsRequest = z.infer<typeof EmbeddingsRequest>;

const app = Fastify({ logger: true });

await app.register(cors, { origin: true });

app.get("/healthz", async () => {
  const info = await getModelInfo();
  return { ok: true, ...info };
});

/**
 * POST /v1/embeddings
 * Body:
 * {
 *   input: string | string[],
 *   encoding_format?: 'float' | 'base64' (default 'float'),
 *   mode?: 'query' | 'passage' (default 'passage'),
 *   instruction?: 'none' | 'e5' (default 'e5'),
 *   normalize?: boolean (default true)
 * }
 *
 * Returns (OpenAI-like):
 * {
 *   "object": "list",
 *   "data": [{ "object": "embedding", "index": 0, "embedding": number[] | string(base64) }, ...],
 *   "model": "Xenova/bge-small-en-v1.5"
 * }
 */
app.post("/v1/embeddings", async (req, reply) => {
  const parse = EmbeddingsRequest.safeParse(req.body);
  if (!parse.success) {
    reply.code(400);
    return { error: "Bad Request", details: parse.error.flatten() };
  }
  const body: EmbeddingsRequest = parse.data;

  const inputs = Array.isArray(body.input) ? body.input : [body.input];

  const vectors = await embedBatch(inputs, {
    mode: body.mode,
    instruction: body.instruction,
    normalize: body.normalize
  });

  const data = vectors.map((v, i) => ({
    object: "embedding",
    index: i,
    embedding: body.encoding_format === "base64" ? float32ToBase64(v) : v
  }));

  const info = await getModelInfo();

  return {
    object: "list",
    data,
    model: info.model
    // Note: token usage not reported (no tokenizer here)
  };
});

// Friendly root
app.get("/", async () => {
  const info = await getModelInfo();
  return {
    name: "RAG Embedder",
    model: info.model,
    dim: info.dim,
    endpoints: ["/v1/embeddings", "/healthz"]
  };
});

app.listen({ port: PORT, host: "0.0.0.0" }).then(() => {
  app.log.info(`Embedding server listening on :${PORT}`);
});
