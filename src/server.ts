import Fastify from "fastify";
import cors from "@fastify/cors";
import { z } from "zod";
import type { AddressInfo } from "node:net";
import { embedBatch, getModelInfo, float32ToBase64 } from "./embeddings";

// ---- Config ----
// Use PORT=4916 by default. To use a random free port: set PORT=0 or RANDOM_PORT=1
const HOST =  "0.0.0.0";
const DEFAULT_PORT = 4916;
const envPort = process.env.PORT;
const wantRandom = process.env.RANDOM_PORT === "1" || envPort === "0";
const PORT = wantRandom ? 0 : Number(envPort ?? DEFAULT_PORT);

const EmbeddingsRequest = z.object({
  model: z.string().optional(),
  input: z.union([z.string(), z.array(z.string())]),
  encoding_format: z.enum(["float", "base64"]).default("float"),
  mode: z.enum(["query", "passage"]).default("passage"),
  instruction: z.enum(["none", "e5"]).default("e5"),
  normalize: z.boolean().default(true),
});
type EmbeddingsRequest = z.infer<typeof EmbeddingsRequest>;

const app = Fastify({ logger: true });
await app.register(cors, { origin: true });

app.get("/healthz", async () => {
  const info = await getModelInfo();
  return { ok: true, ...info };
});

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
    normalize: body.normalize,
  });

  const data = vectors.map((v, i) => ({
    object: "embedding",
    index: i,
    embedding: body.encoding_format === "base64" ? float32ToBase64(v) : v,
  }));

  const info = await getModelInfo();
  return { object: "list", data, model: info.model };
});

app.get("/", async () => {
  const info = await getModelInfo();
  return {
    name: "RAG Embedder",
    model: info.model,
    dim: info.dim,
    endpoints: ["/v1/embeddings", "/healthz"],
  };
});

// ---- Start server & log actual port ----
const addressString = await app.listen({ port: PORT, host: HOST });
// Note: when port=0, Fastify picks a free port. Get it from the underlying server:
const addr = app.server.address() as AddressInfo | null;
const actualPort = addr && typeof addr === "object" ? addr.port : new URL(addressString).port;

app.log.info({ url: `http://${HOST}:${actualPort}` }, `Embedding server listening`);

// ---- Graceful shutdown ----
const shutdown = async (signal: string) => {
  try {
    app.log.info(`${signal} received. Closing server...`);
    await app.close();
    process.exit(0);
  } catch (err) {
    app.log.error({ err }, "Error during shutdown");
    process.exit(1);
  }
};
process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
