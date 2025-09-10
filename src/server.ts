import express, { Request, Response } from "express";
import cors from "cors";
import { z } from "zod";
import type { AddressInfo } from "node:net";
import { embedBatch, getModelInfo, float32ToBase64 } from "./embeddings";

// ---- Config ----
// Default to port 4916. To use a random free port: set PORT=0 or RANDOM_PORT=1
const HOST = "0.0.0.0";
const DEFAULT_PORT = 4916;
const envPort = process.env.PORT;
const wantRandom = process.env.RANDOM_PORT === "1" || envPort === "0";
const PORT = wantRandom ? 0 : Number(envPort ?? DEFAULT_PORT);

// ---- Schema ----
const EmbeddingsRequest = z.object({
  model: z.string().optional(),
  input: z.union([z.string(), z.array(z.string())]),
  encoding_format: z.enum(["float", "base64"]).default("float"),
  mode: z.enum(["query", "passage"]).default("passage"),
  instruction: z.enum(["none", "e5"]).default("e5"),
  normalize: z.boolean().default(true),
});
type EmbeddingsRequest = z.infer<typeof EmbeddingsRequest>;

// ---- App ----
const app = express();
app.disable("x-powered-by");
app.use(cors({ origin: true }));
app.use(express.json({ limit: "2mb" })); // adjust if needed

app.get("/healthz", async (_req: Request, res: Response) => {
  try {
    const info = await getModelInfo();
    res.json({ ok: true, ...info });
  } catch (err) {
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.post("/v1/embeddings", async (req: Request, res: Response) => {
  const parse = EmbeddingsRequest.safeParse(req.body);
  if (!parse.success) {
    return res.status(400).json({ error: "Bad Request", details: parse.error.flatten() });
  }
  const body: EmbeddingsRequest = parse.data;
  const inputs = Array.isArray(body.input) ? body.input : [body.input];

  try {
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
    res.json({ object: "list", data, model: info.model });
  } catch (err) {
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.get("/", async (_req: Request, res: Response) => {
  try {
    const info = await getModelInfo();
    res.json({
      name: "RAG Embedder",
      model: info.model,
      dim: info.dim,
      endpoints: ["/v1/embeddings", "/healthz"],
    });
  } catch {
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// ---- Start server & log actual port ----
const server = app.listen(PORT, HOST, () => {
  const addr = server.address() as AddressInfo | null;
  const actualPort = addr && typeof addr === "object" ? addr.port : envPort ?? DEFAULT_PORT;
  console.log(`Embedding server listening on http://${HOST}:${actualPort}`);
});

// ---- Graceful shutdown ----
const shutdown = (signal: string) => {
  console.log(`${signal} received. Closing server...`);
  server.close((err) => {
    if (err) {
      console.error("Error during shutdown", err);
      process.exit(1);
    }
    process.exit(0);
  });
};
process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
