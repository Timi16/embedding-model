import express from "express";
import cors from "cors";
import { z } from "zod";
import { embedBatch, getModelInfo, float32ToBase64 } from "./embeddings.js";
// Fixed port
const PORT = 4916;
// Schema
const EmbeddingsRequest = z.object({
    model: z.string().optional(),
    input: z.union([z.string(), z.array(z.string())]),
    encoding_format: z.enum(["float", "base64"]).default("float"),
    mode: z.enum(["query", "passage"]).default("passage"),
    instruction: z.enum(["none", "e5"]).default("e5"),
    normalize: z.boolean().default(true),
});
const app = express();
app.use(cors());
app.use(express.json());
app.get("/healthz", async (_req, res) => {
    try {
        const info = await getModelInfo();
        res.json({ ok: true, ...info });
    }
    catch {
        res.status(500).json({ error: "Internal Server Error" });
    }
});
app.post("/v1/embeddings", async (req, res) => {
    const parse = EmbeddingsRequest.safeParse(req.body);
    if (!parse.success) {
        return res.status(400).json({ error: "Bad Request", details: parse.error.flatten() });
    }
    const body = parse.data;
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
    }
    catch {
        res.status(500).json({ error: "Internal Server Error" });
    }
});
app.get("/", async (_req, res) => {
    try {
        const info = await getModelInfo();
        res.json({
            name: "RAG Embedder",
            model: info.model,
            dim: info.dim,
            endpoints: ["/v1/embeddings", "/healthz"],
        });
    }
    catch {
        res.status(500).json({ error: "Internal Server Error" });
    }
});
// Start (no host arg)
app.listen(PORT, () => {
    console.log(`Embedding server listening on http://localhost:${PORT}`);
});
