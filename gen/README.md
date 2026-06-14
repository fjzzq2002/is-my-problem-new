# Data & index pipeline of v2

The pipeline has three stages. All scripts read `YT_ROOT` (default `.`) as the data root and expect a `.env` there with `OPENROUTER_KEY` and `DEEPINFRA_KEY`. Bring your own `problems.jsonl` (schema below) and stages 2–3 reproduce everything the server loads.

## Stage 1 — corpus (reference only, not shipped)

We gather statements from 60+ online judges, and perform the following cleaning pipeline.

- **Statement text recovery**, in priority order per problem: HTML statement → PDF text layer (PyMuPDF, lossless reading order) → vision OCR for image/broken-CID PDFs, `.doc/.docx/.html` attachment pulls (macOS textutil). Every candidate text was gated by a single garble detector (control-chars / mojibake / dingbats / script-coherence) and ones not passing the gate gets OCR.
- **Combined-contest PDFs** (Gym training contests, one PDF for the whole contest) were sliced per-problem with an embedded "focus on problem X" reminder.
- **Formula images** (Szkopuł/SGU) were OCR'd to LaTeX per-image.

The output is **`problems.jsonl`**, one JSON object per line:

```json
{"uid": "CodeForces/1784C", "title": "...", "url": "https://...", "source": "CodeForces",
 "statement": "full statement text", "statement_kind": "text|text+media|pdf_text"}
```

`uid` must be `Source/ProblemId` and unique (later duplicate-uid lines are ignored).

## Stage 2 — `rewrites/`

`gen_rewrites.py` normalizes phrasing: every problem gets **4 rewrites** with gemma-3-12b-it (DeepInfra): 2 prompt templates (T0 faithful / T1 succinct, inlined in the script) × 2 samples (temp 0 + temp 1). Async, append-only JSONL cached by statement hash.

```bash
FRACTION=1.0 CONCURRENCY=200 TEMPS=0,1 BACKEND=deepinfra python3 gen_rewrites.py
```

Cost/time for 255k problems: ~$60, ~10h at concurrency 200.

## Stage 3 — `indexes/`

Run in this order; everything lands in `$YT_ROOT/emb*/` and is exactly what `ui/server.py` loads.

| script | output | notes |
|---|---|---|
| `embed_rewrites.py` | `emb/{vectors.f16, order.jsonl, done.u8}` | qwen3-embedding-8b via OpenRouter, 4096-dim; crash-safe/resumable (frozen order + done-bitmap). ~$25 / 1M rewrites |
| `embed_rewrites_gemini.py` | `emb_gemini/…` | gemini-embedding-001, 3072-dim, no instruct prefix. ~$27, ~70 min at conc 24. Prod uses gemini (`YT_EMB=gemini`); offline/local serving uses qwen |
| `build_centroids[_gemini].py` | `emb*/centroids.f16 + centroid_uids.json` | problem vector = L2-normalized mean of its 4 rewrite vectors (centroid beat max-over-4 in ablation) |
| `build_meta.py` | `emb/meta.jsonl` | uid → title/url/source/snippet for the UI |
| `build_t_docs.py` | `emb/rerank_docs.jsonl, t1_docs.jsonl` | per-problem temp-0 rewrite per template (display + rerank docs); statement-hash-matched against the cache |
| `build_dup_groups.py` | `emb/dup_groups.json` | serve-time duplicate collapse. Byte-exact only: statement md5 + title-stripped/whitespace-stripped content md5. |
| `build_short_uids.py` | `emb/short_uids.json` | stub statements (<20 chars after digit/whitespace strip) the server can skip |

## Ablations

We performed a quatitative ablation for most of our design choice. Qualitative ablation: hold out N problems, generate fresh gpt-4.1-mini "user phrasings" of each (never shown to the index), and measure the rank of the true problem in the centroid index (R@k / MRR).

**Embedder — gemini-embedding-001 beats qwen3-embedding-8b.** 1000 held-out queries (500 problems × 2 phrasings). "paired" averages the embeddings of both phrasings of a query, which is what the server does with its two query rewrites:

| | R@1 | R@5 | R@10 | R@50 | MRR |
|---|---|---|---|---|---|
| qwen single | .609 | .922 | .965 | .985 | .742 |
| qwen paired | .632 | .944 | .974 | .988 | .762 |
| gemini single | .657 | .951 | .981 | .989 | .781 |
| gemini paired | **.676** | **.962** | **.986** | **.994** | **.794** |

Gemini wins every metric, embeds in ~0.3 s vs multi-second qwen (and the qwen providers TPM-throttled under load), and costs about the same to index (~$27). Earlier, qwen3-8b vs voyage-large-2-instruct was a tie at small N — the gap only shows at eval scale.

**Index representation — one centroid per problem beats keeping all 4 rewrite vectors.**
Scoring a problem by max-over-its-4-vectors (or max-over-2) lost to the simple L2-normalized mean, while costing 4× the index size and search time. Averaging seems to cancel per-rewrite phrasing noise the same way it does on the query side.

**Rewrite sampling — temp {0, 1} per template.** Per template we keep one greedy (temp 0) anchor and one diverse (temp 1) sample. All-greedy duplicates phrasings (the two samples collapse); all-hot drifts. The mixed pair gave the best doc centroid.

**A third, more aggressive template didn't help.** A "strip to the algorithmic core" T2 (e.g. reducing everything to "knapsack problem"-style abstracts) was tested as a doc representation and did not separate problems better than T0+T1 — past a point, abstraction deletes the details that distinguish near-neighbors. Two templates is the sweet spot we kept.

**Final sanity check:** We gathered a small set of 24 real human Chinese 简要题意 queries for final sanity check. The top hit is usually either that problem or a duplicate in some other OJs. We estimate hit-1 to be >80%.
