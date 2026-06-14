# Yuantiji: Offline Deployment Guide

Run the full search engine on a single Mac with no cloud APIs and no keys.

Embedding and query rewriting run locally via [ollama](https://ollama.com); rerank is not
available offline so performance will be lower than the online version.

Apple Silicon with **24 GB+ RAM** recommended: centroid index ~2 GB + embed model ~7 GB +
rewrite model ~9 GB resident. On an M3 pro 36 GB: ~12 s/query steady-state, ~40 s cold while models load. For smaller RAM, swap `GEN_LOCAL` in `server.py` to the weaker `gemma3:4b-it-qat` (~4 GB).

## 1. File list

```
yuantiji/
├── ui/
│   ├── server.py
│   ├── index.html
│   └── stats.html
└── emb/                      # qwen index — offline mode is qwen-only
    ├── centroids.f16         # 1.9 GB   254,940 × 4096 f16 problem vectors
    ├── centroid_uids.json    # 4.5 MB
    ├── meta.jsonl            # 248 MB   uid → title/url/source/snippet
    ├── rerank_docs.jsonl     # 204 MB   rewrite-1 display text
    ├── t1_docs.jsonl         #  98 MB   rewrite-2 display text
    ├── dup_groups.json       # 1.6 MB   duplicate collapse groups (optional but recommended)
    └── short_uids.json       #  64 KB   stub-statement skip list (optional but recommended)
```

~2.5 GB total.

## 2. Install

```bash
brew install ollama python      # or the ollama.com installer; python ≥ 3.10
pip3 install fastapi uvicorn numpy requests pydantic

ollama serve &                  # or just launch the Ollama app
ollama pull qwen3-embedding:8b-q4_K_M   # ~5 GB — embedding
ollama pull gemma3:12b-it-qat           # ~9 GB — query rewriting (same model as cloud)
```

## 3. Run

```bash
cd yuantiji/ui
python3 server.py
```

Defaults are auto-detected: `emb/` next to `ui/` becomes the data root, and with no `.env`
present the server runs in offline (local ollama) mode.

Open <http://localhost:8000>. Startup loads the index (~30 s) then warms up the embed model
(first ollama load can take a minute; later starts are fast).

| env var      | default                 | meaning                                          |
|--------------|-------------------------|--------------------------------------------------|
| `YT_BACKEND` | auto (`local` if no `.env`) | force `cloud` / `local`                      |
| `YT_ROOT`    | auto (dir containing `ui/`) | dir containing `emb/`                        |
| `YT_PORT`    | `8000`                  | listen port                                      |
| `YT_STATS`   | `$YT_ROOT/api_stats.jsonl` | usage log (counts/tokens only, never query text) |

`YT_EMB` is ignored offline — local mode forces the qwen index (gemini embeddings require
the cloud API).

## 4. What the frontend learns from the backend

`GET /api/health` reports everything the UI adapts to:

```json
{"ok": true, "backend": "local", "emb": "qwen", "problems": 254940, "rerank": false,
 "models": {"embed": "qwen3-embedding:8b-q4_K_M", "rewrite": "gemma3:12b-it-qat", "rerank": null}}
```

The footer shows the problem count and embed model from this; `"rerank": false` hides the rerank button. A rerank request sent anyway is silently served as plain cosine order.

## 5. Notes & troubleshooting

- **Quality vs cloud**: rewriting uses the same gemma-3-12B as the cloud (and as the corpus build), so phrasing matches the index; retrieval is the same qwen index cloud qwen mode uses.
- **Latency**: ~12 s/query on an M3 Pro (two 12B rewrites + one embed); the in-memory query cache makes repeats and rerank-style follow-ups instant. The embed call caps `num_ctx` at 8192 — ollama's 32k default balloons the embedder from ~7 GB to ~10 GB for nothing.
- **500 errors / "embed: …Connection refused"**: ollama isn't running — `ollama serve`.
- **Stalls or swapping**: not enough RAM for both models; skip pulling the rewrite model — the server detects it's missing, the UI hides the rewrite toggle, and queries embed the raw statement instead.
- **Privacy**: nothing leaves the machine; the stats log contains timestamps, flags and token counts only.
