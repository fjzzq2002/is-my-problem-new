# 原题机 v2 — Is my problem new?

Semantic search over 250k+ competitive programming problems: paste a statement (any
language, any phrasing), get the most similar known problems. v2 is a ground-up rebuild
of [is-my-problem-new](https://github.com/fjzzq2002/is-my-problem-new): new corpus
pipeline, LLM phrasing normalization, embedding centroid index, and a single-file
server + frontend.

## How it works

1. **Corpus** — statements from 60+ online judges, cleaned through a PDF/OCR/garble
   pipeline into one `problems.jsonl` (not shipped — see `gen/README.md`).
2. **Rewrites** — every problem gets 4 LLM rewrites (gemma-3-12b-it, 2 templates ×
   2 temps) that strip stories and normalize phrasing.
3. **Index** — each rewrite is embedded (gemini-embedding-001 in prod, qwen3-embedding-8b
   offline); a problem's vector is the centroid of its 4 rewrite vectors.
4. **Serving** — `ui/server.py` (FastAPI, one file) rewrites + embeds your query, walks the
   centroid index by cosine, collapses exact duplicates, and optionally reranks the top 100
   with Qwen3-Reranker-8B. `ui/index.html` (one file) is the whole frontend.

## Layout

```
ui/        server + frontend (server.py, index.html, stats.html, fonts/)
gen/       data & index pipeline + docs on the corpus stage and ablations
OFFLINE.md run the whole thing on a Mac with no cloud APIs (ollama)
```

## Running

The server needs the index artifacts in `$YT_ROOT/emb*/` — build them with `gen/`
(your own corpus; see `gen/README.md` for the `problems.jsonl` contract) or obtain a
prebuilt index bundle.

**Cloud mode** (used by the public site): put `OPENROUTER_KEY` and `DEEPINFRA_KEY` in
`$YT_ROOT/.env`, then

```bash
pip3 install -r requirements.txt
cd ui && python3 server.py        # http://localhost:8000
```

**Offline mode**: no keys, models run locally via ollama — see [OFFLINE.md](OFFLINE.md).
Mode is auto-detected: `.env` present → cloud, absent → local.

`/api/health` reports the backend, models, and problem count; the UI adapts (hides
rerank/rewrite toggles when the backend lacks them). `/stats` shows usage counts and cost —
the stats log records counts and token totals only, never query text.

## Credits

UI fonts are [IBM Plex](https://github.com/IBM/plex) Sans + Mono, © IBM Corp.,
bundled under the [SIL Open Font License 1.1](ui/fonts/OFL.txt). Everything else is
MIT (see [LICENSE](LICENSE)).
