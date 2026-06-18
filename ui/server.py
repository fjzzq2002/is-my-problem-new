"""yuantiji backend — one /api/search the SPA calls.

BACKEND=cloud : embed via OpenRouter, rewrite + rerank via DeepInfra. Needs $ROOT/.env with
                OPENROUTER_KEY and DEEPINFRA_KEY.
BACKEND=local : embed + rewrite via local ollama; no rerank. Fully offline, no keys.

Run:  python3 server.py        (serves http://localhost:8000)
Defaults are auto-detected: ROOT = the directory containing ui/ (override with YT_ROOT),
backend = cloud iff $ROOT/.env exists. Needs $ROOT/emb/{centroids.f16, centroid_uids.json,
meta.jsonl, rerank_docs.jsonl, t1_docs.jsonl} (+ optional dup_groups.json, short_uids.json).
"""
import os, re, json, time, threading, hashlib
import numpy as np, requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn

UI   = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("YT_ROOT") or os.path.dirname(UI)   # self-contained layout: emb/ next to ui/
EMB  = os.path.join(ROOT, "emb")
if not os.path.isdir(EMB):
    raise SystemExit(f"no index at {EMB} — set YT_ROOT to the directory that contains emb/")
# cloud needs API keys; no .env → assume an offline (local ollama) deployment
BACKEND = os.environ.get("YT_BACKEND") or ("cloud" if os.path.exists(os.path.join(ROOT, ".env")) else "local")
EMODEL  = os.environ.get("YT_EMB", "qwen")        # qwen | gemini — gemini won the 1000-query eval (R@1 67.6 vs 63.2)
RPOOL = 100  # rerank candidate pool

if BACKEND == "local": EMODEL = "qwen"   # offline serving is qwen-only (gemini embeds need the cloud API)

# ---- secrets / templates ----
try:   # .env is cloud-only; offline deployments have no keys
    for ln in open(os.path.join(ROOT, ".env")):
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.strip().split("=", 1); os.environ.setdefault(k, v.strip().strip("\"'"))
except FileNotFoundError:
    pass
ORK = os.environ.get("OPENROUTER_KEY", ""); DK = os.environ.get("DEEPINFRA_KEY", "")
# the production rewrite prompts — identical to what the corpus rewrites were generated with;
# optionally overridden by $ROOT/settings.json {"TEMPLATES": [t0, t1]}
TEMPLATES = [
    "I have the following competitive programming problem that I want to show someone else:\n\n=======\n[[ORIGINAL]]\n=======\n\n"
    "Strip off all the stories, legends, characters, backgrounds etc. from the statement while still enabling everyone to understand "
    "the problem. Also remove the name of the character if possible. This is to say, do not remove anything necessary to understand "
    "the full problem and one should feel safe to replace the original statement with your version of the statement. If it is not in "
    "English make it English. Provide the simplified statement directly without jargon. Use mathjax ($...$) for math. "
    "Start your response with \"Simplified statement:\".",
    "I have the following competitive programming problem that I want to show someone else:\n\n=======\n[[ORIGINAL]]\n=======\n\n"
    "Strip off all the stories, legends, characters, backgrounds, examples, well-known definitions etc. from the statement while "
    "still enabling everyone to understand the problem. Also remove the name of the character if applicable. If it is not in English "
    "translate it. Make it as succinct as possible while still being understandable. Try to avoid formulas and symbols. Abstract "
    "freely - for example, if the problem is about buying sushi, you can just phrase it as a knapsack problem. If necessary, "
    "mathjax ($...$) for math. Provide the *succinct* simplified statement directly without jargon. "
    "Start your response with \"Simplified statement:\".",
]
try:
    TEMPLATES = json.load(open(os.path.join(ROOT, "settings.json")))["TEMPLATES"]
except Exception:
    pass

EMB_LOCAL = "qwen3-embedding:8b-q4_K_M"; GEN_LOCAL = "gemma3:12b-it-qat"  # local mode is qwen-only; 12b matches the cloud rewriter
EMB_CLOUD = "google/gemini-embedding-001" if EMODEL == "gemini" else "qwen/qwen3-embedding-8b"
GEN_CLOUD = "google/gemma-3-12b-it"
OR_PROVIDER = None
RERANKER  = "Qwen/Qwen3-Reranker-8B"
RERANK_OK = BACKEND != "local"   # offline: no rerank (8B cross-encoder needs the cloud)
_rw_ok = {"t": 0.0, "v": True}
def rewrite_ok():
    """Offline: is the rewrite model actually pulled? Checked against ollama's tag list, cached 60s.
    Lets low-RAM deployments skip the rewrite model — queries degrade to raw-query embedding."""
    if BACKEND != "local": return True
    if time.time() - _rw_ok["t"] > 60:
        try:
            tags = sess.get("http://localhost:11434/api/tags", timeout=5).json()
            _rw_ok["v"] = any((m.get("name") or "").startswith(GEN_LOCAL) for m in tags.get("models", []))
        except Exception:
            _rw_ok["v"] = False
        _rw_ok["t"] = time.time()
    return _rw_ok["v"]
TASK = "Given a competitive programming problem, retrieve similar problems"
def instruct(q): return q if EMODEL == "gemini" else f"Instruct: {TASK}\nQuery:{q}"  # gemini takes raw text

# ---- index ----
print(f"[{BACKEND}/{EMODEL}] loading index…", flush=True)
CENT_D = os.path.join(ROOT, "emb_gemini") if EMODEL == "gemini" else EMB   # centroids are per-model; docs/meta shared
DIM = 3072 if EMODEL == "gemini" else 4096
uids = json.load(open(os.path.join(CENT_D, "centroid_uids.json"))); P = len(uids)
cent = np.memmap(os.path.join(CENT_D, "centroids.f16"), dtype=np.float16, mode="r", shape=(P, DIM))  # ~1.6-2GB, lazy
CHUNK = 8000  # rows cast f16->f32 per step; small so concurrent searches don't blow up RAM (50k*3072*4≈614MB each)
def cosine_all(qv):
    sims = np.empty(P, dtype=np.float32)
    for s in range(0, P, CHUNK):
        e = min(s + CHUNK, P); sims[s:e] = np.asarray(cent[s:e], dtype=np.float32) @ qv
    return sims
meta = {}
for ln in open(os.path.join(EMB, "meta.jsonl"), encoding="utf-8"):
    r = json.loads(ln); meta[r["uid"]] = r
rdoc = {}
for ln in open(os.path.join(EMB, "rerank_docs.jsonl"), encoding="utf-8"):
    r = json.loads(ln); rdoc[r["uid"]] = r["doc"]
t1doc = {}
for ln in open(os.path.join(EMB, "t1_docs.jsonl"), encoding="utf-8"):
    r = json.loads(ln); t1doc[r["uid"]] = r["doc"]
DUP = {}   # uid -> its exact-duplicate group (canonical first); byte-identical statements only
try:
    _groups = json.load(open(os.path.join(EMB, "dup_groups.json")))
    for _g in _groups:
        for _u in _g: DUP[_u] = _g
    print(f"dup collapse: {len(_groups)} groups / {len(DUP)} uids", flush=True)
except Exception as _e:
    print(f"dup collapse off ({_e})", flush=True)
SHORT = set()  # uids whose statement is <20 chars after stripping digits/whitespace — stubs like "download"
try:
    SHORT = set(json.load(open(os.path.join(EMB, "short_uids.json"))))
    print(f"short-statement skip: {len(SHORT)} uids", flush=True)
except Exception as _e:
    print(f"short-statement skip off ({_e})", flush=True)
print(f"loaded {P} centroids · {len(meta)} meta · {len(rdoc)} t0 · {len(t1doc)} t1", flush=True)
sess = requests.Session()

# ---- usage stats: one JSONL line per /api/search — counts/flags/tokens ONLY, never query text or results ----
STATS = os.environ.get("YT_STATS", os.path.join(ROOT, "api_stats.jsonl"))
_TL = threading.local()          # per-request accumulator (each request runs on one threadpool thread)
_slock = threading.Lock()
def _acc(**kw):
    u = getattr(_TL, "u", None)
    if u is None: return
    for k, v in kw.items():
        if v: u[k] = round(u.get(k, 0) + v, 6)
def _log_stats(rec):
    try:
        with _slock, open(STATS, "a") as f: f.write(json.dumps(rec) + "\n")
    except Exception:
        pass

# ---- engine ----
def embed_batch(texts):
    if BACKEND == "local":
        # native endpoint so we can cap num_ctx — default 32k context balloons the 4.7GB model to ~10GB resident
        r = sess.post("http://localhost:11434/api/embed",
                      json={"model": EMB_LOCAL, "input": texts, "options": {"num_ctx": 8192}}, timeout=180).json()
        if "embeddings" not in r: raise RuntimeError(f"embed: {str(r)[:200]}")
        _acc(emb_toks=r.get("prompt_eval_count", 0), emb_calls=1)
        E = np.array(r["embeddings"], dtype=np.float32)
        return E / np.linalg.norm(E, axis=1, keepdims=True)
    else:
        r = None
        for attempt in range(4):  # qwen/SiliconFlow TPM-limits under bursts (429) — back off, then let Nebius take it
            prov = OR_PROVIDER
            body = {"model": EMB_CLOUD, "input": texts}
            if prov: body["provider"] = prov
            r = sess.post("https://openrouter.ai/api/v1/embeddings", headers={"Authorization": f"Bearer {ORK}"},
                          json=body, timeout=120).json()
            if "data" in r: break
            msg = str((r.get("error") or {}))
            if "429" in msg or "rate" in msg.lower():
                print(f"embed 429, retry {attempt + 1}", flush=True); time.sleep(1.5 * (attempt + 1)); continue
            break
    if "data" not in r: raise RuntimeError(f"embed: {str(r)[:200]}")
    _acc(emb_toks=(r.get("usage") or {}).get("total_tokens", 0), emb_calls=1)
    E = np.array([e["embedding"] for e in r["data"]], dtype=np.float32)
    return E / np.linalg.norm(E, axis=1, keepdims=True)

def rewrite(query, t):
    tmpl = TEMPLATES[t].replace("[[ORIGINAL]]", query)
    if BACKEND == "local":
        body = {"model": GEN_LOCAL, "stream": False, "messages": [{"role": "user", "content": tmpl}],
                "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1024}}
        j = sess.post("http://localhost:11434/api/chat", json=body, timeout=300).json()
        if "error" in j: raise RuntimeError(f"rewrite: {str(j['error'])[:200]}")   # e.g. model not pulled — never embed ""
        _acc(rw_in=j.get("prompt_eval_count", 0), rw_out=j.get("eval_count", 0), rw_calls=1)
        txt = (j.get("message", {}) or {}).get("content") or ""
    else:
        body = {"model": GEN_CLOUD, "temperature": 0, "max_tokens": 1024, "messages": [{"role": "user", "content": tmpl}]}
        j = sess.post("https://api.deepinfra.com/v1/openai/chat/completions", headers={"Authorization": f"Bearer {DK}"},
                      json=body, timeout=120).json()
        _u = j.get("usage") or {}
        _acc(rw_in=_u.get("prompt_tokens", 0), rw_out=_u.get("completion_tokens", 0), rw_calls=1)
        txt = j["choices"][0]["message"]["content"] or ""
    return re.sub(r"^\s*Simplified statement:\s*", "", txt.strip()).strip()

def query_vec(query, do_rewrite):
    if do_rewrite:
        rws = [rewrite(query, 0), rewrite(query, 1)]
        E = embed_batch([instruct(rws[0]), instruct(rws[1])]); v = E.mean(0); v /= np.linalg.norm(v)
        return v, {"t0": rws[0], "t1": rws[1]}
    return embed_batch([instruct(query)])[0], None

QV_CACHE = {}; _qv_lock = threading.Lock(); QV_TTL = 600   # md5(query) -> (ts, qv, rewrites); rerank click reuses the search's vector
def query_vec_cached(query, do_rewrite):
    # privacy: key is md5 only — the raw query string is never stored; entries (incl. rewrite text) expire after QV_TTL
    key = hashlib.md5(f"{do_rewrite}|{EMODEL}|{query}".encode()).hexdigest()
    now = time.time()
    with _qv_lock:
        for k in [k for k, (ts, _, _) in QV_CACHE.items() if now - ts > QV_TTL]: del QV_CACHE[k]
        hit = QV_CACHE.get(key)
    if hit: return hit[1], hit[2], True
    v, rw = query_vec(query, do_rewrite)
    with _qv_lock:
        QV_CACHE[key] = (now, v, rw)
        if len(QV_CACHE) > 500:
            for k in sorted(QV_CACHE, key=lambda k: QV_CACHE[k][0])[:100]: del QV_CACHE[k]
    return v, rw, False

def rerank(query, cand_uids):
    docs = [(rdoc.get(u) or (meta.get(u, {}).get("snip")) or u)[:1200] for u in cand_uids]
    r = sess.post(f"https://api.deepinfra.com/v1/inference/{RERANKER}", headers={"Authorization": f"bearer {DK}"},
                  json={"queries": [query] * len(docs), "documents": docs}, timeout=180).json()
    if "scores" not in r: raise RuntimeError(f"rerank: {str(r)[:200]}")
    _st = r.get("inference_status") or {}
    _acc(rr_in=_st.get("tokens_input", 0), rr_cost=_st.get("cost", 0) / 100.0, rr_calls=1)
    return np.array(r["scores"], dtype=np.float32), r.get("inference_status", {}).get("cost", 0) / 100.0

def mkrow(u, cos, rr, base_rank, also=None):
    m = meta.get(u, {}); orig = m.get("snip") or ""
    return {"uid": u, "title": m.get("title") or u, "src": m.get("source") or "", "url": m.get("url") or "#",
            "cos": round(float(cos), 4), "rr": (round(float(rr), 4) if rr is not None else None), "base_rank": base_rank,
            "also": [{"uid": x, "url": (meta.get(x, {}).get("url") or "#")} for x in (also or [])[:8]],
            "original": orig, "t0": rdoc.get(u) or orig, "t1": t1doc.get(u) or rdoc.get(u) or orig}

def collapse(order_idx, sims, limit, skip=None, allow=None):
    """Walk ranked rows, keep one per exact-duplicate group; mirrors ride along on the kept row.
    allow: if set, only keep rows whose source OJ is in it (so the top-k is pulled from those OJs)."""
    kept = []; seen_g = set()
    for i in order_idx:
        u = uids[int(i)]
        if skip and u in skip: continue
        if allow is not None and (meta.get(u, {}).get("source") not in allow): continue
        g = DUP.get(u)
        if g:
            if g[0] in seen_g: continue
            seen_g.add(g[0]); also = [x for x in g if x != u]
        else:
            also = []
        kept.append((u, float(sims[int(i)]), also))
        if len(kept) >= limit: break
    return kept

def source_hist(order_idx, skip, cap=1000):
    """Per-OJ match counts over the top `cap` deduped results (ignores the source filter) —
    populates the filter UI with the OJs that actually matched this query, biggest first."""
    c = {}; seen_g = set(); n = 0
    for i in order_idx:
        u = uids[int(i)]
        if skip and u in skip: continue
        g = DUP.get(u)
        if g:
            if g[0] in seen_g: continue
            seen_g.add(g[0])
        s = meta.get(u, {}).get("source") or "?"
        c[s] = c.get(s, 0) + 1; n += 1
        if n >= cap: break
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))

# ---- warm up so the first user query is fast (cloud: connection pool; local: ollama model load) ----
try:
    _t = time.time(); embed_batch(["warmup"]); print(f"warmup embed ok ({time.time()-_t:.1f}s)", flush=True)
except Exception as e:
    print(f"warmup embed failed: {e}", flush=True)

# ---- api ----
app = FastAPI()
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
if os.path.isdir(os.path.join(UI, "fonts")):   # self-hosted font fallback (used when Google Fonts is unreachable)
    from fastapi.staticfiles import StaticFiles
    app.mount("/fonts", StaticFiles(directory=os.path.join(UI, "fonts")), name="fonts")

class SearchReq(BaseModel):
    query: str = Field(..., max_length=16000)   # LAN-exposed: bound rewrite/embed cost per request
    k: int = Field(50, ge=1, le=200); rewrite: bool = False; rerank: bool = False; skip_short: bool = True
    sources: Optional[List[str]] = None   # inclusion filter by OJ; None/[] = all OJs (3.8-safe annotation)

@app.get("/")
def index(): return FileResponse(os.path.join(UI, "index.html"))

@app.get("/favicon.ico")
def favicon(): return FileResponse(os.path.join(UI, "favicon.ico"))

@app.get("/en")
def lang_en(): return RedirectResponse("/?lang=en", status_code=301)   # v1's locale paths live on

@app.get("/zh")
def lang_zh(): return RedirectResponse("/?lang=zh", status_code=301)

@app.get("/api/health")
def health(): return {"ok": True, "backend": BACKEND, "emb": EMODEL, "problems": P, "rerank": RERANK_OK, "rewrite": rewrite_ok(),
                      "models": {"embed": EMB_LOCAL if BACKEND == "local" else EMB_CLOUD,
                                 "rewrite": GEN_LOCAL if BACKEND == "local" else GEN_CLOUD,
                                 "rerank": RERANKER if RERANK_OK else None}}

# public stats — aggregates api_stats.jsonl (counts/tokens only, no raw data exists in it); page is unlinked from the UI
PRICE = {"emb_toks": 0.15 / 1e6, "rw_in": 0.05 / 1e6, "rw_out": 0.10 / 1e6}   # $/token: gemini-embedding OR, gemma-3-12b DeepInfra

@app.get("/stats")
def stats_page(): return FileResponse(os.path.join(UI, "stats.html"))

@app.get("/api/stats")
def api_stats():
    days = {}; hours = [{"h": h, "searches": 0, "reranks": 0, "cost": 0.0} for h in range(24)]
    cutoff = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 31 * 86400))
    try:
        lines = open(STATS)
    except Exception:
        lines = []
    for ln in lines:
        try:
            r = json.loads(ln)
        except Exception:
            continue
        ts = r.get("ts", "")
        if ts < cutoff: continue
        cost = sum(r.get(k, 0) * p for k, p in PRICE.items()) + r.get("rr_cost", 0)
        key = "reranks" if r.get("rerank") else "searches"
        b = days.setdefault(ts[:10], {"searches": 0, "reranks": 0, "cost": 0.0})
        b[key] += 1; b["cost"] += cost
        hb = hours[int(ts[11:13])]
        hb[key] += 1; hb["cost"] += cost
    out_days = [{"d": d, "searches": v["searches"], "reranks": v["reranks"], "cost": round(v["cost"], 4)}
                for d, v in sorted(days.items())]
    for hb in hours: hb["cost"] = round(hb["cost"], 4)
    tot = {"searches": sum(v["searches"] for v in days.values()), "reranks": sum(v["reranks"] for v in days.values()),
           "cost": round(sum(v["cost"] for v in days.values()), 4)}
    return {"window_days": 31, "days": out_days, "hours": hours, "total": tot}

@app.post("/api/search")
def search(req: SearchReq):
    q = (req.query or "").strip()
    if not q: return {"results": [], "rewrites": None, "reranked": False, "timing": {}, "cost": 0}
    _TL.u = {}; t0 = time.time(); ok = True
    do_rerank = req.rerank and RERANK_OK
    do_rewrite = req.rewrite and rewrite_ok()
    try:
        qv, rewrites, qv_cached = query_vec_cached(q, do_rewrite); te = time.time() - t0
        sims = cosine_all(qv)
        cost = 0.0; rr_time = 0.0
        order = np.argsort(-sims); skip = SHORT if req.skip_short else None
        allow = set(req.sources) if req.sources else None
        ranked = collapse(order, sims, RPOOL if do_rerank else req.k, skip, allow)
        src_counts = source_hist(order, skip)
        if do_rerank:
            cu = [u for u, _, _ in ranked]; ccos = [c for _, c, _ in ranked]; calso = [a for _, _, a in ranked]
            tr = time.time(); rr, cost = rerank(q, cu); rr_time = time.time() - tr
            order = np.argsort(-rr)
            rows = [mkrow(cu[int(order[p])], ccos[int(order[p])], float(rr[int(order[p])]), int(order[p]) + 1, calso[int(order[p])])
                    for p in range(min(req.k, len(order)))]
            reranked = True
        else:
            rows = [mkrow(u, c, None, p + 1, a) for p, (u, c, a) in enumerate(ranked)]
            reranked = False
        return {"results": rows, "rewrites": rewrites, "reranked": reranked, "backend": BACKEND,
                "src_counts": src_counts,
                "timing": {"embed": round(te, 2), "rerank": round(rr_time, 2), "total": round(time.time() - t0, 2)},
                "cost": round(cost, 4)}
    except Exception:
        ok = False; raise
    finally:
        _log_stats({"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), "ok": ok, "qlen": len(q), "k": req.k,
                    "rewrite": do_rewrite, "rerank": do_rerank, "skip_short": req.skip_short, "qv_cached": locals().get("qv_cached", False),
                    "emb": EMODEL, "s": round(time.time() - t0, 2), **getattr(_TL, "u", {})})
        _TL.u = None

if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("YT_HOST", "0.0.0.0"), port=int(os.environ.get("YT_PORT", "8000")))
