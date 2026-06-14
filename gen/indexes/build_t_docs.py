"""Display/rerank docs for the UI: each problem's temp-0 rewrite per template.
rerank_docs.jsonl = the T0 (faithful) temp-0 rewrite — also fed to the reranker.
t1_docs.jsonl     = the T1 (succinct) temp-0 rewrite.

Selected from gen/rewrites.jsonl (t = template id, k = sample id; k=0 is the temp-0
greedy anchor). The cache is append-only and versioned by statement hash (`sh` =
md5(rewrite input)[:12], where the input is the stripped statement truncated to
MAX_STMT — must match gen_rewrites.py). A gen whose sh matches the CURRENT statement
wins; otherwise the last appended gen is used (stale fallback, flagged in the summary).
Output: emb/rerank_docs.jsonl, emb/t1_docs.jsonl = {"uid", "doc"} per line.
"""
import json, os, hashlib

ROOT = os.environ.get('YT_ROOT', '.')
MAX_STMT = 16000                                  # keep in sync with gen_rewrites.py
def sthash(s): return hashlib.md5(s.encode('utf-8')).hexdigest()[:12]

cur_sh = {}; seen = set()
for ln in open(os.path.join(ROOT, 'problems.jsonl'), encoding='utf-8'):
    try: r = json.loads(ln)
    except Exception: continue
    u = r.get('uid')
    if not u or u in seen: continue
    seen.add(u)
    cur_sh[u] = sthash(((r.get('statement') or '').strip())[:MAX_STMT])

docs = [{}, {}]          # template -> uid -> doc
fresh = [{}, {}]         # uid -> True when the gen's sh matches the current statement
for ln in open(os.path.join(ROOT, 'gen', 'rewrites.jsonl'), encoding='utf-8'):
    try: r = json.loads(ln)
    except Exception: continue
    if r.get('k') != 0: continue
    t = 0 if r.get('t') == 0 else 1; u = r['uid']
    if fresh[t].get(u): continue                      # already have the current-statement gen
    docs[t][u] = r.get('text') or ''
    fresh[t][u] = (r.get('sh') == cur_sh.get(u))

for name, t in (('rerank_docs.jsonl', 0), ('t1_docs.jsonl', 1)):
    with open(os.path.join(ROOT, 'emb', name), 'w', encoding='utf-8') as f:
        for u in sorted(docs[t]):
            f.write(json.dumps({'uid': u, 'doc': docs[t][u]}, ensure_ascii=False) + '\n')
    stale = sum(1 for u in docs[t] if not fresh[t][u])
    print(f'{len(docs[t])} docs -> emb/{name} ({stale} stale fallbacks)')
