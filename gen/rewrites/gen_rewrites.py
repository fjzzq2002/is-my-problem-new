"""Generate gemma-3-12b-it rewrites for the corpus (T0x2 + T1x2 per problem).

Pinned to DeepInfra (cheap bf16). Async + semaphore. Append-only JSONL cache: every
finished job is one line, so the run is fully resumable and partial progress is never lost.
Uniform nested sampling by md5(uid): FRACTION=0.001 ⊂ 0.01 ⊂ 1.0, so a bigger run reuses a
smaller one's cache. No embedding here — generations only.

Env knobs: FRACTION, CONCURRENCY, TEMP, MAXTOK.
"""
import os, sys, json, re, time, asyncio, hashlib
import aiohttp

# ---- config / secrets ----
ROOT = os.environ.get('YT_ROOT', '.')
for ln in open(os.path.join(ROOT, '.env')):
    ln = ln.strip()
    if ln and not ln.startswith('#') and '=' in ln:
        k, v = ln.split('=', 1); os.environ.setdefault(k, v.strip().strip('"\''))
TEMPLATES = [  # the production rewrite prompts (T0 faithful, T1 succinct)
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

PROBLEMS = f'{ROOT}/problems.jsonl'
OUT = f'{ROOT}/gen/rewrites.jsonl'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

FRACTION    = float(os.environ.get('FRACTION', '0.001'))
CONCURRENCY = int(os.environ.get('CONCURRENCY', '64'))
TEMPS       = [float(x) for x in os.environ.get('TEMPS', '0,1').split(',')]  # per-sample temp: k=0->0, k=1->1
MAXTOK      = int(os.environ.get('MAXTOK', '2048'))
BACKEND     = os.environ.get('BACKEND', 'deepinfra')
MODEL    = 'google/gemma-3-12b-it'
MAX_STMT = 16000
KINDS    = {'text', 'text+media', 'pdf_text'}
SAMPLES  = len(TEMPS)              # samples per template -> 2*SAMPLES gens/problem
THRESH   = int(round(FRACTION * 100_000))
if BACKEND == 'deepinfra':         # direct: own account, higher concurrency, returns estimated_cost
    URL = 'https://api.deepinfra.com/v1/openai/chat/completions'
    HEADERS = {'Authorization': f'Bearer {os.environ["DEEPINFRA_KEY"]}', 'Content-Type': 'application/json'}
    PROVIDER = None
else:                              # openrouter, pinned to DeepInfra provider
    URL = 'https://openrouter.ai/api/v1/chat/completions'
    HEADERS = {'Authorization': f'Bearer {os.environ["OPENROUTER_KEY"]}', 'Content-Type': 'application/json'}
    PROVIDER = 'DeepInfra'

def bucket(uid): return int(hashlib.md5(uid.encode()).hexdigest(), 16) % 100_000
def selected(uid): return bucket(uid) < THRESH

def sthash(s): return hashlib.md5(s.encode('utf-8')).hexdigest()[:12]

# ---- pass 1: select problems + record the CURRENT input-statement hash (cache versioning) ----
sel = {}; total_usable = 0                        # uid -> truncated statement actually fed to the model
with open(PROBLEMS, encoding='utf-8') as f:
    for ln in f:
        try: r = json.loads(ln)
        except Exception: continue
        s = (r.get('statement') or '').strip()
        if r.get('statement_kind') not in KINDS or not s:
            continue
        total_usable += 1
        uid = r['uid']
        if selected(uid):
            sel[uid] = s[:MAX_STMT]
cur_sh = {uid: sthash(s) for uid, s in sel.items()}
full_jobs = total_usable * 2 * SAMPLES

# ---- resume: a cached gen is 'done' ONLY if its stored input-hash matches the current statement.
#      Mismatch (statement changed since gen, e.g. re-convert) -> stale, regenerate (don't resume). ----
done = set(); stale = 0; orphan = 0
if os.path.exists(OUT):
    for ln in open(OUT):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get('text') is None: continue       # failures: retry
        uid = r['uid']; k = r['k']
        if uid not in cur_sh: orphan += 1; continue   # cached for a uid not in this sample
        cur_temp = TEMPS[k] if k < len(TEMPS) else None
        if r.get('sh') == cur_sh[uid] and r.get('temp') == cur_temp:
            done.add((uid, r['t'], k))                # valid only if statement AND sample-temp match
        else: stale += 1
if stale:
    print(f'WARNING: {stale} cached gens are STALE (input statement changed since generation) '
          f'-> ignoring & regenerating', flush=True)
print(f'resume: {len(done)} valid cached gens'
      + (f' ({orphan} cached for non-sampled uids ignored)' if orphan else ''), flush=True)

# ---- build job list ----
jobs = []
for uid, s in sel.items():
    sh = cur_sh[uid]
    for t in (0, 1):
        for k in range(SAMPLES):
            if (uid, t, k) not in done:
                jobs.append((uid, t, k, s, sh))
print(f'corpus: {total_usable} usable problems  ->  full run = {full_jobs} gens', flush=True)
print(f'fraction {FRACTION}: {len(sel)} problems, {len(jobs)} gens to do '
      f'(concurrency={CONCURRENCY}, temps={TEMPS}, max_tok={MAXTOK})', flush=True)
if not jobs:
    print('nothing to do.'); sys.exit(0)

# ---- async generation ----
fout = open(OUT, 'a', encoding='utf-8')
st = {'ok': 0, 'fail': 0, 'pt': 0, 'ct': 0, 'cap': 0, 'retries': 0,
      'cost': 0.0, 'errs': {}, 'lat': [], 't0': time.time()}

def record_err(tag): st['errs'][tag] = st['errs'].get(tag, 0) + 1

async def one(session, sem, job):
    uid, t, k, stmt, sh = job
    temp = TEMPS[k]
    body = {'model': MODEL,
            'messages': [{'role': 'user', 'content': TEMPLATES[t].replace('[[ORIGINAL]]', stmt)},
                         {'role': 'assistant', 'content': 'Simplified statement:'}],
            'temperature': temp, 'max_tokens': MAXTOK}
    if PROVIDER:
        body['provider'] = {'order': [PROVIDER], 'allow_fallbacks': False}
    async with sem:
        t_start = time.time()
        for attempt in range(8):
            try:
                async with session.post(URL, json=body, headers=HEADERS) as resp:
                    if resp.status == 200:
                        j = await resp.json()
                        if 'choices' not in j:
                            record_err('no_choices:' + str(j.get('error', {}).get('code', '?')))
                            await asyncio.sleep(min(2 ** attempt, 30)); st['retries'] += 1; continue
                        txt = (j['choices'][0]['message'].get('content') or '').strip()
                        txt = re.sub(r'^\s*Simplified statement:\s*', '', txt).strip()
                        u = j.get('usage', {}) or {}
                        pt, ct = u.get('prompt_tokens', 0), u.get('completion_tokens', 0)
                        fr = j['choices'][0].get('finish_reason')
                        rec = {'uid': uid, 't': t, 'k': k, 'temp': temp, 'sh': sh, 'text': txt or None,
                               'pt': pt, 'ct': ct, 'fr': fr}
                        fout.write(json.dumps(rec, ensure_ascii=False) + '\n'); fout.flush()
                        if txt:
                            st['ok'] += 1; st['pt'] += pt; st['ct'] += ct
                            st['cost'] += (u.get('estimated_cost') or 0.0)
                            st['lat'].append(time.time() - t_start)
                            if fr == 'length': st['cap'] += 1
                        else:
                            st['fail'] += 1; record_err('empty')
                        return
                    elif resp.status in (429, 500, 502, 503, 520, 522, 524):
                        record_err('http' + str(resp.status))
                        ra = resp.headers.get('Retry-After')
                        wait = float(ra) if (ra and ra.isdigit()) else min(2 ** attempt, 30)
                        await asyncio.sleep(wait); st['retries'] += 1; continue
                    else:
                        body_txt = (await resp.text())[:200]
                        record_err('http' + str(resp.status))
                        await asyncio.sleep(min(2 ** attempt, 30)); st['retries'] += 1; continue
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                record_err(type(e).__name__)
                await asyncio.sleep(min(2 ** attempt, 30)); st['retries'] += 1; continue
        # exhausted retries
        fout.write(json.dumps({'uid': uid, 't': t, 'k': k, 'temp': temp, 'sh': sh, 'text': None, 'err': 'retries'},
                              ensure_ascii=False) + '\n'); fout.flush()
        st['fail'] += 1

async def progress(total):
    while True:
        await asyncio.sleep(5)
        d = st['ok'] + st['fail']; el = time.time() - st['t0']
        rate = d / el if el else 0
        eta = (total - d) / rate if rate else 0
        print(f'  {d}/{total}  ok={st["ok"]} fail={st["fail"]} cap={st["cap"]} '
              f'retries={st["retries"]}  {rate:.1f}/s  eta {eta/60:.1f}m  errs={st["errs"]}', flush=True)
        if d >= total: return

async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=180, connect=15, sock_read=120)
    conn = aiohttp.TCPConnector(limit=CONCURRENCY, ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        prog = asyncio.create_task(progress(len(jobs)))
        await asyncio.gather(*(one(session, sem, j) for j in jobs))
        prog.cancel()

asyncio.run(main())
fout.close()

# ---- report + extrapolation ----
el = time.time() - st['t0']
done_n = st['ok'] + st['fail']
rate = done_n / el if el else 0
PIN, POUT = 0.04e-6, 0.13e-6                     # DeepInfra gemma-3-12b fallback prices
spent = st['cost'] if st['cost'] else st['pt'] * PIN + st['ct'] * POUT
print(f'\n=== {BACKEND} fraction {FRACTION} done in {el:.0f}s ===', flush=True)
print(f'ok={st["ok"]} fail={st["fail"]} cap(hit max_tok)={st["cap"]} retries={st["retries"]}')
print(f'throughput {rate:.1f} gen/s  |  errors: {st["errs"]}')
if st['ok']:
    apt, act = st['pt'] / st['ok'], st['ct'] / st['ok']
    lat = sorted(st['lat']); p50 = lat[len(lat)//2]; p95 = lat[int(len(lat)*0.95)]
    per_gen = spent / st['ok']
    print(f'avg tokens/gen: prompt={apt:.0f} completion={act:.0f}  latency p50={p50:.1f}s p95={p95:.1f}s')
    print(f'this run cost: ${spent:.4f}  (${per_gen*1000:.4f}/1k gens)')
    full_time = full_jobs / rate if rate else 0
    print(f'\nEXTRAPOLATION to full corpus ({full_jobs} gens):')
    print(f'  cost  ~ ${per_gen * full_jobs:.2f}')
    print(f'  time  ~ {full_time/3600:.1f}h at this throughput ({rate:.1f}/s)')
