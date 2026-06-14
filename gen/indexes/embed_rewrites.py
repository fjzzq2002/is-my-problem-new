"""Embed all rewrites with qwen3-embedding-8b (OpenRouter), crash-safe + resumable.

Storage (emb/):
  order.jsonl   row -> [uid,t,k]   (frozen on first run, reused on resume; row indices never shift)
  vectors.f16   np.memmap float16 (N, 4096)  raw per-rewrite vectors (~8.4GB)
  done.u8       np.memmap uint8 (N,)          0=todo 1=done
  meta.json     {N, dim, model}
Workers embed batches of 256 (48 concurrent), write rows directly to the memmap, mark done.
Flush order is vectors-then-done, so a crash only ever leaves a row RE-DOABLE (never done-with-garbage).
Resume = todo is `rows where done==0`; failed batches stay todo and auto-retry next run.

Env: CONC (48), BATCH (256), FLUSH_EVERY (20).
"""
import os, sys, json, time, threading
import numpy as np, requests
from concurrent.futures import ThreadPoolExecutor

ROOT = os.environ.get('YT_ROOT', '.')
for ln in open(os.path.join(ROOT, '.env')):
    if '=' in ln and not ln.startswith('#'):
        k, v = ln.strip().split('=', 1); os.environ.setdefault(k, v.strip().strip('"\''))
KEY = os.environ['OPENROUTER_KEY']
URL, MODEL, DIM = 'https://openrouter.ai/api/v1/embeddings', 'qwen/qwen3-embedding-8b', 4096
REWRITES = f'{ROOT}/gen/rewrites.jsonl'
D = f'{ROOT}/emb'; os.makedirs(D, exist_ok=True)
ORDER, VEC, DONE, META = f'{D}/order.jsonl', f'{D}/vectors.f16', f'{D}/done.u8', f'{D}/meta.json'
CONC  = int(os.environ.get('CONC', '48'))
BATCH = int(os.environ.get('BATCH', '256'))
FLUSH_EVERY = int(os.environ.get('FLUSH_EVERY', '20'))

# ---- key -> text (all rewrites) ----
print('loading rewrites...', flush=True)
text_by_key = {}
for ln in open(REWRITES, encoding='utf-8'):
    try: r = json.loads(ln)
    except Exception: continue
    if r.get('text'): text_by_key[(r['uid'], r['t'], r['k'])] = r['text']
print(f'{len(text_by_key)} rewrites with text', flush=True)

# ---- frozen ordering (load existing or build once) ----
if os.path.exists(ORDER):
    order = [tuple(json.loads(l)) for l in open(ORDER)]
    print(f'resumed order: {len(order)} rows', flush=True)
else:
    order = sorted(text_by_key.keys())
    with open(ORDER, 'w') as f:
        for k in order: f.write(json.dumps(list(k)) + '\n')
    print(f'built order: {len(order)} rows', flush=True)
N = len(order)
json.dump({'N': N, 'dim': DIM, 'model': MODEL}, open(META, 'w'))

# ---- memmaps ----
vecs = np.memmap(VEC, dtype='float16', mode='r+' if os.path.exists(VEC) else 'w+', shape=(N, DIM))
done = np.memmap(DONE, dtype='uint8', mode='r+' if os.path.exists(DONE) else 'w+', shape=(N,))
todo = np.nonzero(done[:] == 0)[0]
print(f'N={N}  done={N-len(todo)}  todo={len(todo)}  (conc={CONC} batch={BATCH})', flush=True)
if len(todo) == 0:
    print('nothing to do.'); sys.exit(0)
batches = [todo[i:i+BATCH] for i in range(0, len(todo), BATCH)]

SESS = requests.Session()
SESS.mount('https://', requests.adapters.HTTPAdapter(pool_connections=CONC, pool_maxsize=CONC + 16))
def embed(texts):
    for a in range(6):
        try:
            r = SESS.post(URL, headers={'Authorization': f'Bearer {KEY}'},
                          json={'model': MODEL, 'input': texts}, timeout=(10, 240))
            if r.status_code == 200:
                d = r.json().get('data')
                if d and len(d) == len(texts): return np.array([e['embedding'] for e in d], dtype=np.float32)
            time.sleep(min(2 ** a, 30))
        except Exception:
            time.sleep(min(2 ** a, 30))
    return None

st = {'ok': 0, 'fail': 0}; lock = threading.Lock(); t0 = time.time()
def work(rows):
    arr = embed([text_by_key[order[i]] for i in rows])
    if arr is None:
        with lock: st['fail'] += 1
        return                                   # leave done=0 -> retried on resume
    vecs[rows] = arr.astype(np.float16)          # disjoint rows -> GIL-safe write
    done[rows] = 1
    with lock:
        st['ok'] += 1
        if st['ok'] % FLUSH_EVERY == 0:
            vecs.flush(); done.flush()           # vectors first, then done = crash-safe
            n = st['ok'] * BATCH; el = time.time() - t0; rate = n / el
            print(f'  {n}/{len(todo)}  {rate:.0f} emb/s  eta {(len(todo)-n)/rate/60:.0f}m  fail={st["fail"]}', flush=True)

with ThreadPoolExecutor(max_workers=CONC) as ex:
    list(ex.map(work, batches))
vecs.flush(); done.flush()
remaining = int((done[:] == 0).sum())
print(f'\nDONE: ok_batches={st["ok"]} fail_batches={st["fail"]}  rows still todo={remaining}  '
      f'in {time.time()-t0:.0f}s', flush=True)
print(f'vectors at {VEC} (float16, {N}x{DIM})  -> next: centroid per problem', flush=True)
