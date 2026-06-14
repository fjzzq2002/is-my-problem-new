"""Build the centroid index: mean each problem's 4 rewrite vectors -> 1 normalized vector.
order.jsonl is sorted by (uid,t,k), so each problem is exactly 4 consecutive rows -> reshape+mean.
Output: emb/centroids.f16 (P x 4096 float16, L2-normalized) + emb/centroid_uids.json (row -> uid).
"""
import json, os, numpy as np

D = os.path.join(os.environ.get('YT_ROOT', '.'), 'emb')
meta = json.load(open(f'{D}/meta.json')); N, DIM = meta['N'], meta['dim']
order = [json.loads(l) for l in open(f'{D}/order.jsonl')]
assert N % 4 == 0, N
P = N // 4
# verify each block of 4 is one uid with the 4 (t,k) combos
for i in (0, 1, P//2, P-1):
    blk = order[4*i:4*i+4]
    assert len({u for u, t, k in blk}) == 1 and {(t, k) for u, t, k in blk} == {(0,0),(0,1),(1,0),(1,1)}, blk
uids = [order[4*i][0] for i in range(P)]
print(f'{P} problems', flush=True)

vecs = np.memmap(f'{D}/vectors.f16', dtype='float16', mode='r', shape=(N, DIM))
cent = np.zeros((P, DIM), dtype=np.float32)
CH = 20000  # problems per chunk
for s in range(0, P, CH):
    e = min(s+CH, P)
    blk = np.asarray(vecs[4*s:4*e], dtype=np.float32).reshape(e-s, 4, DIM)
    cent[s:e] = blk.mean(1)
    print(f'  {e}/{P}', flush=True)
cent /= np.linalg.norm(cent, axis=1, keepdims=True)
cent.astype(np.float16).tofile(f'{D}/centroids.f16')
json.dump(uids, open(f'{D}/centroid_uids.json', 'w'))
print(f'saved {D}/centroids.f16 ({P}x{DIM} float16) + centroid_uids.json', flush=True)
