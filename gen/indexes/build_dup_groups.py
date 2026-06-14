"""Exact-duplicate groups for serve-time collapse. Two signals, both byte-exact:

1) md5 of the literal statement text
2) md5 of the statement with the embedded title heading removed, all whitespace
   stripped, lowercased — digits KEPT. Catches mirrors that differ only in the
   title line (e.g. POJ "Ant Counting" vs OpenJ_NOI "[Usaco2005 Nov]Ant Counting").

NO fuzzy merging by design (no title matching, no embedding similarity):
constraint variants (easy/hard versions, GCJ small/large, (A)/(B)/(C) relistings,
digit-only differences) are distinct problems and stay separate.

Stubs (<250 chars after normalization) are excluded — identical boilerplate, not clones.
Canonical = first element: non-Gym beats Gym (Gym clones), known mirrors rank below
their origin (POJ>Bailian, OJUZ>QOJ), then earliest numeric id (mashups clone from
earlier contests). Output: emb/dup_groups.json = [[canonical, mirror, ...], ...]
"""
import json, hashlib, re, os, unicodedata
from collections import defaultdict

ROOT = os.environ.get('YT_ROOT', '.')
PRIO = {'CodeForces': 0, 'AtCoder': 0, 'TopCoder': 1, 'POJ': 1, 'OJUZ': 1,
        '洛谷': 2, 'QOJ': 3, 'OpenJ_Bailian': 3, '计蒜客': 3, 'Gym': 9}
def key(u):
    s, i = u.split('/', 1)
    m = re.search(r'\d+', i)
    return (PRIO.get(s, 2), int(m.group()) if m else 10**12, u)

def norm_content(st, title):
    s = unicodedata.normalize('NFKC', st or '')
    t = unicodedata.normalize('NFKC', title or '').strip()
    if len(t) >= 3:
        s = re.sub(re.escape(t), ' ', s, flags=re.I)   # drop embedded title/heading
    return re.sub(r'\s+', '', s).lower()

parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b): parent[find(a)] = find(b)

seen = set(); h_raw = defaultdict(list); h_content = defaultdict(list)
for ln in open(os.path.join(ROOT, 'problems.jsonl'), encoding='utf-8'):
    try: r = json.loads(ln)
    except Exception: continue
    u = r.get('uid'); s = r.get('statement') or ''
    if not u or u in seen: continue
    seen.add(u)
    if len(s) >= 250:
        h_raw[hashlib.md5(s.encode('utf-8')).hexdigest()].append(u)
    n = norm_content(s, r.get('title'))
    if len(n) >= 250:
        h_content[hashlib.md5(n.encode('utf-8')).hexdigest()].append(u)

for h in (h_raw, h_content):
    for us in h.values():
        for u in us[1:]: union(u, us[0])

groups = defaultdict(list)
for u in parent: groups[find(u)].append(u)
out = sorted((sorted(g, key=key) for g in groups.values() if len(g) > 1), key=lambda g: g[0])
json.dump(out, open(os.path.join(ROOT, 'emb', 'dup_groups.json'), 'w'), ensure_ascii=False)
print(f'{len(out)} groups, {sum(len(g) for g in out)} members')
print('samples:', out[0], out[1000])
