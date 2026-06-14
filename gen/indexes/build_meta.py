"""Compact per-problem metadata for the beta UI: uid -> title, url, source, statement snippet.
Deduped by uid (matches the centroid index). Reminder prefix stripped for display.
Output: emb/meta.jsonl
"""
import json, os, re
ROOT = os.environ.get('YT_ROOT', '.')
SEEN = set()
REM = re.compile(r'^Note: the following may contain statements of multiple problems\..*?Ignore the rest\.\s*', re.S)
out = open(f'{ROOT}/emb/meta.jsonl', 'w', encoding='utf-8')
n = 0
for ln in open(f'{ROOT}/problems.jsonl', encoding='utf-8'):
    try: r = json.loads(ln)
    except Exception: continue
    u = r['uid']
    if u in SEEN: continue
    SEEN.add(u)
    s = REM.sub('', r.get('statement') or '')
    out.write(json.dumps({'uid': u, 'title': r.get('title'), 'url': r.get('url'),
                          'source': r.get('source'), 'snip': s[:700]}, ensure_ascii=False) + '\n')
    n += 1
out.close()
print(f'wrote {n} unique-uid metadata records -> emb/meta.jsonl')
