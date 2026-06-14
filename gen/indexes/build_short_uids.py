"""Stub-statement skip list for serve-time filtering (the UI's 跳过过短题面 switch).
A problem is a stub if its statement is <20 chars after stripping digits and whitespace
("download", "sample input", "### Hint 无数据" …). Only in-index uids are listed — problems
that failed the usability gate never got embedded and can't be returned anyway.
Output: emb/short_uids.json = [uid, ...]
"""
import json, re, os

ROOT = os.environ.get('YT_ROOT', '.')
idx = set(json.load(open(os.path.join(ROOT, 'emb', 'centroid_uids.json'))))
seen = set(); short = []
for ln in open(os.path.join(ROOT, 'problems.jsonl'), encoding='utf-8'):
    try: r = json.loads(ln)
    except Exception: continue
    u = r.get('uid')
    if not u or u in seen: continue
    seen.add(u)
    if u in idx and len(re.sub(r'[\d\s]+', '', r.get('statement') or '')) < 20:
        short.append(u)
json.dump(sorted(short), open(os.path.join(ROOT, 'emb', 'short_uids.json'), 'w'))
print(f'{len(short)} stub uids -> emb/short_uids.json')
