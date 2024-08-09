import pickle
import requests
import json
from .utils import read_problem, problem_filenames, dump_json_safe, dump_json_safe_utf8, dump_pickle_safe
from openai import AsyncOpenAI
from together import Together, AsyncTogether
import anthropic
import hashlib
import asyncio
from tqdm.auto import tqdm

# from tqdm import tqdm
import time, os
import random
import voyageai


with open("settings.json") as f:
    settings = json.load(f)

voyage_client = voyageai.Client(
    api_key=settings['VOYAGE_API_KEY'],
    max_retries=3,
    timeout=120,
)

# client = Together(
#     api_key=settings['TOGETHER_API_KEY'],
# )


def processed_promptmd5(statement, template):
    ORIGINAL = statement
    prompt = template.replace("[[ORIGINAL]]", ORIGINAL).strip()
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]

import numpy as np

def problem_embeds(problem_file_cur):
    try:
        problem = read_problem(problem_file_cur)
        # load from corresponding npy
    except Exception as e:
        print('error',problem_file_cur,e)
        return None, None
    try:
        embeds = []
        with open(problem_file_cur.replace(".json", ".vopkl"), "rb") as f:
            embeds = pickle.load(f)
    except:
        pass
    return problem, embeds

# quick and dirty vector database implementation
class VectorDB:
    def __init__(self):
        pass
    
    def load_all(self, shuffle = False, load_around = None, record_tasks=False, skipped_sources = []):
        self.arr = []
        self.metadata = []
        self.todos = []
        self.sources = {}
        fns = list(problem_filenames())
        if shuffle:
            random.shuffle(fns)
        for problem_file_cur in tqdm(fns):
            # if '洛谷' not in problem_file_cur:
            #     continue
            if load_around is not None and len(self.arr) > load_around * 2:
                break
            if not record_tasks and not os.path.exists(problem_file_cur.replace(".json", ".vopkl")):
                continue
            problem, embeds = problem_embeds(problem_file_cur)
            if problem is None:
                continue
            statement = problem['statement']
            source = problem['source']
            if source in skipped_sources:
                continue
            self.sources[source] = self.sources.get(source, 0) + 1
            need_work = False
            for template in settings["TEMPLATES"]:
                md5 = processed_promptmd5(statement, template)
                found = False
                for m, u in embeds:
                    if m[:8] == md5:
                        found = True
                        self.arr.append(np.array(u/np.linalg.norm(u),dtype=np.float16))
                        self.metadata.append((problem_file_cur, source, len(statement.strip())))
                        break
                if not found:
                    need_work = True
            if need_work and record_tasks:
                self.todos.append(problem_file_cur)
        print('found',len(self.arr),'embeds')
        self.arr = np.array(self.arr,dtype=np.float16)
        if record_tasks:
            print('found',len(self.todos),'todos')

    
    def complete_todos(self, chunk_size = 200, length_limit = 1300, shuffle = False):
        todos = self.todos
        if shuffle:
            import random
            random.shuffle(todos)
        for i in tqdm(range(0,len(todos),chunk_size)):
            problems = todos[i:i+chunk_size]
            infos = {}
            for problem_file_cur in problems:
                try:
                    full_problem = read_problem(problem_file_cur)
                    statement = full_problem['statement']
                    # load from corresponding npy
                except Exception as e:
                    print('error',problem_file_cur,e)
                    continue
                try:
                    embeds = []
                    with open(problem_file_cur.replace(".json", ".vopkl"), "rb") as f:
                        embeds = pickle.load(f)
                except:
                    pass
                infos[problem_file_cur] = full_problem.get('processed',[]), statement, embeds
            for template in settings["TEMPLATES"]:
                queues = []
                max_length = 0
                for problem_file_cur, (processed, statement, embeds) in infos.items():
                    md5 = processed_promptmd5(statement, template)
                    if any(m[:8] == md5 for m, u in embeds): continue
                    # get processed
                    processed_text = None
                    for f in processed:
                        if f["prompt_md5"][:8] == md5:
                            if len(f['result']) > length_limit:
                                continue # too long?
                            processed_text = f["result"]
                            max_length = max(max_length, len(processed_text))
                    if processed_text is None:
                        continue
                    queues.append((processed_text, problem_file_cur, md5))
                if len(queues) == 0:
                    continue
                print('batch',len(queues),' maxlen',max_length)
                try:
                    t0 = time.time()
                    response = voyage_client.embed(
                        [
                            x[0] for x in queues
                        ],
                        model="voyage-large-2-instruct",
                        input_type='document'
                    )
                    print('Token spent',response.total_tokens)
                    t1 = time.time()
                    # wait till 0.5s
                    if t1 - t0 < 0.2:
                        time.sleep(0.2 - (t1 - t0))
                    for q,e in zip(queues, response.embeddings):
                        infos[q[1]][2].append((q[2], np.array(e)))
                except Exception as e:
                    print('error',e)
            for problem_file_cur, (processed, statement, embeds) in infos.items():
                dump_pickle_safe(embeds, problem_file_cur.replace(".json", ".vopkl"))


    def query_nearest(self, emb, k=1000, dedup=True):
        # return the k nearest embeddings with cosine similarity
        # return a list of (cosine similarity, metadata) tuples
        # the list is sorted by cosine similarity
        # normailze emb
        emb = np.array(emb)
        if len(emb.shape) == 1:
            emb = emb[None, :]
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        emb = np.array(emb, dtype=np.float16)
        sims = np.max(self.arr @ emb.T, axis=1)
        sims = np.clip((sims+1)/2, 0, 1)  # [-1,1] -> [0,1]
        topk = np.argsort(sims)[::-1]
        nearest = []
        keys = set()
        # print(f'query nearest {len(emb)=} {len(sims)=} {len(topk)=} {k=}')
        for i in topk:
            if dedup:
                key = self.metadata[i][0]
                if key in keys:
                    continue
                keys.add(key)
            nearest.append((sims[i], i))
            if len(nearest) >= k:
                break
        return nearest

if __name__ == "__main__":
    db = VectorDB()
    db.load_all(record_tasks=True)
    db.complete_todos(chunk_size=128)
