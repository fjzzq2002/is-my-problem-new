# run summarize for all the problems
# use the chatgpt api
import requests
import json
from .utils import read_problems, problems_filenames, dump_json_safe, dump_pickle_safe, dump_numpy_safe
from openai import AsyncOpenAI
import hashlib
import asyncio
from tqdm.auto import tqdm
import time
from openai import OpenAI
import numpy as np
import pickle

# it seems most vector database libraries are too heavy for my use case

with open('settings.json') as f:
    settings = json.load(f)

client = OpenAI(
    api_key=settings['OPENAI_API_KEY'],
)

def get_embeddings(texts, model="text-embedding-ada-002"):
    texts = [t.replace("\n", " ") for t in texts]
    return np.array([x.embedding for x in client.embeddings.create(input = texts, model=model).data])

class VectorDB:
    def __init__(self):
        self.arr = np.array([])
        self.metadata = []
    
    def insert(self, values, metadata):
        values = np.array(values)
        # concat
        if len(self.arr)>0:
            self.arr = np.concatenate([self.arr, values])
        else:
            self.arr = values
        self.metadata.extend(metadata)

    def save(self, filename='embs/embs'):
        dump_numpy_safe(self.arr, filename+'.npy')
        dump_json_safe(self.metadata, filename+'.json')

    def load(self, filename='embs/embs'):
        try:
            with open(filename+'.npy', 'rb') as f:
                self.arr = np.load(f)
            with open(filename+'.json') as f:
                self.metadata = json.load(f)
            if len(self.arr)!=len(self.metadata):
                raise Exception('length mismatch, corrupted db')
        except Exception as e:
            print('failed to load embs:',e)
        return self

db = VectorDB().load()
emb_keys = set([x[0] for x in db.metadata])
print('read',len(emb_keys),'embeddings from db')


# initialize all encodings if not already done
for problem_file in problems_filenames():
    problems = read_problems('problems/'+problem_file)
    todos = []
    for t in problems:
        uid = t['uid']
        for u in t['processed']:
            todos.append((u['result'], uid))
    # process todos in chunk of 100
    chunk_size = 100
    for i in tqdm(range(0,len(todos),chunk_size)):
        chunk = [
            x for x in todos[i:i+chunk_size]
            if x[0] not in emb_keys
        ]
        if len(chunk)==0:
            continue
        emb = get_embeddings([x[0] for x in chunk])
        for x in chunk: emb_keys.add(x[0])
        db.insert(emb, chunk)
        db.save()

def rd(x):
    if '0'<=x[-1]<='9':
        return x[:-1]
    return x

mx=(float('-inf'),0)
for i in range(len(db.arr)):
    for j in range(i):
        if rd(db.metadata[i][1])==rd(db.metadata[j][1]):
            continue
        cs = np.dot(db.arr[i], db.arr[j])/(np.linalg.norm(db.arr[i])*np.linalg.norm(db.arr[j]))
        mx=max(mx,(cs,i,j))
cs,i,j=mx
print(db.metadata[i][0])
print(db.metadata[j][0])
print(cs)
print(db.metadata[i][1])
print(db.metadata[j][1])