# run summarize for all the problems
# use the chatgpt api
import requests
import json
from .utils import (
    read_problems,
    problems_filenames,
    dump_json_safe,
    dump_pickle_safe,
    dump_numpy_safe,
)
from openai import AsyncOpenAI
import hashlib
import asyncio
from tqdm.auto import tqdm
import time
from openai import OpenAI
import numpy as np
import pickle
import cohere
import os

# it seems most vector database libraries are too heavy for my use case

with open("settings.json") as f:
    settings = json.load(f)

if settings["EMBED"] == "COHERE":
    co = cohere.Client(settings["COHERE_API_KEY"])


client = OpenAI(
    api_key=settings["OPENAI_API_KEY"],
)


def get_embeddings(texts, query=False):
    texts = [t.replace("\n", " ") for t in texts]
    if settings["EMBED"] == "COHERE":
        response = co.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_query" if query else "search_document",
        )
        return np.array(response.embeddings)
    elif settings["EMBED"] == "OPENAI":
        return np.array(
            [
                x.embedding
                for x in client.embeddings.create(
                    input=texts, model="text-embedding-ada-002"
                ).data
            ]
        )
    else:
        assert False


# quick and dirty vector database implementation
class VectorDB:
    def __init__(self):
        self.arr = np.array([])
        self.metadata = []

    def insert(self, values, metadata):
        values = np.array(values)
        # concat
        if len(self.arr) > 0:
            self.arr = np.concatenate([self.arr, values])
        else:
            self.arr = values
        self.metadata.extend(metadata)

    def save(self, filename=None):
        if filename is None:
            filename = "embs/embs_" + settings["EMBED"].lower()
            if not os.path.exists("embs"):
                os.mkdir("embs")
        dump_numpy_safe(self.arr, filename + ".npy")
        dump_json_safe(self.metadata, filename + ".json")

    def load(self, filename=None):
        if filename is None:
            filename = "embs/embs_" + settings["EMBED"].lower()
        try:
            with open(filename + ".npy", "rb") as f:
                self.arr = np.load(f)
            with open(filename + ".json") as f:
                self.metadata = json.load(f)
            if len(self.arr) != len(self.metadata):
                raise Exception("length mismatch, corrupted db")
        except Exception as e:
            print("failed to load embs:", e)
        return self

    def query_nearest(self, emb, dedup=lambda t: t[1], k=10):
        # return the k nearest embeddings with cosine similarity
        # return a list of (cosine similarity, metadata) tuples
        # the list is sorted by cosine similarity
        sims = np.dot(self.arr, emb)
        topk = np.argsort(sims)[::-1]
        nearest = []
        keys = set()
        for i in topk:
            if dedup is not None:
                key = dedup(self.metadata[i])
                if key in keys:
                    continue
                keys.add(key)
            nearest.append((sims[i], self.metadata[i]))
            if len(nearest) >= k:
                break
        return nearest
