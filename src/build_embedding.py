from .embedder import VectorDB, get_embeddings
from .utils import read_problems, problems_filenames
from tqdm.auto import tqdm

db = VectorDB().load()
emb_keys = set([x[0] for x in db.metadata])
print("read", len(emb_keys), "embeddings from db")


# initialize all embeddings if not already done
for problem_file in problems_filenames():
    problems = read_problems("problems/" + problem_file)
    print("processing", problem_file)
    todos = []
    for t in problems:
        uid = t["uid"]
        for u in t["processed"]:
            todos.append((u["result"], uid))
    # process todos in chunk of 100
    chunk_size = 100
    for i in tqdm(range(0, len(todos), chunk_size)):
        chunk = [x for x in todos[i : i + chunk_size] if x[0] not in emb_keys]
        if len(chunk) == 0:
            continue
        emb = get_embeddings([x[0] for x in chunk])
        for x in chunk:
            emb_keys.add(x[0])
        db.insert(emb, chunk)
        db.save()
