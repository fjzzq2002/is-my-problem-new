import json
import pickle
import os

def read_problems(filename):
    # read as a json
    with open(filename) as f:
        return json.load(f)
    
def problems_filenames():
    for filename in os.listdir('problems'):
        if filename.endswith('.json'):
            yield filename

def read_all_problems():
    # list all problems under problems/
    problems = []
    for filename in problems_filenames():
        problems+=read_problems('problems/'+filename)
    return problems

def dump_json_safe(obj, filename):
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        json.dump(obj, f)
    os.replace(f.name, filename)

def dump_pickle_safe(obj, filename):
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(f.name, filename)

def dump_numpy_safe(obj, filename):
    import numpy as np
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        np.save(f, obj)
    os.replace(f.name, filename)