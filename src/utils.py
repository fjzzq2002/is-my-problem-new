import json
import pickle
import os
import shutil
import typing
import bs4

# https://stackoverflow.com/a/66835172
def get_text(tag: bs4.Tag) -> str:
    _inline_elements = {"a","span","em","strong","u","i","font","mark","label",
    "s","sub","sup","tt","bdo","button","cite","del","b","a","font",}

    def _get_text(tag: bs4.Tag) -> typing.Generator:
        for child in tag.children:
            if isinstance(child, bs4.Tag):
                # if the tag is a block type tag then yield new lines before after
                is_block_element = child.name not in _inline_elements
                if is_block_element:
                    yield "\n"
                yield from ["\n"] if child.name == "br" else _get_text(child)
                if is_block_element:
                    yield "\n"
            elif isinstance(child, bs4.NavigableString):
                yield child.string
    return "".join(_get_text(tag))

def read_problems(filename):
    # read as a json
    with open(filename) as f:
        problems = json.load(f)
        return [x for x in problems if len(x['statement'].strip())>=5]
    
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
    shutil.move(f.name, filename)

def dump_pickle_safe(obj, filename):
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        pickle.dump(obj, f, protocol=4)
    shutil.move(f.name, filename)

def dump_numpy_safe(obj, filename):
    import numpy as np
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        np.save(f, obj)
    shutil.move(f.name, filename)