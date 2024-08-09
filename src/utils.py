import json
import pickle
import os
import shutil
import typing
import bs4
import numpy as np
import tempfile


# https://stackoverflow.com/a/66835172
def get_text(tag: bs4.Tag) -> str:
    _inline_elements = {
        "a",
        "span",
        "em",
        "strong",
        "u",
        "i",
        "font",
        "mark",
        "label",
        "s",
        "sub",
        "sup",
        "tt",
        "bdo",
        "button",
        "cite",
        "del",
        "b",
        "a",
        "font",
    }

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

def cleanup_str(s: str, allow_double_breaks = False) -> str:
    s = '\n'.join( line.strip() for line in s.splitlines() ).strip()
    # remove redundant linebreaks
    while True:
        if allow_double_breaks:
            ss = s.replace('\n\n\n','\n\n')
        else:
            ss = s.replace('\n\n','\n')
        if ss == s: break
        s = ss
    return s

def read_problem(filename):
    # read as a json
    with open(filename, encoding='utf-8') as f:
        return json.load(f)

def problem_filenames(path='problems/'):
    for root, dirs, files in os.walk(path):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            yield os.path.join(root, filename)


def list_problems(embed = False):
    # list all problems under problems/
    for problem_filename in problem_filenames():
        assert problem_filename.endswith(".json")
        if not embed:
            yield problem_filenames
            continue
        npy_filename = problem_filename[:-5] + ".npy"
        if os.path.exists(npy_filename):
            yield read_problem(problem_filename), np.load(npy_filename)


def dump_json_safe(obj, filename):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(obj, f)
    shutil.move(f.name, filename)


def dump_json_safe_utf8(obj, filename):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
    shutil.move(f.name, filename)

def dump_pickle_safe(obj, filename):
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        pickle.dump(obj, f, protocol=4)
    shutil.move(f.name, filename)


def dump_numpy_safe(obj, filename):
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        np.save(f, obj)
    shutil.move(f.name, filename)
