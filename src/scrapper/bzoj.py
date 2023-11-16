from ..utils import read_problems, dump_json_safe, get_text
from bs4 import BeautifulSoup
import git
import os
from tqdm.auto import tqdm

if not os.path.exists("tmp/my_bzoj"):
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    os.system("git clone https://github.com/linkfqy/my_bzoj.git tmp/my_bzoj")

problems = []
for f in tqdm(os.listdir("tmp/my_bzoj/p")):
    pid = f[:4]
    if f != pid + ".html":
        continue
    # read file
    with open("tmp/my_bzoj/p/" + f, "rb") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    statement = soup.find(class_="card-inner")
    title = soup.find(class_="content-heading")
    title = get_text(title).strip()
    text = "\n" + get_text(statement) + "\n"
    text = text.replace("\n题目描述\n", "")
    for t in ["输入格式", "输出格式", "样例输入", "样例输出", "提示", "题目来源"]:
        f = text.find("\n" + t + "\n")
        if f != -1:
            text = text[:f]
    text = text.replace("\n\n", "\n").replace("\n\n", "\n").strip()
    problem = {
        "uid": "BZOJ" + pid,
        "url": "https://darkbzoj.cc/problem/" + pid,
        "tags": [],
        "title": title,
        "statement": text,
    }
    problems.append(problem)

dump_json_safe(problems, "problems/bzoj.json")
