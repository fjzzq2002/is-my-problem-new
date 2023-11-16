from ..utils import read_problems, dump_json_safe, get_text
import json
import os
import requests
import time
from bs4 import BeautifulSoup
from tqdm.auto import tqdm


scrapped_problems = []
try:
    scrapped_problems = read_problems("problems/codeforces.json")
    print(f"Recalled {len(scrapped_problems)} scrapped problems")
except:
    print("Cannot find scrapped problems")
scrapped_uids = set(p["uid"] for p in scrapped_problems)

codeforces_endpoint = "https://codeforces.com/api/problemset.problems"
# get list of problems
list_problems = requests.get(codeforces_endpoint).json()["result"]["problems"]
# the website is down, read problems.txt instead
# with open('problems.txt') as f:
#     list_problems = json.load(f)['result']['problems']
print("# problems:", len(list_problems))


# a scrapper for codeforces
def scrap_problem(contestId, index, rating, tags, uid):
    url = f"https://codeforces.com/contest/{contestId}/problem/{index}"
    response = requests.get(url, timeout=30)
    soup = BeautifulSoup(response.content, "html.parser")
    statement = soup.find(class_="problem-statement")
    try:
        statement.find(class_="header").decompose()
    except:
        pass
    statement_body = statement.find("div")
    statement_body = get_text(statement_body)
    # \r -> \n, remove duplicate \n, strip
    statement_body = (
        statement_body.replace("\r", "\n")
        .replace("\n\n", "\n")
        .replace("$$$", "$")
        .strip()
    )
    problem = {
        "uid": uid,
        "url": url,
        "tags": tags,
        #        'raw': str(response.content),
        "statement": statement_body,
        "contestId": contestId,
        "index": index,
        "rating": rating,
    }
    return problem


for problem in tqdm(list_problems):
    contestId, index, rating, tags = (
        problem["contestId"],
        problem["index"],
        problem.get("rating", -1),
        problem["tags"],
    )
    uid = f"Codeforces{contestId}{index}"
    if uid in scrapped_uids:
        continue
    print(f"Scrapping {uid}")
    result = None
    try:
        result = scrap_problem(contestId, index, rating, tags, uid)
    except Exception as e:
        print("Error while scrapping:", e)
    if result is not None:
        scrapped_problems.append(result)
    time.sleep(0.1)
    # save to file every 10 problems
    import random

    if random.random() < 0.1:
        dump_json_safe(scrapped_problems, "problems/codeforces.json")

dump_json_safe(scrapped_problems, "problems/codeforces.json")
