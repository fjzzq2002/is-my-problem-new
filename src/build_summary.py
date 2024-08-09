# run summarize for all the problems
# use the chatgpt api
import requests
import json
from .utils import read_problem, problem_filenames, dump_json_safe, dump_json_safe_utf8
from openai import AsyncOpenAI
from together import AsyncTogether
import anthropic
import hashlib
import asyncio
from tqdm.auto import tqdm

# from tqdm import tqdm
import time

start_time = time.time()


with open("settings.json") as f:
    settings = json.load(f)

client = AsyncTogether(
    api_key=settings['TOGETHER_API_KEY'],
)


def check_processed(p, template):
    ORIGINAL = p["statement"]
    prompt = template.replace("[[ORIGINAL]]", ORIGINAL).strip()
    prompt_md5 = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
    for f in p["processed"]:
        if f["prompt_md5"][:8] == prompt_md5:
            return True
    return False


async def process(p, template, delay = 0):
    # sleep for delay first
    await asyncio.sleep(delay)
    ORIGINAL = p["statement"]
    prompt = template.replace("[[ORIGINAL]]", ORIGINAL).strip()
    template_md5 = hashlib.md5(template.encode("utf-8")).hexdigest()[:8]
    prompt_md5 = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
    already_processed = False
    for f in p["processed"]:
        if f["prompt_md5"][:8] == prompt_md5:
            already_processed = True
    if already_processed:
        return
    # print(prompt, prompt_md5)
    # print(num_tokens_from_string(prompt))
    result = None
    try:
        response = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
                { "role": "assistant", "content": "Simplified statement:" }
            ],
            model="google/gemma-2-9b-it",
        )
#        assert chat_completion.stop_reason=='end_turn'
        result = response.choices[0].message.content.strip()
        print(f"Number of tokens spent: {response.usage.total_tokens}")
    except Exception as e:
        print("Error while prompting:", e)
    if result is None:
        return []
    return [
        {
            "prompt_md5": prompt_md5,
            "template_md5": template_md5,
            "result": result,
        }
    ]


async def process_all_problems():
    # apparently some mysterious OJs are spending my money ;_;
    goodojs = ['UOJ', 'Codeforces', '洛谷', 'DMOJ', 'HDU', 'CodeChef', 'AtCoder', 'LibreOJ', 'TopCoder', 'SPOJ', '51Nod', '黑暗爆炸', 'UVA'] #, 'USACO'
    badojs = ['HYSBZ', 'BZOJ']
    fns = sorted(list(problem_filenames()),key=lambda x:int(not any(goodoj.lower() in x.lower() for goodoj in goodojs))+int(any(badoj.lower() in x.lower() for badoj in badojs)))
    chunk_size = 50
    gap_every = 1/8.5
    problem_files = []
    for problem_file_cur in tqdm(fns):#tqdm(range(0,len(fns),chunk_size)):
        try:
            p = read_problem(problem_file_cur)
        except Exception as e:
            print('error',problem_file_cur,e)
            continue
        need_work = False
        for template in settings["TEMPLATES"]:
            if 'processed' in p and check_processed(p, template):
                continue
            need_work = True
        if need_work:
            problem_files.append(problem_file_cur)
        if len(problem_files) >= chunk_size or problem_file_cur == fns[-1]:
            for template in settings["TEMPLATES"]:
                t0 = time.time()
                tasks = []
                notprocessed = []
                for idx, problem_file in enumerate(problem_files):
                    p = read_problem(problem_file)
                    if "processed" not in p:
                        p["processed"] = []
                    if check_processed(p, template):
                        continue
                    notprocessed.append(problem_file)
                    tasks.append(process(p, template, idx * gap_every))
                if not len(tasks):
                    continue
                WAIT = chunk_size * gap_every + .5
                results = await asyncio.gather(*tasks)
                for problem_file, result in zip(notprocessed, results):
                    if not len(result):
                        WAIT = 6
                        continue
                    p = read_problem(problem_file)
                    if "processed" not in p:
                        p["processed"] = []
                    p["processed"].extend(result)
                    print(problem_file)
                    dump_json_safe_utf8(p, problem_file)
                t1 = time.time()
                print('time elapsed',t1-t0)
                # wait till WAIT
                if t1-t0 < WAIT:
                    await asyncio.sleep(WAIT-(t1-t0))
            problem_files = []

if __name__ == "__main__":
    asyncio.run(process_all_problems())