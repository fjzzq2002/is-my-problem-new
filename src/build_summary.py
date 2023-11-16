# run summarize for all the problems
# use the chatgpt api
import requests
import json
from .utils import read_problems, problems_filenames, dump_json_safe
from openai import AsyncOpenAI
import hashlib
import asyncio

# from tqdm import tqdm
import time

start_time = time.time()


with open("settings.json") as f:
    settings = json.load(f)

client = AsyncOpenAI(
    api_key=settings["OPENAI_API_KEY"],
)


def check_processed(p, template):
    ORIGINAL = "\n" + p["statement"] + "\n"
    prompt = template.replace("[[ORIGINAL]]", ORIGINAL).strip()
    prompt_md5 = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    for f in p["processed"]:
        if f["prompt_md5"] == prompt_md5:
            return True
    return False


async def process(p, template):
    ORIGINAL = "\n" + p["statement"] + "\n"
    prompt = template.replace("[[ORIGINAL]]", ORIGINAL).strip()
    template_md5 = hashlib.md5(template.encode("utf-8")).hexdigest()
    prompt_md5 = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    already_processed = False
    for f in p["processed"]:
        if f["prompt_md5"] == prompt_md5:
            already_processed = True
    if already_processed:
        return
    # print(prompt, prompt_md5)
    # print(num_tokens_from_string(prompt))
    result = None
    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            timeout=40,
            max_tokens=1500,
            model="gpt-3.5-turbo",  # GPT-3.5-turbo-0613
        )
        assert chat_completion.choices[0].finish_reason == "stop"
        result = chat_completion.choices[0].message.content
        num_tokens_spent = chat_completion.usage.total_tokens
        print(f"Number of tokens spent: {num_tokens_spent}")
    except Exception as e:
        print("Error while prompting:", e)
    if result is None:
        return
    # print(result)
    p["processed"].append(
        {
            "prompt_md5": prompt_md5,
            "template_md5": template_md5,
            "result": result,
        }
    )


async def process_all_problems():
    for problems_file in problems_filenames():
        print("Building summary for", problems_file + "...")
        problems = read_problems("problems/" + problems_file)
        tasks = []
        batch_size = 25
        for template in settings["TEMPLATES"]:
            batch_tasks = []
            for p in problems:
                if "processed" not in p:
                    p["processed"] = []
                if check_processed(p, template):
                    continue
                batch_tasks.append(process(p, template))
                if len(batch_tasks) >= batch_size:
                    tasks.append(batch_tasks)
                    batch_tasks = []
            if len(batch_tasks):
                tasks.append(batch_tasks)
        print(len(tasks), "batches")
        # wait for all tasks to finish
        for batch_id, batch_tasks in enumerate(tasks):
            # somehow tqdm wasn't working that well for me
            print(
                "batch",
                batch_id,
                "of",
                len(tasks),
                f"  elapsed {time.time()-start_time:.2f}s",
            )
            time_start = time.time()
            await asyncio.gather(*batch_tasks)
            # handle the rate limit
            token_per = 2000
            token_limit = settings["OPENAI_TPM_LIMIT"]
            expected_time = token_per / (token_limit / 60.0) * len(batch_tasks)
            sleep = max(0, expected_time - (time.time() - time_start))
            print(f"finished batch, sleeping {sleep:.2f}s")
            time.sleep(sleep)
            # save the file
            dump_json_safe(problems, "problems/" + problems_file)
            print("saved!")


asyncio.run(process_all_problems())
