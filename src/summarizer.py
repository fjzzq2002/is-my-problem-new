# run summarize for all the problems
# use the chatgpt api
import requests
import json
from .utils import read_problems, problems_filenames, dump_json_safe
from openai import AsyncOpenAI
import hashlib
import asyncio
#from tqdm import tqdm
import time

start_time = time.time()


with open('settings.json') as f:
    settings = json.load(f)

client = AsyncOpenAI(
    api_key=settings['OPENAI_API_KEY'],
)


async def process(p,template):
    ORIGINAL = '\n'+p['statement']+'\n'
    prompt = template.replace('[[ORIGINAL]]',ORIGINAL).strip()
    template_md5 = hashlib.md5(template.encode('utf-8')).hexdigest()
    prompt_md5 = hashlib.md5(prompt.encode('utf-8')).hexdigest()
    alreday_processed = False
    for f in p['processed']:
        if f['prompt_md5'] == prompt_md5:
            alreday_processed = True
    if alreday_processed:
        return
    #print(prompt, prompt_md5)
    #print(num_tokens_from_string(prompt))
    result = None
    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            timeout = 40,
            max_tokens = 1500,
            model="gpt-3.5-turbo",
        )
        assert chat_completion.choices[0].finish_reason == "stop"
        result = chat_completion.choices[0].message.content
        num_tokens_spent = chat_completion.usage.total_tokens
        print(f"Number of tokens spent: {num_tokens_spent}")
    except Exception as e:
        print('Error while prompting:', e)
    if result is None:
        return
    # print(result)
    p['processed'].append({
        'prompt_md5': prompt_md5,
        'template_md5': template_md5,
        'result': result,
    })

async def process_all_problems():
    for problems_file in problems_filenames():
        problems = read_problems('problems/'+problems_file)
        tasks = []
        batch_size = 15
        for i in range(0, len(problems), batch_size):
            batch = problems[i:i+batch_size]
            for template in settings['TEMPLATES']:
                batch_tasks = []
                for p in batch:
                    if 'processed' not in p:
                        p['processed']=[]
                    batch_tasks.append(process(p, template))
                tasks.append(batch_tasks)
        print(len(tasks),'batches')
        # wait for all tasks to finish
        for batch_id, batch_tasks in enumerate(tasks):
            # somehow tqdm wasn't working that well for me
            print('batch',batch_id,'of',len(tasks),f'  elapsed {time.time()-start_time:.2f}s')
            await asyncio.gather(*batch_tasks)
            # save the file
            dump_json_safe(problems, 'problems/'+problems_file)
            print('saved!')

asyncio.run(process_all_problems())