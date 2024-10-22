import numpy as np
from .embedder import VectorDB, processed_promptmd5
from .utils import read_problem
from tqdm.auto import tqdm
import gradio as gr
import json
import asyncio
from openai import AsyncOpenAI
from together import AsyncTogether
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import urllib
import time
import voyageai
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=8)

db = VectorDB()
db.load_all()
print("read", len(set(x[0] for x in db.metadata)), "problems")
print(db.metadata[:100])

with open("settings.json") as f:
    settings = json.load(f)

voyage_client = voyageai.Client(
    api_key=settings['VOYAGE_API_KEY'],
    max_retries=3,
    timeout=120,
)

openai_client = AsyncOpenAI(
    api_key=settings["OPENAI_API_KEY"],
)

together_client = AsyncTogether(
    api_key=settings['TOGETHER_API_KEY'],
)

async def querier_i18n(locale, statement, *template_choices):
    assert len(template_choices) % 3 == 0
    yields = []
    ORIGINAL = statement.strip()
    t1 = time.time()

    async def process_template(engine, prompt, prefix):
        if 'origin' in engine.lower() or '保' in engine.lower():
            return ORIGINAL
        if 'none' in engine.lower() or '跳' in engine.lower():
            return ''

        prompt = prompt.replace("[[ORIGINAL]]", ORIGINAL).strip()
        
        if "gemma" in engine.lower():
            response = await together_client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": prefix}
                ],
                model="google/gemma-2-27b-it",
            )
            return response.choices[0].message.content.strip()
        elif "gpt" in engine.lower():
            response = await openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                model="gpt-4o-mini"
            )
            return response.choices[0].message.content.strip().replace(prefix.strip(), '', 1).strip()
        else:
            raise NotImplementedError(engine)

    tasks = [process_template(template_choices[i], template_choices[i+1], template_choices[i+2]) 
             for i in range(0, len(template_choices), 3)]
    yields = await asyncio.gather(*tasks)

    t2 = time.time()
    print('query llm', t2-t1)
    response = voyage_client.embed(
        list(set(y.strip() for y in yields if len(y))),
        model="voyage-large-2-instruct",
        input_type='query'
    )
    print('Token spent',response.total_tokens)
    emb = [d for d in response.embeddings]
    t3 = time.time()
    print('query emb', t3-t2)

    loop = asyncio.get_running_loop()
    nearest = await loop.run_in_executor(executor, db.query_nearest, emb, 5000)
#    nearest = db.query_nearest(emb, k=5000)
    t4 = time.time()
    print('query nearest', t4-t3)

    sim = np.array([x[0] for x in nearest])
    ids = np.array([x[1] for x in nearest], dtype=np.int32)

    info = 'Fetched top ' + str(len(sim)) + ' matches! Go to the next tab to view results~' if locale == 'en' else \
           '已查找到前' + str(len(sim)) + '个匹配！进入下一页查看结果~'

    return [info, (sim, ids)] + yields


def format_problem_i18n(locale, uid, sim):
    def tr(en,zh):
        if locale == 'en': return en
        if locale == 'zh': return zh
        raise NotImplementedError(locale)
    # be careful about arbitrary reads
    uid = db.metadata[int(uid)][0]
    problem = read_problem(uid)
    statement = problem["statement"].replace("\n", "\n\n")
    # summary = sorted(problem.get("processed",[]), key=lambda t: t["template_md5"])
    # if len(summary):
    #     summary = summary[0]["result"]
    # else:
    #     summary = None
    title = problem['title']
    lang = problem.get('locale',('un', 'Unknown'))
    def to_flag(t,u):
        if t == 'un':
            # get a ? with border, 14x20
            return f"""<div style="display: inline-block; border: 1px solid black; width: 20px; text-align: center;" alt="{u}" title="{u}">?</div>"""
        else:
            return f"""<img
    src="https://flagcdn.com/w20/{t}.png"
    srcset="https://flagcdn.com/w40/{t}.png 2x"
    style="display: inline-block"
    height="14"
    width="20"
    title="{u}"
    alt="{u}" />"""
    # flag = ''.join(to_flag(t) for t in lang_mapper.values()) # debug only
    flag = to_flag(*lang)
    url = problem["url"]
    problemlink = uid.replace('/',' ').replace('\\',' ').strip().replace('problems vjudge','',1).strip().replace('_','-')
    assert problemlink.endswith('.json')
    problemlink = problemlink[:-5].strip()
    # markdown = f"# [{title} ({problemlink})]({url})\n\n"
    html = f'<p><span style="font-size:22px; font-weight: 500;">{title}</span>&nbsp;&nbsp;<span style="font-size:15px">{problemlink} ({round(sim*100)}%)</span></p>\n'
    link0 = 'https://www.google.com/search?'+urllib.parse.urlencode({'q': problemlink})
    link1 = 'https://www.google.com/search?'+urllib.parse.urlencode({'q': problem['source']+' '+title})
    link0_bd = 'https://www.baidu.com/s?'+urllib.parse.urlencode({'wd': problemlink})
    link1_bd = 'https://www.baidu.com/s?'+urllib.parse.urlencode({'wd': problem['source']+' '+title})
    # <a href="{link0}" target="_blank">{tr("Google2","谷歌2")}</a> 
    #  <a href="{link0_bd}" target="_blank">Baidu2</a>
    html += f'{flag}&nbsp;&nbsp;&nbsp;<a href="{url}" target="_blank">VJudge</a>&nbsp;&nbsp;<a href="{link1}" target="_blank">{tr("Google","谷歌")}</a>&nbsp;&nbsp;<a href="{link1_bd}" target="_blank">{tr("Baidu","百度")}</a>'
    markdown = ''
    rsts = []
    for template in settings['TEMPLATES']:
        md5 = processed_promptmd5(problem['statement'], template)
        rst = None
        for t in problem.get("processed",[]):
            if t["prompt_md5"][:8] == md5:
                rst = t["result"]
        if rst is not None:
            rsts.append(rst)
    rsts.sort(key=len)
    for idx, rst in enumerate(rsts):
        markdown += f'### {tr("Summary", "简要题意")} {idx+1}\n\n{rst}\n\n'
    if markdown != '':
        markdown += '<br/>\n\n'
    markdown += f'### {tr("Raw Statement", "原始题面")}\n\n{statement}'
    return html, markdown

def get_block(locale):
    def tr(en,zh):
        if locale == 'en': return en
        if locale == 'zh': return zh
        raise NotImplementedError(locale)

    with gr.Blocks(
        title=tr("Is my problem new?","原题机"), css="""
        .mymarkdown {font-size: 15px !important}
        footer{display:none !important}
        .centermarkdown{text-align:center !important}
        .pagedisp{text-align:center !important; font-size: 20px !important}
        .realfooter{color: #888 !important; font-size: 14px !important; text-align: center !important;}
        .realfooter a{color: #888 !important;}
        .smallbutton {min-width: 30px !important;}
        """,
        head=settings.get('CUSTOM_HEADER','')
    ) as demo:
        gr.Markdown(
            tr("""
        # Is my problem new?
        A semantic search engine for competitive programming problems.
        ""","""
# 原题机
原题在哪里啊，原题在这里~"""
        ))
        with gr.Tabs() as tabs:
            with gr.TabItem(tr("Search",'搜索'),id=0):
                input_text = gr.TextArea(
                    label=tr("Statement",'题目描述'),
                    info=tr("Paste your statement here!",'在这里粘贴你要搜索的题目！'),
                    value=tr("Calculate the longest increasing subsequence of the input sequence.",
                             '计算最长上升子序列长度。'),
                )
                bundles = []
                with gr.Accordion(tr("Rewriting Setup (Advanced)","高级设置"), open=False):
                    gr.Markdown(tr("Several rewritten version of the original statement will be calculated and the maximum embedding similarity is used for sorting.",
                                   "输入的问题描述将被重写为多个版本并计算与每个原问题的最大相似度。"))
                    for template_id in range(5):
                        with gr.Accordion(tr("Template ",'版本 ')+str(template_id+1)):
                            with gr.Row():
                                with gr.Group():
                                    template = settings['TEMPLATES'][(template_id-1)%2] if template_id in [1,2] else None
                                    # engines = [tr("Keep Original",'保留原描述'), "Gemma 2 (27B)", "GPT4o Mini", tr('None', '跳过该版本')]
                                    # engine = gr.Radio(
                                    #     engines,
                                    #     label=tr("Engine",'使用的语言模型'),
                                    #     value=engines[-1] if template is None else engines[1] if template_id<=2 else engines[2],
                                    #     interactive=True,
                                    # )
                                    engines = [tr("Keep Original",'保留原描述'), "GPT4o Mini", tr('None', '跳过该版本')]
                                    engine = gr.Radio(
                                        engines,
                                        label=tr("Engine",'使用的语言模型'),
                                        value=engines[-1] if template is None else engines[1],# if template_id<=2 else engines[2],
                                        interactive=True,
                                    )
                                    prompt = gr.TextArea(
                                        label=tr("Prompt ([[ORIGINAL]] will be replaced)",'提示词 ([[ORIGINAL]] 将被替换为问题描述)'),
                                        value=template if template is not None else settings['TEMPLATES'][0],
                                        interactive=True,
                                        visible=template is not None,
                                    )
                                    prefix = gr.Textbox(
                                        label=tr("Prefix", '回复前缀'),
                                        value="Simplified statement:",
                                        interactive=True,
                                        visible=template is not None,
                                    )
                                    # hide these when engine has wrong value
                                    engine.change(lambda engine: (gr.update(visible=any(s in engine.lower() for s in ['gpt','gemma'])),)*2, engine, [prompt, prefix])
                                output_text = gr.TextArea(
                                    label=tr('Output','重写结果'),
                                    value="",
                                    interactive=False,
                                )
                        bundles.append((engine, prompt, prefix, output_text))
                search_result = gr.State(([],[]))
                submit_button = gr.Button(tr("Search!",'搜索！'))
                status_text = gr.Markdown("", elem_classes="centermarkdown")
            with gr.TabItem(tr("View Results",'查看结果'),id=1):
                cur_idx = gr.State(0)
                num_columns = gr.State(1)
                ojs = [f'{t} ({c})' for t,c in sorted(db.sources.items())]
                oj_dropdown = gr.Dropdown(
                    ojs, value=ojs, multiselect=True, label=tr("Displayed OJs",'展示的OJ'),
                    info=tr('Problems from OJ not in this list will be ignored.',
                            '不在这个列表里的OJ的题目将被忽略。可以在这里删掉你不认识的OJ。'),
                )
                # on change, change cur_idx to 1
                oj_dropdown.change(lambda: 0, None, cur_idx)
                statement_min_len = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    label=tr("Minimum Statement Length",'最小题面长度'),
                    value=20,
                    info=tr('The statements shorter than this after removing digits + blanks will be ignored. Useful for filtering out meaningless statements.',
                            '去除数字和空白字符后题面长度小于该值的题目将被忽略。可以用来筛掉一些奇怪的题面。'),
                )

                with gr.Row():
                    # home_page = gr.Button("H")
                    add_column = gr.Button("+", elem_classes='smallbutton')
                    prev_page = gr.Button("←", elem_classes='smallbutton')
                    home_page = gr.Button("H", elem_classes='smallbutton')
                    next_page = gr.Button("→", elem_classes='smallbutton')
                    remove_column = gr.Button("-", elem_classes='smallbutton')
                    # bind to cur_page and num_columns
                    # home_page.click(lambda: 1, None, cur_page)
                    prev_page.click(lambda cur_idx, num_columns: max(cur_idx - num_columns, 0), [cur_idx, num_columns], cur_idx, concurrency_limit=None)
                    next_page.click(lambda cur_idx, num_columns: cur_idx + num_columns, [cur_idx, num_columns], cur_idx, concurrency_limit=None)
                    home_page.click(lambda: 0, None, cur_idx, concurrency_limit=None)
                    def adj_idx(idx, col):
                        return int(round(idx / col)) * col
                    add_column.click(lambda cur_idx, num_columns: (adj_idx(cur_idx, num_columns + 1), num_columns + 1), [cur_idx, num_columns], [cur_idx, num_columns], concurrency_limit=None)
                    remove_column.click(lambda cur_idx, num_columns: (adj_idx(cur_idx, num_columns - 1), num_columns - 1) if num_columns >1 else (cur_idx, num_columns), [cur_idx, num_columns], [cur_idx, num_columns], concurrency_limit=None)


                @gr.render(inputs=[search_result, oj_dropdown, cur_idx, num_columns, statement_min_len], concurrency_limit=None)
                def show_OJs(search_result, oj_dropdown, cur_idx, num_columns, statement_min_len):
                    allowed_OJs = set([oj[:oj.find(' (')] for oj in oj_dropdown])
                    tot = 0
                    # print(len(search_result[0]),len(search_result[1]))
                    for sim, idx in zip(search_result[0], search_result[1]):
                        if db.metadata[idx][1] not in allowed_OJs or db.metadata[idx][2] < statement_min_len:
                            continue
                        tot += 1
                    gr.Markdown(tr(f"Page {round(cur_idx/num_columns)+1} of {(tot+num_columns-1)//num_columns} ({num_columns} per page)",
                                   f'第 {round(cur_idx/num_columns)+1} 页 / 共 {(tot+num_columns-1)//num_columns} 页 (每页显示 {num_columns} 个)'),
                                   elem_classes="pagedisp")
                    cnt = 0
                    with gr.Row():
                        for sim, idx in zip(search_result[0], search_result[1]):
                            if db.metadata[idx][1] not in allowed_OJs or db.metadata[idx][2] < statement_min_len:
                                continue
                            cnt += 1
                            if cur_idx+1 <= cnt:
                                if cnt > cur_idx+num_columns: break
                                with gr.Column(variant='compact'):
                                    html, md = format_problem_i18n(locale, idx, sim)
                                    gr.HTML(html)
                                    gr.Markdown(
                                        latex_delimiters=[
                                            {"left": "$$", "right": "$$", "display": True},
                                            {"left": "$", "right": "$", "display": False},
                                            {"left": "\\(", "right": "\\)", "display": False},
                                            {"left": "\\[", "right": "\\]", "display": True},
                                        ],
                                        value=md,
                                        elem_classes="mymarkdown",
                                    )
            if 'CUSTOM_ABOUT_PY' in settings and settings['CUSTOM_ABOUT_PY'].endswith('.py'):
                with gr.TabItem(tr("About",'关于'),id=2):
                    with open(settings['CUSTOM_ABOUT_PY'], 'r', encoding='utf-8') as f: eval(f.read())

        # add a footer
        gr.HTML(
            """<div class="realfooter">Built with ❤️ by <a href="https://github.com/fjzzq2002">@TLE</a></div>"""
        )
        async def async_querier_wrapper(*args):
            result = await querier_i18n(locale, *args)
            return (gr.Tabs(selected=1),) + tuple(result)
        submit_button.click(
            fn=async_querier_wrapper,
            inputs=sum([list(t[:-1]) for t in bundles], [input_text]),
            outputs=[tabs, status_text, search_result] + [t[-1] for t in bundles],
            concurrency_limit=7,
        )
        # output_labels.select(fn=show_problem, inputs=None, outputs=[my_markdown])
    return demo



app = FastAPI()
favicon_path = 'favicon.ico'

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)
@app.get("/", response_class=HTMLResponse)
async def read_main():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Is my problem new?</title>
        <script type="text/javascript">
            window.onload = function() {
                var userLang = navigator.language || navigator.userLanguage;
                if (userLang.startsWith('zh')) {
                    window.location.href = "/zh";
                } else {
                    window.location.href = "/en";
                }
            }
        </script>
    </head>
    <body>
        <p>Redirecting based on your browser's locale...</p>
        <p><a href="/en">English</a> | <a href="/zh">中文</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

app = gr.mount_gradio_app(app, get_block('zh'), path="/zh")
app = gr.mount_gradio_app(app, get_block('en'), path="/en")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
