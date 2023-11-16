from .embedder import VectorDB, get_embeddings
from .utils import read_problems, problems_filenames
from tqdm.auto import tqdm
import gradio as gr
import json
from openai import OpenAI

db = VectorDB().load()
emb_keys = set([x[0] for x in db.metadata])
print("read", len(emb_keys), "embeddings from db")
problems = {}
for f in problems_filenames():
    for p in read_problems("problems/" + f):
        problems[p["uid"]] = p
print("read", len(problems), "problems from db")

with open("settings.json") as f:
    settings = json.load(f)

client = OpenAI(
    api_key=settings["OPENAI_API_KEY"],
)


def querier(statement, template_choice, topk):
    # print(statement, template_choice)
    paraphrased = statement
    if "None" not in template_choice:
        template_id = int(template_choice.split(" ")[1]) - 1
        template = settings["TEMPLATES"][template_id]
        ORIGINAL = "\n" + statement + "\n"
        prompt = template.replace("[[ORIGINAL]]", ORIGINAL).strip()
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            timeout=40,
            max_tokens=1500,
            model="gpt-3.5-turbo",
        )
        assert chat_completion.choices[0].finish_reason == "stop"
        paraphrased = chat_completion.choices[0].message.content
    emb = get_embeddings([paraphrased])[0]
    # query nearest
    nearest = db.query_nearest(emb, k=topk)
    # print(nearest)
    return paraphrased, {b[1]: a for a, b in nearest}


def show_problem(evt: gr.SelectData):  # SelectData is a subclass of EventData
    uid = evt.value
    statement = problems[uid]["statement"].replace("\n", "\n\n")
    summary = sorted(problems[uid]["processed"], key=lambda t: t["template_md5"])
    if len(summary):
        summary = summary[0]["result"]
    else:
        summary = None
    title = uid  # problems[uid]['title']
    url = problems[uid]["url"]
    markdown = f"# [{title}]({url})\n\n"
    if summary is not None:
        markdown += f"### Summary (auto-generated)\n\n{summary}\n\n"
    markdown += f"### Statement\n\n{statement}"
    return markdown


with gr.Blocks(
    title="Is my problem new?", css=".mymarkdown {font-size: 15px !important}"
) as demo:
    gr.Markdown(
        """
    # Is my problem new?
    A semantic search engine for competitive programming problems.
    """
    )
    with gr.Row():
        # column for inputs
        with gr.Column():
            input_text = gr.Textbox(
                label="Statement",
                info="Paste your statement here!",
                value="Calculate the longest increasing subsequence of the input sequence.",
            )
            template_type = gr.Radio(
                ["Template " + str(x + 1) for x in range(len(settings["TEMPLATES"]))]
                + ["None (faster)"],
                label="Paraphrase with chatgpt?",
                value="Template 2",
            )
            topk_slider = gr.Slider(
                minimum=1,
                maximum=100,
                step=1,
                value=10,
                label="Number of similar problems to show",
            )
            submit_button = gr.Button("Submit")
            my_markdown = gr.Markdown(
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                    {"left": "\\(", "right": "\\)", "display": False},
                    {"left": "\\[", "right": "\\]", "display": True},
                ],
                elem_classes="mymarkdown",
            )
        # column for outputs
        with gr.Column():
            output_text = gr.Textbox(
                label="Paraphrased",
                value="",
            )
            output_labels = gr.Label(
                label="Similar problems",
            )
    submit_button.click(
        fn=querier,
        inputs=[input_text, template_type, topk_slider],
        outputs=[output_text, output_labels],
    )
    output_labels.select(fn=show_problem, inputs=None, outputs=[my_markdown])

demo.launch()
