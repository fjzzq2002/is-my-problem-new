# Is my problem new?
A simple semantic search engine on competitive programming problems.

#### Screenshots

![](screenshots/demo1.png)

![](screenshots/demo0.png)

![](screenshots/demo2.png)

#### How does this work?

This idea is simple:

1. Simplify the statement & remove background by prompting chatgpt.

2. Embed the simplified documents and queries to perform vector searches.

It only happens recently that both models are good and cheap enough.

This pipeline is also not limited, of course, to competitive programming problems. You can use it to search for any kind of documents by modifying the prompt.

#### Deploy

*Disclaimer:* This project was finished in ~2 days so do not expect high quality code. Deploy at your own risk. If you want to serve it publicly, add safety measures such as rate limits, authentication, etc.

You will need API keys from OpenAI (https://platform.openai.com/) and Cohere (https://cohere.com/). The model `gpt-3.5-turbo` (also known as chatgpt) from OpenAI and `cohere-embed-english-v3.0` from Cohere (empirically works better than OpenAI's embeddings at the same price) are used. You can check their pricings online.

1. Copy `settings_sample.json` to `settings.json`. Fill in the API keys.

2. Download embeddings from [here]() or run `python -m src.build_embedding.py` (regenerate embeddings using cohere, costs ~$0.5).

3. Run `python -m src.ui.py` to start your server!

*Can we make this completely open-source?* You will need to replace the summary model and the embedding model with open source versions. For embedding models, [this leaderboard](https://huggingface.co/spaces/mteb/leaderboard) may be a good place to look for.

#### Adding a new set of problems

1. Generate a json file in `problems/` folder. You should supply `uid`, `url`, `tag` and `statement` fields. See `problems/format.md` for detailed specifications and see `src/scrapper/` for two examples: a crawler for Codeforces and a parser for bzoj (using a local dump).

2. Generate summaries with ChatGPT. Run `python -m src.build_summary.py`.

3. Generate embeddings with Cohere. Run `python -m src.build_embedding.py`.

For reference, adding all problems from codeforces & bzoj costs about $20.

*Known issue:* The OpenAI api has a rate limit and to make sure our requests fit in the limit, currently the code sleeps after requests following to the rate limit. You may want to adjust the logic to increase throughput.