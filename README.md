# Is my problem new?
A simple semantic search engine on competitive programming problems.
<a href="http://yuantiji.ac" target="_blank" style="color: blue">http://yuantiji.ac</a> | <a href="https://www.buymeacoffee.com/fjzzq2002" target="_blank" style="color: brown">Buy me a boba</a>

<img src="logo.gif" style="zoom:50%;" />

**Update (2024/7/16):** It has been a long time :) Reorganized problems path. Switched LLM / embedder to [Gemma 2 9B](https://huggingface.co/google/gemma-2-9b-it) hosted by [together.ai](https://docs.together.ai) and [voyage-large-2-instruct](https://docs.voyageai.com/docs/pricing). Tweaked the prompt a little bit. Bought a new domain (see the link above) and switched to [vjudge](https://vjudge.net) as data source. See branch `old_ver` or history commits for the previous version.

**Update (2024/5/19):** Added AtCoder. Thanks [@fstqwq](https://github.com/fstqwq) for the contribution!

#### How does this work?

This idea is simple:

1. Simplify the statement & remove background by prompting LLM.

2. Embed the simplified documents and queries to perform vector searches.

It only happens recently that both models are good and cheap enough.

This pipeline is also not limited, of course, to competitive programming problems. You can use it to search for any kind of documents by modifying the prompt.

#### Deploy

You will need API keys from OpenAI, Together and Voyage. You can check their pricings online.

Put problems in `problems/` folder following the given example (`problems/1000.json`). Naming could be arbitrary and you could also have nested folders. Run `python -m src.build_summary` to get paraphrased statements, run `python -m src.build_embedding` to build embeddings and run `python -m src.build_locale` to detect language of problems. Finally, run `python -m src.ui` to start serving.

For large-scale running decent CPUs are needed as doing vector searching is CPU-dense. You might also want to modify `max_workers` in `src/ui.py`.

Due to copyright concerns we're not providing scrapped vjudge problems and vjudge scrapper. Sorry D: We also did not process the statements in PDF. If you have problems you want to add that are not supported in vjudge feel free to contact me or send PR and I'll see what I can do (would be perfect if you can just send over a zip in the correct format xd).

For reference, adding all ~160k problems from vjudge cost ~$60 and as of writing the deployed site is running on a 8vCPU server.
