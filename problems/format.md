## Format of the problem json files

Each file (e.g. `codeforces.json`) contains an array of problems. Each problem has the following fields:

- `uid`: The unique identifier of the problem. Will be displayed to the users. Make sure it doesn't duplicate across different problem sets.
- `url`: The url to access the original problem. Will be displayed to the users.
- `tag`: The tag of the problem. Might be different across different problem sets.
- `statement`: The original problem statement scrapped.
  - Use single $ to represent inline math, and double $$ to represent block math.
- `processed`: Processed problem statements.
  - For each summarization, there are three fields: `prompt_md5`, `template_md5` and `result`.
  - `prompt_md5` is the md5 of the exact prompt sent to openai.
  - `template_md5` is the md5 of the prompt template.