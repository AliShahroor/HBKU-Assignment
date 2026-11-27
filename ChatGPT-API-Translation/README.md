# Assignment 6 - API Translation

This project translates exactly 200 Palestinian Arabic sentences into English through the OpenRouter API. The required experiment uses `openai/gpt-5.4-nano` with zero-shot and three-example few-shot prompts. The same complete experiment is repeated with `baidu/ernie-5.5-flash-preview:free`, and `nvidia/nemotron-3.5-lightning:free` is evaluated on a fixed 20-sentence subset.

## Main results

- Source sentences: 200
- Required OpenAI-model requests: 400
- Missing translations: 0
- Zero-shot BLEU / chrF++: `59.42` / `76.49`
- Few-shot BLEU / chrF++: `68.37` / `81.63`
- Baidu full-set requests: 400; BLEU / chrF++: `73.71` / `85.12` (few-shot)
- Nemotron subset requests: 40; BLEU / chrF++: `68.71` / `82.38` (few-shot)

## Files

- `prepare_data.py`: reconstructs and selects exactly 200 unique sentences from the Palestinian Digital Campus corpus.
- `sentences_ar.txt`: the 200 Arabic source sentences.
- `sentences.json`: sentence IDs and evaluation-subset flags.
- `evaluation_references.json`: 20 manually reviewed English references.
- `run_translation.py`: resumable zero-shot and few-shot API program.
- `evaluate_results.py`: completeness, latency, BLEU, and chrF++ evaluation.
- `translation_results.csv`: source sentences and all available translations.
- `translations_zero_shot.txt`: 200 required zero-shot outputs.
- `translations_few_shot.txt`: 200 required few-shot outputs.
- `evaluation_examples.csv`: the 20 reference examples and four system outputs.
- `evaluation_summary.json`: final metrics and usage totals.
- `Assignment6_Translation_API.ipynb`: Colab walkthrough with saved results.
- `Assignment6_ChatGPT_API.pdf`: final submission report.

The API key is never stored in this folder or committed to GitHub.

## Reproduce

Install the evaluation dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the key in the environment:

```bash
export OPENROUTER_API_KEY="your-key"
```

Run the workflow:

```bash
python prepare_data.py
python run_translation.py
python evaluate_results.py
```

`run_translation.py` saves each response immediately and skips completed sentence/prompt combinations when restarted. The main requests use eight workers; free-model requests are sequential and rate-limited.

## Prompting conditions

The zero-shot prompt provides only the translation instruction and source sentence. The few-shot prompt adds three Palestinian Arabic-English demonstrations. Both conditions use temperature `0` and request only the English translation.
