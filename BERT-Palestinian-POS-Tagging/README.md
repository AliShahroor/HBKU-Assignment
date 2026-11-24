# BERT for Palestinian POS Tagging

This folder contains the code used to fine-tune `aubmindlab/bert-base-arabertv02-twitter` for Palestinian Arabic POS tagging. The experiment extends Curras with a deterministic Palestinian Digital Campus corpus and compares the extended model with a controlled Curras-only baseline.

## Files

- `BERT_POS_Digital_Campus.ipynb`: Google Colab workflow and recommended starting point.
- `build_digital_campus_corpus.py`: generates the annotated extension and combined data splits.
- `train_and_evaluate.py`: trains either the extended or Curras-only model and saves evaluation metrics.
- `build_challenge_set.py`: creates the manually annotated Palestinian challenge set.
- `evaluate_challenge_set.py`: evaluates a saved model and writes token-level predictions.

Large checkpoints, generated datasets, result files, and report files are intentionally excluded from this code-only folder.

## Run in Google Colab

1. Open [`BERT_POS_Digital_Campus.ipynb`](https://colab.research.google.com/github/AliShahroor/HBKU-Assignment/blob/main/BERT-Palestinian-POS-Tagging/BERT_POS_Digital_Campus.ipynb) in Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Select **Runtime > Run all**. The notebook clones its scripts automatically, installs dependencies, downloads the class Curras files, creates the extension, trains both models, runs the challenge evaluation, and prints the final comparison table.

The notebook already displays the reference output from the completed experiment on GitHub. Running all cells replaces it with the results from the new Colab run. Training time and very small floating-point differences may depend on the hardware and package versions.

## Run from a terminal

Install the dependencies:

```bash
pip install torch "transformers==4.57.3" "datasets==3.6.0" "accelerate==1.12.0" "scikit-learn==1.8.0" gdown
```

After placing the Curras files under `data/Curras/`, run:

```bash
python3 build_digital_campus_corpus.py
python3 train_and_evaluate.py --epochs 2 --batch-size 12
python3 train_and_evaluate.py --training-data curras-only --epochs 2 --batch-size 12
python3 build_challenge_set.py
python3 evaluate_challenge_set.py
python3 evaluate_challenge_set.py --model-dir final_model_curras_only --output-stem challenge_curras_only
```

The default extended run and the Curras-only baseline use the same AraBERT checkpoint, 47-label Curras tagset, seed `42`, two epochs, batch size `12`, and learning rate `5e-5`. This keeps the comparison focused on the change in training data.

The notebook pins the tested library versions because newer development releases can change the `TrainingArguments` API.
