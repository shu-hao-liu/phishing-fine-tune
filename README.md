# Phishing model training

The project has one configuration file and one folder for CSV files:

```text
phishing-detect/
├── config.yaml
├── finetune.py
├── inference.py
├── analyze_results.py
└── data/
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    ├── inference.csv                 # original before the validation/test split
    └── *.original.csv, *.before-empty-removal.csv  # backups
```

`data/` also receives prediction CSVs. Other scripts in the project are utilities and tests. CSV files are excluded from Git, so upload the `data/` folder to the Pod separately.

## Train

On your Runpod Pod, place the project under `/workspace` and open a terminal in the project folder:

```bash
python3 -m pip install -r requirements.txt
python3 finetune.py --check
python3 finetune.py
```

`config.yaml` controls the model, dataset paths and hyperparameters. The default model is downloaded from Hugging Face into the Pod's local cache. CUDA is selected automatically when available. Training reads `data/train.csv`, prints progress, then evaluates a classifier on `data/validation.csv`. Label `1.0` is phishing. The validation set is not used to update model weights.

The final model and tokenizer are saved to `/workspace/fine-tuned-weights/<train.output_model_name>/`. Choose a new `train.output_model_name` if that folder already exists; training will not overwrite it.

## Validate and test

Classifier training evaluates the validation set after training and writes `evaluation.json` with accuracy, precision, recall, F1 and confusion counts. To run a separate validation or test inference:

```bash
python3 inference.py --dataset validation
python3 inference.py --dataset test
```

`test` is the default. The model comes from `inference.model_source` in `config.yaml`; use `local:Original` for `/workspace/fine-tuned-weights/Original`, or configure a different local model or Hugging Face repository. To use a different folder for one run:

```bash
python3 inference.py --dataset test --model-dir /workspace/fine-tuned-weights/Original
```

The validation and test files are a fixed-seed, class-stratified 50/50 split of `data/inference.csv`. Use validation for model and threshold choices and reserve test for the final check. The scripts feed the configured text column to the model, not the CSV label. The original label remains in the prediction CSV for scoring.

## Files produced

- **CSV data and predictions:** `data/`. Each inference run creates a uniquely named `predict-<timestamp>-predictions.csv` next to the selected input CSV and prints its path. For classifier models, it includes `predicted_label` and `phishing_probability`; generative models output `model_response`.
- **Training run:** `/workspace/phishing-detect-output/train-<timestamp>/` contains the run configuration, `run.log`, metadata, validation `evaluation.json` for classifiers, one `checkpoints/epoch-NNN.pt` per completed epoch, and `COMPLETE` after success.
- **Inference run:** `/workspace/phishing-detect-output/predict-<timestamp>/` contains the run configuration, log, metadata, and `COMPLETE` after success. `metrics.json` is also written if ground-truth labels and predictions can be compared.
- **Final model:** `/workspace/fine-tuned-weights/<name>/` contains model weights and tokenizer files for later inference.

Metrics use `data.positive_label`, currently `1.0`. Accuracy, precision, recall, and F1 are fractions from 0 to 1; an undefined metric is `null`. Free-form generative responses are scored only if they match the ground-truth labels exactly; ambiguous responses are not silently interpreted. To analyze a prediction CSV separately:

```bash
python3 analyze_results.py data/predict-<timestamp>-predictions.csv
```

That command writes `metrics.json` next to the specified CSV unless `--output` is supplied. It will not overwrite an existing metrics file.

## Resume an interrupted run

Use the original run's configuration and the desired checkpoint:

```bash
python3 finetune.py \\
  --config /workspace/phishing-detect-output/train-<timestamp>/config.yaml \\
  --resume /workspace/phishing-detect-output/train-<timestamp>/checkpoints/epoch-001.pt
```

The data, base model/tokenizer, device type, precision and training settings must match. If the final model folder already exists, change `train.output_model_name` in a copy of the config before resuming. Checkpoints include optimizer state and can be large; load only checkpoints you trust.

## Other settings and utilities

`train.task` and `inference.task` independently select `auto`, `classification`, `causal_lm`, or `seq2seq`. Generative models use `data.prompt_template`; inference generation settings are under `inference`. Not every model or modality works with the Transformers text workflow, and large models may exceed GPU memory. Use `trust_remote_code` only for repository code you trust.

`python3 clean_csv.py` keeps the six configured data columns in `data/train.csv` and `data/inference.csv`, retaining original backups. `split_csv.py` and `split_validation_test.py` are available for creating splits; they do not overwrite existing outputs. Avoid re-splitting the current files unless you intend to create a new dataset.

Stop the Runpod Pod when finished to end compute charges. The project includes a Dockerfile, but the image has not been tested on Runpod.
