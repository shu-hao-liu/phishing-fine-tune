# Phishing model training

Place the project on your Runpod Pod and upload the three datasets into its `data` folder:

```text
phishing-detect/
├── config.yaml
├── finetune.py
├── test.py
└── data/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

These are the default dataset paths. The CSVs are excluded from Git and need to be uploaded separately. Label `1.0` means phishing.

Install dependencies once:

```bash
python3 -m pip install -r requirements.txt
```

## Choose a dataset

Set `train.dataset` in `config.yaml` to `original` or `fcn`. The selection applies to training, automatic validation, and the default test run:

| Selection | Training | Validation | Test |
| --- | --- | --- | --- |
| `original` | `data/train.csv` | `data/validation.csv` | `data/test.csv` |
| `fcn` | `data/FCN_train.csv` | `data/FCN_validation.csv` | `data/FCN_test.csv` |

The FCN dataset is split 80/10/10 by label using seed 42. The original formatted FCN file remains intact. CSV files are excluded from Git; upload the three FCN files to the Pod separately. You can still use `test.py --csv` to test another file for a single run.

## Train and validate

Set the model, output model name and training settings in the `train` section of `config.yaml`. Use a Hugging Face repository ID or URL, or `local:Model-1` for a model previously saved under that name. Choose a new `output_model_name` for each completed model; existing final weights are not overwritten. CUDA is detected automatically.

```bash
python3 finetune.py
```

Training and automatic validation use the CSVs for the selected dataset. The validation labels do not enter the model input and validation does not update the weights. Classifier validation reports accuracy, precision, recall and F1. Generative validation saves the model's responses; metrics are available only when the responses match the labels unambiguously. The `validate.threshold` setting controls classifier validation.

To resume an interrupted run from a saved epoch, use the original settings and pass the checkpoint:

```bash
python3 finetune.py --resume /workspace/Original/weights/checkpoints/train-<timestamp>/epoch-001.pt
```

You can instead set `train.resume_from` in the config. The training data, model, tokenizer, device type, precision and training settings must match the checkpoint. Load checkpoints only from trusted sources.

## LoRA fine-tuning

To train LoRA adapters instead of updating the full base model, set `train.lora.enabled: true` in `config.yaml`. The other LoRA settings control rank, alpha, dropout, and target module names. The default `[query, value]` targets XLM-RoBERTa; other architectures may need different names, or `target_modules: null` where PEFT supports automatic selection. For classifiers, the classification head is also trained.

Epoch checkpoints contain adapter and optimizer state for resuming. The final folder contains a full model with the adapter merged into the base weights, so `test.py` can use it normally. Testing or resuming an epoch checkpoint also requires the original base model and tokenizer in the local cache or available from Hugging Face. Install the updated requirements before using LoRA.

## Test

The `test` section chooses the model and test settings independently. By default, `local:Original` uses the final trained model. You can also select a Hugging Face model, or set `test.checkpoint` to an epoch checkpoint. A checkpoint takes precedence over the model setting. The base model and tokenizer must be available in the local cache or on Hugging Face when loading a checkpoint.

```bash
python3 test.py
```

To run `python3 test.py` directly from `/workspace` when the project is in a subfolder, create a link once from inside the project folder:

```bash
ln -s "$(pwd)/test.py" /workspace/test.py
cd /workspace
python3 test.py
```

The link uses the project folder's `config.yaml` and `data/test.csv`. If the project files are already directly in `/workspace`, no link is needed. If `/workspace/test.py` already exists, inspect it before replacing it.

To select a checkpoint for one run:

```bash
python3 test.py --checkpoint /workspace/Original/weights/checkpoints/train-<timestamp>/epoch-001.pt
```

By default, testing reads the test CSV for the selected dataset. To test another CSV, pass its path (relative paths are relative to the current working directory):

```bash
python3 test.py --csv data/another-dataset.csv
```

You can combine `--csv` with `--checkpoint`. The CSV must contain the configured text column (`body` by default). Include the configured label column (`label` by default) with labels matching the model and positive label to calculate metrics. The label is not included in model input; it is retained in the output for scoring. Classifiers produce predictions and metrics. Generative models produce responses and are scored only when their responses match the labels exactly.

## Files produced

- Model and tokenizer: `/workspace/<model>/weights/final/`.
- Resumable epoch checkpoints: `/workspace/<model>/weights/checkpoints/train-<timestamp>/epoch-NNN.pt`.
- Training logs, settings and validation results: `/workspace/<model>/outputs/train-<timestamp>/`.
- Test predictions, metrics when available, logs and settings: `/workspace/<model>/outputs/test-<checkpoint-or-final>-<timestamp>/`.

Each model has one folder containing weights and outputs. Each training or test run has its own output folder; test folder names identify the checkpoint or final model. Existing files are not moved. Models in the previous `/workspace/fine-tuned-weights/<model>/` layout remain readable with `local:<model>`, and old checkpoint paths remain usable.

The Hugging Face download cache is on the Pod's local disk. Each output run has its own folder. `train.task` and `test.task` support `auto`, `classification`, `causal_lm` and `seq2seq`; model architecture, available memory and Transformers support still determine which models can run.

## If torchvision fails to import

If you see `RuntimeError: operator torchvision::nms does not exist`, the Pod's torchvision installation may not match its PyTorch installation. This text-only project does not need torchvision. If no other software in that environment needs it, remove it and retry:

```bash
python3 -m pip uninstall -y torchvision
python3 finetune.py --check
python3 finetune.py
```

If you need torchvision for another project, use a separate environment or install a torchvision build compatible with your installed PyTorch and CUDA instead.

## Format another CSV

Use `format_csv.py` to prepare one CSV at a time. It creates a new file and will not overwrite an existing file or split the data. For example:

```bash
python3 format_csv.py data/raw.csv data/formatted.csv --mode train \
  --text-column message --label-column category --positive phishing --negative legitimate
```

`--mode` is `train`, `validate`, or `test`. The example maps the source columns and labels to the configured text and label columns, with `1.0` for phishing and `0.0` for legitimate by default. Omit the column and label mapping options when the source already uses the configured column names and labels. To format a test dataset with no label column, use `--mode test --unlabeled`; it produces predictions without metrics.

The output retains the subject, text, URLs, label, language and source columns when available; missing optional columns are empty and other columns are omitted. All rows are preserved in order. The formatter reports invalid or missing text and labels instead of silently dropping records. Training requires both classes. Upload or move the formatted file to the appropriate dataset name if you want to use it for training or validation. For testing, pass its path with `python3 test.py --csv data/formatted.csv`.
