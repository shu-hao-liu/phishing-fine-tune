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

The scripts always use these paths. The CSVs are excluded from Git and need to be uploaded separately. Label `1.0` means phishing.

Install dependencies once:

```bash
python3 -m pip install -r requirements.txt
```

## Train and validate

Set the model, output model name and training settings in the `train` section of `config.yaml`. Use a Hugging Face repository ID or URL, or `local:Model-1` for a model previously saved under that name. Choose a new `output_model_name` for each completed model; existing final weights are not overwritten. CUDA is detected automatically.

```bash
python3 finetune.py
```

Training reads `data/train.csv` and automatically validates the trained model with `data/validation.csv`. The validation labels do not enter the model input and validation does not update the weights. Classifier validation reports accuracy, precision, recall and F1. Generative validation saves the model's responses; metrics are available only when the responses match the labels unambiguously. The `validate.threshold` setting controls classifier validation.

To resume an interrupted run from a saved epoch, use the original settings and pass the checkpoint:

```bash
python3 finetune.py --resume /workspace/fine-tuned-weights/Original/checkpoints/train-<timestamp>/epoch-001.pt
```

You can instead set `train.resume_from` in the config. The training data, model, tokenizer, device type, precision and training settings must match the checkpoint. Load checkpoints only from trusted sources.

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
python3 test.py --checkpoint /workspace/fine-tuned-weights/Original/checkpoints/train-<timestamp>/epoch-001.pt
```

Testing always reads `data/test.csv`. The label is not included in model input; it is retained in the output for scoring. Classifiers produce predictions and metrics. Generative models produce responses and are scored only when their responses match the labels exactly.

## Files produced

- Model and tokenizer: `/workspace/fine-tuned-weights/<model>/final/`.
- Resumable epoch checkpoints: `/workspace/fine-tuned-weights/<model>/checkpoints/train-<timestamp>/epoch-NNN.pt`.
- Training logs, settings and validation results: `/workspace/phishing-detect-output/<model>/training/train-<timestamp>/`.
- Test predictions, metrics when available, logs and settings: `/workspace/phishing-detect-output/<model>/<checkpoint-or-final>/predict-<timestamp>/`.

The Hugging Face download cache is on the Pod's local disk. Each output run has its own folder. `train.task` and `test.task` support `auto`, `classification`, `causal_lm` and `seq2seq`; model architecture, available memory and Transformers support still determine which models can run.

## If torchvision fails to import

If you see `RuntimeError: operator torchvision::nms does not exist`, the Pod's torchvision installation may not match its PyTorch installation. This text-only project does not need torchvision. If no other software in that environment needs it, remove it and retry:

```bash
python3 -m pip uninstall -y torchvision
python3 finetune.py --check
python3 finetune.py
```

If you need torchvision for another project, use a separate environment or install a torchvision build compatible with your installed PyTorch and CUDA instead.
