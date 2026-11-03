# Phishing detection: fine-tuning and inference

Fine-tune a Hugging Face **sequence classification** model on a binary labeled CSV, then classify a separate CSV. Set a repository ID or model-page link in `config.yaml`. This is full-parameter supervised fine-tuning, not a generative chat-model instruction tuning pipeline. The model architecture must be supported by `AutoModelForSequenceClassification`; gated models require Hugging Face access and authentication. Large models may exceed Pod memory. The output probability is a model score, not a guarantee that an item is safe.

## Run on a Runpod GPU Pod

1. Start a GPU Pod with a Python environment, CUDA-compatible PyTorch support, and writable `/workspace` storage. A PyTorch Pod template with a terminal is convenient. Transfer this project folder to the Pod. Pick sufficient GPU memory for the model, optimizer and batch size; start with a small model and reduce batch size if necessary. Provision adequate **container disk** for PyTorch and downloaded weights, plus enough `/workspace` storage for datasets and full model checkpoints. Disk and VRAM needs vary by model and dataset and have not been measured here.
2. From a terminal in the transferred project folder:

   ```bash
   python3 -m pip install -r requirements.txt
   python3 finetune.py --check
   ```

3. Upload the labeled training CSV to the project folder as `training.csv`. By default, the separate inference CSV is `/workspace/phishing-data/inference/inference.csv`. The files must exist; the program never creates or invents examples. Optional evaluation data may be specified separately. Training and inference CSVs must be in different directories.
4. Edit `config.yaml`, in particular `model.source`, `data.*` column names and paths, and `data.positive_label`. Use a model commit hash as `model.revision` for reproducibility. The default `main` is a moving revision; the resolved commit is recorded when available. Model weights and tokenizer files download into `model.cache_dir` (default `/tmp/phishing-detect-hf-cache`) on the Pod's local container storage and are reused while that cache remains available. The default cache is not persistent across Pod replacement. Set `HF_TOKEN` in the Pod environment for a gated model, and accept its license on Hugging Face first; never put tokens in `config.yaml`.
5. Train and then use the **model directory printed by training**:

   ```bash
   python3 finetune.py
   python3 inference.py --model-dir /workspace/phishing-detect-output/train-<timestamp>/model
   ```

   Replace `<timestamp>` with the actual directory. Both scripts read the same `config.yaml`. To use a differently named CSV in the project folder, set `data.train_csv` in the config or run `python3 finetune.py --train-csv your-file.csv`. Relative dataset paths resolve from the project folder, regardless of the current working directory. Set `inference.model_dir` to the saved model directory to run `python3 inference.py` without an argument. Pass `--config /path/to/config.yaml` to either script to select another shared configuration; `--model-dir` overrides the configured model directory. `finetune.py --check` validates settings and reports the detected device without downloading a model or checking dataset contents. Training reads only the training CSV and optional evaluation CSV; prediction reads the inference CSV and saved model, without downloading the base model.
6. Retrieve outputs from `/workspace/phishing-detect-output` and stop the Pod when finished. A running GPU Pod continues to incur charges. Mount persistent storage at `/workspace` if results must survive Pod replacement; the path alone does not guarantee persistence.

An optional `Dockerfile` builds a CUDA/Python image for a Pod. It keeps the container alive without starting an experiment or exposing a terminal service. You must have an appropriate way to execute commands inside the container and supply `/workspace` storage. The image includes the default config; mount or copy a custom config and use `--config` to override it. The image has not been built or tested on Runpod.

## CSV format

Training requires a text column and exactly two nonempty label values, including `data.positive_label`. For example, with the default column names:

```csv
text,label
"Please verify your account via this link",phishing
"The meeting is at 3 PM",legitimate
```

Inference requires the text column; other columns are preserved in the predictions file:

```csv
text,id
"Your account needs verification",123
"See you at the meeting",124
```

Set `eval_csv` to a different labeled file to report accuracy. It is never used for training. Keep evaluation data representative and independent when assessing real-world performance. Accuracy is a basic check and alone is not sufficient to validate phishing detection.

## Settings and outputs

`config.yaml` controls epochs, learning rate, batch size, gradient accumulation, weight decay, gradient clipping, seed, maximum token length, and precision (`fp32`, `fp16`, or `bf16`; mixed precision needs a compatible CUDA GPU). Long text is truncated to `max_length`. Inference controls batch size and the decision threshold for the positive label. The same precision setting applies to inference. CPU fallback is available with `fp32` but can be slow. Runs do not resume interrupted training.

Each training run gets a timestamped directory with `model/` (the full trained model and tokenizer), `config.yaml`, `run.log`, `metadata.json`, optional `evaluation.json`, and a `COMPLETE` marker on success. Prediction produces a separate timestamped directory with `predictions.csv`, config, log, metadata and completion marker. Prediction rows preserve the input fields and add `predicted_label` and `phishing_probability`. Data hashes, model revision when available, device and library versions are recorded. Failed runs do not get a completion marker and may contain partial outputs. Do not use input column names `predicted_label` or `phishing_probability`.

Run local offline tests with `python3 -m unittest -v`. They train and predict with a tiny randomly initialized synthetic model; they do not validate model quality, real Hugging Face downloads, GPU operation, or Runpod deployment.

## Automatic GitHub commits

Run `python3 auto_commit.py` in the project folder to check for changes every two seconds, commit unignored files, and push to the configured upstream branch. Keep the process running; stop it with Ctrl-C. Use `--once` for one check, `--no-push` for local commits only, or `--interval 10` to change the interval. It refuses to include already-staged changes and never force-pushes. Git credentials and network access must be available to the running process. If a push fails, it reports the error and retries on a later check. Only one updated auto-committer can run per checkout; a second instance exits with a clear message. Other Git commands can still briefly lock the index, so transient lock errors are retried. If an index lock persists after all Git processes have stopped, inspect the repository before removing it.

CSV and common dataset, checkpoint, model-weight, and secret-file extensions are ignored by default. Review `.gitignore` before using automatic pushes: other unignored files will be committed and sent to GitHub.

## Split the final 20% of a CSV

```bash
python3 split_csv.py
```

This removes the last 20% of data records from `train.csv` in the project folder and writes them to a new `inference.csv` in that folder. Both files keep the original header and record text; no rows are shuffled or modified. If 20% is not a whole number of records, the holdout count is rounded up. The tool does not overwrite an existing `inference.csv`.

To use other paths, run `python3 split_csv.py --train /path/to/train.csv --inference /path/to/inference.csv`. Update `data.train_csv` and `data.inference_csv` in `config.yaml` to those paths; the training and inference CSVs must be in different directories for the experiment. The splitter does not modify the configuration.
