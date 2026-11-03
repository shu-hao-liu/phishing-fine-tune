#!/usr/bin/env python3
"""Fine-tune and run a Hugging Face binary sequence classifier on CSV data."""
import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
from urllib.parse import urlparse

import torch
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger('phishing')


def repository_id(source):
    if not isinstance(source, str) or not source.strip():
        raise ValueError('model.source must be a Hugging Face URL or repository ID')
    source = source.strip()
    if '://' in source:
        u = urlparse(source)
        if u.scheme != 'https' or u.netloc.lower() not in ('huggingface.co', 'www.huggingface.co') or u.query or u.fragment:
            raise ValueError('model.source URL must be an HTTPS Hugging Face model page')
        source = u.path.strip('/')
    parts = source.split('/')
    if len(parts) not in (1, 2) or any(not p or p in ('.', '..') for p in parts):
        raise ValueError('Use a model repository ID or model page URL; put the revision in model.revision')
    return source


def read_config(path, train_csv_override=None):
    with open(path, encoding='utf-8') as f:
        c = yaml.safe_load(f)
    for section in ('model', 'data', 'training', 'inference'):
        if not isinstance(c.get(section), dict):
            raise ValueError(f'Missing {section} configuration')
    repository_id(c['model'].get('source'))
    for key in ('revision', 'cache_dir'):
        if not isinstance(c['model'].get(key), str) or not c['model'][key].strip():
            raise ValueError(f'model.{key} must be a nonempty string')
    if train_csv_override is not None:
        c['data']['train_csv'] = train_csv_override
    for key in ('train_csv', 'inference_csv', 'text_column', 'label_column', 'positive_label'):
        if not isinstance(c['data'].get(key), str) or not c['data'][key].strip():
            raise ValueError(f'data.{key} must be a nonempty string')
    for key in ('train_csv', 'inference_csv', 'eval_csv'):
        if c['data'].get(key):
            path = Path(c['data'][key]).expanduser()
            c['data'][key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    train = Path(c['data']['train_csv'])
    inference = Path(c['data']['inference_csv'])
    if train == inference or train.parent == inference.parent:
        raise ValueError('Training and inference CSVs must be in different directories')
    if c['data'].get('eval_csv') and Path(c['data']['eval_csv']).expanduser().resolve() in (train, inference):
        raise ValueError('Evaluation CSV must differ from training and inference CSVs')
    for sec, key in [('data', 'max_length'), ('training', 'epochs'), ('training', 'batch_size'), ('training', 'gradient_accumulation_steps'), ('inference', 'batch_size')]:
        v = c[sec].get(key)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f'{sec}.{key} must be a positive integer')
    for key in ('learning_rate', 'weight_decay', 'max_grad_norm'):
        v = c['training'].get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or (key != 'weight_decay' and v == 0):
            raise ValueError(f'training.{key} must be a finite valid number')
    seed = c['training'].get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('training.seed must be an integer in [0, 2^32)')
    if c['training'].get('precision') not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('training.precision must be fp32, fp16, or bf16')
    threshold = c['inference'].get('threshold')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('inference.threshold must be between 0 and 1')
    if not isinstance(c.get('output_dir'), str) or not c['output_dir'].strip():
        raise ValueError('output_dir must be a nonempty path')
    return c


def read_csv(path, text_column, label_column=None):
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f'{path}: missing or duplicate column names')
        required = [text_column] + ([label_column] if label_column else [])
        if any(col not in reader.fieldnames for col in required):
            raise ValueError(f'{path}: required columns: {required}')
        rows = []
        for line, row in enumerate(reader, start=2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'{path}: malformed row {line}')
            if not row[text_column].strip() or (label_column and not row[label_column].strip()):
                raise ValueError(f'{path}: missing text or label at row {line}')
            rows.append(row)
        if not rows:
            raise ValueError(f'{path}: no data rows')
        return reader.fieldnames, rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer, text_col, max_length, labels=None, label_col=None):
        self.rows, self.tokenizer, self.text_col = rows, tokenizer, text_col
        self.max_length, self.labels, self.label_col = max_length, labels, label_col

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        item = dict(self.tokenizer(row[self.text_col], truncation=True, max_length=self.max_length))
        if self.labels is not None:
            item['labels'] = self.labels[row[self.label_col].strip()]
        return item


def setup_run(config, operation):
    out = Path(config['output_dir']).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    run = out / (operation + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir()
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run / 'run.log', encoding='utf-8')):
        handler.setFormatter(formatter)
        LOG.addHandler(handler)
    (run / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    LOG.info('Outputs: %s', run)
    return run


def configure_device(c):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    precision = c['training']['precision']
    if precision != 'fp32' and device.type != 'cuda':
        raise ValueError('fp16/bf16 requires a CUDA GPU')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('The selected CUDA GPU does not support bf16')
    return device


def autocast(device, precision):
    if precision == 'fp32':
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=torch.float16 if precision == 'fp16' else torch.bfloat16)


def common_metadata(c, device):
    import transformers
    return {'model_source': c['model']['source'], 'requested_revision': c['model']['revision'], 'torch': torch.__version__, 'transformers': transformers.__version__, 'python': sys.version, 'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None, 'seed': c['training']['seed']}


def make_loader(rows, tokenizer, c, labels=None, shuffle=False, batch_size=None):
    from transformers import DataCollatorWithPadding
    data = c['data']
    ds = TextDataset(rows, tokenizer, data['text_column'], data['max_length'], labels, data['label_column'])
    return DataLoader(ds, batch_size=batch_size or c['training']['batch_size'], shuffle=shuffle, collate_fn=DataCollatorWithPadding(tokenizer))


def train(c, out, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    data, settings = c['data'], c['training']
    _, rows = read_csv(data['train_csv'], data['text_column'], data['label_column'])
    classes = sorted({r[data['label_column']].strip() for r in rows})
    if len(classes) != 2 or data['positive_label'] not in classes:
        raise ValueError(f'Training labels must contain exactly two classes including positive_label; found {classes}')
    negative = next(x for x in classes if x != data['positive_label'])
    labels = {negative: 0, data['positive_label']: 1}
    evaluation = None
    if data.get('eval_csv'):
        _, evaluation = read_csv(data['eval_csv'], data['text_column'], data['label_column'])
        unknown = {r[data['label_column']].strip() for r in evaluation} - set(labels)
        if unknown:
            raise ValueError(f'Evaluation CSV has labels absent from training: {sorted(unknown)}')
    seed = settings['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    repo = repository_id(c['model']['source'])
    opts = {'revision': c['model']['revision'], 'cache_dir': c['model']['cache_dir'], 'trust_remote_code': False}
    LOG.info('Loading %s at %s (cache %s)', repo, opts['revision'], opts['cache_dir'])
    tokenizer = AutoTokenizer.from_pretrained(repo, **opts)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError('Model tokenizer has no padding or EOS token; choose a compatible sequence-classification model')
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(repo, num_labels=2, id2label={0: negative, 1: data['positive_label']}, label2id=labels, **opts)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    loader = make_loader(rows, tokenizer, c, labels, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    accum = settings['gradient_accumulation_steps']
    steps = math.ceil(len(loader) / accum) * settings['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=steps)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and settings['precision'] == 'fp16'))
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(settings['epochs']):
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            group_size = min(accum, len(loader) - (i // accum) * accum)
            with autocast(device, settings['precision']):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss; no final model will be saved')
            total_loss += loss.item()
            scaler.scale(loss / group_size).backward()
            if (i + 1) % accum == 0 or i + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        LOG.info('Epoch %d/%d mean batch loss %.6f', epoch + 1, settings['epochs'], total_loss / len(loader))
    if evaluation:
        correct = total = 0
        model.eval()
        with torch.inference_mode():
            for batch in make_loader(evaluation, tokenizer, c, labels):
                batch = {k: v.to(device) for k, v in batch.items()}
                target = batch.pop('labels')
                with autocast(device, settings['precision']): logits = model(**batch).logits
                correct += (logits.argmax(dim=-1) == target).sum().item()
                total += len(target)
        LOG.info('Evaluation accuracy: %.4f (%d/%d)', correct / total, correct, total)
        (out / 'evaluation.json').write_text(json.dumps({'accuracy': correct / total, 'correct': correct, 'count': total}, indent=2))
    model_dir = out / 'model'
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    metadata = common_metadata(c, device)
    metadata.update({'resolved_model_revision': getattr(model.config, '_commit_hash', None), 'resolved_tokenizer_revision': tokenizer.init_kwargs.get('_commit_hash'), 'labels': labels, 'training_csv_sha256': sha256(data['train_csv']), 'eval_csv_sha256': sha256(data['eval_csv']) if data.get('eval_csv') else None})
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    LOG.info('Saved trained model to %s', model_dir)


def predict(c, out, device, model_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    data = c['data']
    fields, rows = read_csv(data['inference_csv'], data['text_column'])
    extra = ['predicted_label', 'phishing_probability']
    if any(x in fields for x in extra):
        raise ValueError(f'Inference CSV already contains an output column: {extra}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False).to(device)
    labels = {int(k): v for k, v in model.config.id2label.items()}
    if set(labels) != {0, 1} or labels[1] != data['positive_label']:
        raise ValueError('Model label mapping must have positive_label at index 1; use the model saved by this project')
    loader = make_loader(rows, tokenizer, c, batch_size=c['inference']['batch_size'])
    model.eval()
    output = out / 'predictions.csv'
    offset = 0
    with output.open('w', newline='', encoding='utf-8') as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=fields + extra)
        writer.writeheader()
        for batch in loader:
            with autocast(device, c['training']['precision']):
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
            for prob in probs:
                writer.writerow({**rows[offset], 'predicted_label': labels[1 if prob >= c['inference']['threshold'] else 0], 'phishing_probability': prob})
                offset += 1
    meta = common_metadata(c, device)
    meta.update({'trained_model_dir': str(Path(model_dir).resolve()), 'inference_csv_sha256': sha256(data['inference_csv']), 'count': offset, 'threshold': c['inference']['threshold']})
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    LOG.info('Saved %d predictions to %s', offset, output)


def main(forced_action=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    if forced_action is None:
        sub = parser.add_subparsers(dest='action', required=True)
        sub.add_parser('check', help='validate configuration and inspect local hardware')
        sub.add_parser('train', help='fine-tune the configured model on the training CSV')
        prediction = sub.add_parser('predict', help='classify the separate inference CSV')
        prediction.add_argument('--model-dir', help='override inference.model_dir in config')
    elif forced_action == 'train':
        parser.add_argument('--train-csv', help='override data.train_csv in the shared config')
        parser.add_argument('--check', action='store_true', help='validate configuration and inspect local hardware')
    elif forced_action == 'predict':
        parser.add_argument('--model-dir', help='override inference.model_dir in config')
    args = parser.parse_args()
    if forced_action is not None:
        args.action = 'check' if forced_action == 'train' and args.check else forced_action
    try:
        c = read_config(args.config, getattr(args, 'train_csv', None))
        device = configure_device(c)
        if args.action == 'check':
            print(f'Configuration valid. Device: {device}; model: {repository_id(c["model"]["source"])}; cache: {c["model"]["cache_dir"]}')
            return 0
        if args.action == 'predict':
            args.model_dir = args.model_dir or c['inference'].get('model_dir')
            if not isinstance(args.model_dir, str) or not args.model_dir.strip():
                raise ValueError('Set inference.model_dir in config.yaml or pass --model-dir')
        out = setup_run(c, args.action)
        try:
            LOG.info('Device: %s', device)
            if args.action == 'train': train(c, out, device)
            else: predict(c, out, device, args.model_dir)
            (out / 'COMPLETE').write_text('Completed successfully.\n')
        except Exception:
            LOG.exception('Run failed')
            return 1
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main('train'))
