#!/usr/bin/env python3
"""Fine-tune and run Hugging Face text models on CSV data."""
import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
from urllib.parse import urlparse

import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from analyze_results import analyze_predictions, calculate_metrics

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path('/workspace')
WEIGHTS_ROOT = Path('/workspace/fine-tuned-weights')  # Previous layout
OUTPUT_ROOT = Path('/workspace/phishing-detect-output')
CACHE_ROOT = '/tmp/phishing-detect-hf-cache'
LOG = logging.getLogger('phishing')


def repository_id(source):
    if not isinstance(source, str) or not source.strip():
        raise ValueError('Model source must be a Hugging Face URL or repository ID')
    source = source.strip()
    if '://' in source:
        u = urlparse(source)
        if u.scheme != 'https' or u.netloc.lower() not in ('huggingface.co', 'www.huggingface.co') or u.query or u.fragment:
            raise ValueError('Model source URL must be an HTTPS Hugging Face model page')
        source = u.path.strip('/')
    parts = source.split('/')
    if len(parts) not in (1, 2) or any(not p or p in ('.', '..') for p in parts):
        raise ValueError('Use a model repository ID or model page URL; set the model revision separately')
    return source


def validate_model_name(name):
    if not isinstance(name, str) or not name or name in ('.', '..') or name != name.strip() or '/' in name or '\\' in name:
        raise ValueError('train.output_model_name must be a simple folder name')
    return name


def named_weights_path(name):
    name = validate_model_name(name)
    new = MODEL_ROOT / name / 'weights'
    old = WEIGHTS_ROOT / name
    if (new / 'final').is_dir() or (new / 'config.json').is_file():
        return new
    if (old / 'final').is_dir() or (old / 'config.json').is_file():
        return old
    return new


def resolve_model_source(source, revision, cache_dir, require_exists=True):
    """Use a local directory or a Hub repository without silently substituting."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError('Model source must be a local directory or Hugging Face repository')
    source = source.strip()
    if source.startswith('local:'):
        name = validate_model_name(source[len('local:'):])
        path = named_weights_path(name)
        is_local = True
    elif source.startswith('fine-tuned-weights/'):
        name = validate_model_name(source[len('fine-tuned-weights/'):])
        path = named_weights_path(name)
        is_local = True
    else:
        path = Path(source).expanduser()
        named_path = named_weights_path(source) if not source.startswith(('http://', 'https://')) and '/' not in source and source not in ('.', '..') else None
        if named_path is not None and named_path.is_dir():
            path = named_path
        is_local = path.is_dir() or source.startswith(('/', './', '../', '~'))
    if is_local:
        if (path / 'final').is_dir():
            path = path / 'final'
        if require_exists and not path.is_dir():
            raise FileNotFoundError(f'Model directory not found: {path}')
        return str(path.resolve()), {'local_files_only': True, 'trust_remote_code': False}
    return repository_id(source), {'revision': revision, 'cache_dir': cache_dir, 'trust_remote_code': False}


def read_config(path, train_csv_override=None):
    with open(path, encoding='utf-8') as f:
        c = yaml.safe_load(f)
    if isinstance(c, dict) and set(c) == {'train', 'validate', 'test'}:
        train, validate, test = c['train'], c['validate'], c['test']
        if not all(isinstance(section, dict) for section in (train, validate, test)):
            raise ValueError('train, validate, and test must be configuration sections')
        c = {'train': dict(train), 'data': {
            'train_csv': str(ROOT / 'data/train.csv'),
            'eval_csv': str(ROOT / 'data/validation.csv'),
            'inference_csv': str(ROOT / 'data/test.csv'),
            'text_column': train.get('text_column', 'body'),
            'label_column': train.get('label_column', 'label'),
            'positive_label': train.get('positive_label', '1.0'),
            'max_length': train.get('max_length', 512),
            'prompt_template': 'Classify the following text as phishing or legitimate.\nText: {text}\nLabel:',
        }, 'inference': dict(test)}
        c['train'].update(cache_dir=CACHE_ROOT, output_dir=str(OUTPUT_ROOT))
        c['inference'].update(cache_dir=CACHE_ROOT, output_dir=str(OUTPUT_ROOT))
        c['inference'].setdefault('threshold', validate.get('threshold', 0.5))
        c['validation_threshold'] = validate.get('threshold', 0.5)
    for section in ('train', 'data', 'inference'):
        if not isinstance(c.get(section), dict):
            raise ValueError(f'Missing {section} configuration')
    resolve_model_source(c['train'].get('model_source'), c['train'].get('model_revision'), c['train'].get('cache_dir'), require_exists=False)
    for section in ('train', 'inference'):
        for key in ('cache_dir', 'output_dir'):
            if not isinstance(c[section].get(key), str) or not c[section][key].strip():
                raise ValueError(f'{section}.{key} must be a nonempty string')
    if not isinstance(c['train'].get('model_revision'), str) or not c['train']['model_revision'].strip():
        raise ValueError('train.model_revision must be a nonempty string')
    for section in ('train', 'inference'):
        if not isinstance(c[section].get('trust_remote_code', False), bool):
            raise ValueError(f'{section}.trust_remote_code must be boolean')
        if c[section].get('task', 'auto') not in ('auto', 'classification', 'causal_lm', 'seq2seq'):
            raise ValueError(f'{section}.task must be auto, classification, causal_lm, or seq2seq')
    template = c['data'].get('prompt_template', 'Classify the following text as phishing or legitimate.\nText: {text}\nLabel:')
    try:
        if not isinstance(template, str) or '{text}' not in template:
            raise ValueError('data.prompt_template must include {text}')
        template.format(text='sample')
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError('data.prompt_template must contain only a valid {text} placeholder') from exc
    if train_csv_override is not None:
        c['data']['train_csv'] = train_csv_override
    for key in ('train_csv', 'inference_csv', 'text_column', 'label_column', 'positive_label'):
        if not isinstance(c['data'].get(key), str) or not c['data'][key].strip():
            raise ValueError(f'data.{key} must be a nonempty string')
    if c['data']['text_column'] == c['data']['label_column']:
        raise ValueError('data.text_column must differ from data.label_column to prevent label leakage')
    for key in ('train_csv', 'inference_csv', 'eval_csv'):
        if c['data'].get(key):
            path = Path(c['data'][key]).expanduser()
            c['data'][key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    train = Path(c['data']['train_csv'])
    inference = Path(c['data']['inference_csv'])
    if train == inference:
        raise ValueError('Training and inference CSVs must be different files')
    if c['data'].get('eval_csv') and Path(c['data']['eval_csv']).expanduser().resolve() in (train, inference):
        raise ValueError('Evaluation CSV must differ from training and inference CSVs')
    for sec, key in [('data', 'max_length'), ('train', 'epochs'), ('train', 'batch_size'), ('train', 'gradient_accumulation_steps'), ('inference', 'batch_size')]:
        v = c[sec].get(key)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f'{sec}.{key} must be a positive integer')
    for key in ('learning_rate', 'weight_decay', 'max_grad_norm'):
        v = c['train'].get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or (key != 'weight_decay' and v == 0):
            raise ValueError(f'train.{key} must be a finite valid number')
    validate_model_name(c['train'].get('output_model_name'))
    seed = c['train'].get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('train.seed must be an integer in [0, 2^32)')
    if c['train'].get('precision') not in ('auto', 'fp32', 'fp16', 'bf16'):
        raise ValueError('train.precision must be auto, fp32, fp16, or bf16')
    inference_model = c['inference'].get('model_source')
    if inference_model is not None and (not isinstance(inference_model, str) or not inference_model.strip()):
        raise ValueError('inference.model_source must be null or a model repository, URL, or local directory')
    revision = c['inference'].get('model_revision', 'main')
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError('inference.model_revision must be a nonempty string')
    positive_index = c['inference'].get('positive_label_index')
    if positive_index is not None and (isinstance(positive_index, bool) or positive_index not in (0, 1)):
        raise ValueError('inference.positive_label_index must be null, 0, or 1')
    validation_threshold = c.get('validation_threshold', c['inference'].get('threshold'))
    if isinstance(validation_threshold, bool) or not isinstance(validation_threshold, (int, float)) or not math.isfinite(validation_threshold) or not 0 <= validation_threshold <= 1:
        raise ValueError('validate.threshold must be between 0 and 1')
    threshold = c['inference'].get('threshold')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('inference.threshold must be between 0 and 1')
    inf = c['inference']
    if not isinstance(inf.get('do_sample', False), bool):
        raise ValueError('inference.do_sample must be boolean')
    if isinstance(inf.get('max_new_tokens', 32), bool) or not isinstance(inf.get('max_new_tokens', 32), int) or inf.get('max_new_tokens', 32) < 1:
        raise ValueError('inference.max_new_tokens must be a positive integer')
    for key, upper in [('temperature', None), ('top_p', 1)]:
        value = inf.get(key, 0.7 if key == 'temperature' else 0.9)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 or (upper is not None and value > upper):
            raise ValueError(f'inference.{key} must be positive' + (' and at most 1' if upper else ''))
    resume = c['train'].get('resume_from')
    if resume is not None and (not isinstance(resume, str) or not resume.strip()):
        raise ValueError('train.resume_from must be null or a checkpoint path')
    checkpoint = c['inference'].get('checkpoint')
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise ValueError('inference.checkpoint must be null or a checkpoint path')
    progress = c['train'].get('progress_every', 1)
    if isinstance(progress, bool) or not isinstance(progress, int) or progress < 1:
        raise ValueError('train.progress_every must be a positive integer')
    return c


def read_csv(path, text_column, label_column=None):
    csv.field_size_limit(sys.maxsize)
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


def safe_path_label(value):
    original = str(value)
    label = re.sub(r'[^A-Za-z0-9._-]+', '-', original).strip('.-') or 'model'
    if label != original or len(label) > 100:
        label = label[:90] + '-' + hashlib.sha256(original.encode()).hexdigest()[:8]
    return label


def inference_labels(config, model_source, checkpoint=None):
    if checkpoint:
        cp = Path(checkpoint).expanduser()
        if cp.parent.parent.name == 'checkpoints':
            model_dir = cp.parent.parent.parent
            model = model_dir.parent.name if model_dir.name == 'weights' else model_dir.name
            return safe_path_label(model), safe_path_label(cp.stem)
    if model_source.startswith('local:'):
        model = model_source[len('local:'):]
    elif model_source.startswith('fine-tuned-weights/'):
        model = model_source[len('fine-tuned-weights/'):]
    elif '://' in model_source:
        model = repository_id(model_source).replace('/', '--')
    else:
        path = Path(model_source).expanduser()
        if path.is_dir():
            model_path = path.parent if path.name == 'final' else path
            model = model_path.parent.name if model_path.name == 'weights' else model_path.name
        else:
            model = model_source.replace('/', '--')
    if checkpoint:
        version = Path(checkpoint).stem
    elif model_source.startswith(('local:', 'fine-tuned-weights/', '/', './', '../', '~')) or Path(model_source).expanduser().is_dir():
        version = 'final'
    else:
        version = 'revision-' + config['inference'].get('model_revision', 'main')
    return safe_path_label(model), safe_path_label(version)


def setup_run(config, operation, model_source=None, checkpoint=None):
    out = Path(config['train' if operation == 'train' else 'inference']['output_dir']).expanduser()
    if operation == 'train':
        out = MODEL_ROOT / validate_model_name(config['train']['output_model_name']) / 'outputs'
        prefix = 'train'
    elif operation == 'predict':
        model_label, checkpoint_label = inference_labels(config, model_source, checkpoint)
        out = MODEL_ROOT / model_label / 'outputs'
        prefix = 'test-' + checkpoint_label
    else:
        prefix = operation
    out.mkdir(parents=True, exist_ok=True)
    run = out / (prefix + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir()
    LOG.setLevel(logging.INFO)
    for handler in LOG.handlers[:]:
        LOG.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run / 'run.log', encoding='utf-8')):
        handler.setFormatter(formatter)
        LOG.addHandler(handler)
    (run / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    LOG.info('Outputs: %s', run)
    return run


def check_optional_vision_dependency():
    """Catch a broken optional torchvision installation before loading a model."""
    if importlib.util.find_spec('torchvision') is None:
        return
    try:
        importlib.import_module('torchvision')
    except Exception as exc:
        raise RuntimeError(
            'The installed torchvision cannot be imported and may be incompatible with PyTorch. '
            'This text-only project does not require torchvision. If it is not needed by other '
            'software in this environment, remove it with: python3 -m pip uninstall -y torchvision. '
            f'Original error: {type(exc).__name__}: {exc}'
        ) from exc


def configure_device(c):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    precision = c['train']['precision']
    if precision in ('fp16', 'bf16') and device.type != 'cuda':
        raise ValueError('fp16/bf16 requires a CUDA GPU')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('The selected CUDA GPU does not support bf16')
    return device


def effective_precision(c, device):
    precision = c['train']['precision']
    if precision == 'auto':
        if device.type != 'cuda':
            return 'fp32'
        return 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    return precision


def autocast(device, precision):
    if precision == 'fp32':
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=torch.float16 if precision == 'fp16' else torch.bfloat16)


def common_metadata(c, device):
    import transformers
    return {'training_base_model_source': c['train']['model_source'], 'training_base_requested_revision': c['train']['model_revision'], 'torch': torch.__version__, 'transformers': transformers.__version__, 'python': sys.version, 'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None, 'seed': c['train']['seed'], 'effective_precision': effective_precision(c, device)}


def make_loader(rows, tokenizer, c, labels=None, shuffle=False, batch_size=None):
    from transformers import DataCollatorWithPadding
    data = c['data']
    ds = TextDataset(rows, tokenizer, data['text_column'], data['max_length'], labels, data['label_column'])
    return DataLoader(ds, batch_size=batch_size or c['train']['batch_size'], shuffle=shuffle, collate_fn=DataCollatorWithPadding(tokenizer))


def select_task(reference, opts, requested):
    if requested != 'auto':
        return requested
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(reference, **opts)
    saved = getattr(config, 'phishing_task', None)
    if saved in ('classification', 'causal_lm', 'seq2seq'):
        return saved
    if config.is_encoder_decoder:
        return 'seq2seq'
    architectures = ' '.join(config.architectures or []).lower()
    if any(name in architectures for name in ('causallm', 'lmheadmodel', 'conditionalgeneration')):
        return 'causal_lm'
    return 'classification'


def prompt_for(c, text):
    return c['data'].get('prompt_template', 'Classify the following text as phishing or legitimate.\nText: {text}\nLabel:').format(text=text)


class GenerativeDataset(Dataset):
    def __init__(self, rows, tokenizer, c, task):
        self.rows, self.tokenizer, self.c, self.task = rows, tokenizer, c, task

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        data, tok = self.c['data'], self.tokenizer
        prompt = prompt_for(self.c, row[data['text_column']])
        target = row[data['label_column']].strip()
        limit = data['max_length']
        if self.task == 'seq2seq':
            inputs = dict(tok(prompt, truncation=True, max_length=limit))
            inputs['labels'] = tok(text_target=target, truncation=True, max_length=limit)['input_ids']
            return inputs
        target_ids = tok(target, add_special_tokens=False)['input_ids']
        if tok.eos_token_id is not None:
            target_ids += [tok.eos_token_id]
        if not target_ids or len(target_ids) >= limit:
            raise ValueError('Label is empty or too long for data.max_length')
        prompt_ids = tok(prompt, add_special_tokens=True, truncation=True, max_length=limit - len(target_ids))['input_ids']
        return {'input_ids': prompt_ids + target_ids, 'attention_mask': [1] * (len(prompt_ids) + len(target_ids)), 'labels': [-100] * len(prompt_ids) + target_ids}


def make_generative_loader(rows, tokenizer, c, task, model, shuffle=False):
    from transformers import DataCollatorForSeq2Seq
    ds = GenerativeDataset(rows, tokenizer, c, task)
    if task == 'seq2seq':
        collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    else:
        def collator(items):
            labels = [item.pop('labels') for item in items]
            batch = tokenizer.pad(items, padding=True, return_tensors='pt')
            width = batch['input_ids'].shape[1]
            batch['labels'] = torch.tensor([v + [-100] * (width - len(v)) for v in labels], dtype=torch.long)
            return batch
    return DataLoader(ds, batch_size=c['train']['batch_size'], shuffle=shuffle, collate_fn=collator)


def model_tokenizer_fingerprints(model, tokenizer):
    tokenizer_info = {'vocab': tokenizer.get_vocab(), 'special_tokens': tokenizer.special_tokens_map}
    if getattr(tokenizer, 'backend_tokenizer', None) is not None:
        tokenizer_info['backend'] = tokenizer.backend_tokenizer.to_str()
    tokenizer_hash = hashlib.sha256(json.dumps(tokenizer_info, sort_keys=True, default=str).encode()).hexdigest()
    model_config = model.config.to_dict()
    model_config.pop('_name_or_path', None)
    config_hash = hashlib.sha256(json.dumps(model_config, sort_keys=True, default=str).encode()).hexdigest()
    return config_hash, tokenizer_hash


def checkpoint_identity(c, model, tokenizer, task, device, labels, data_hash):
    settings, data = c['train'], c['data']
    training_keys = ('model_source', 'model_revision', 'trust_remote_code', 'epochs', 'learning_rate',
                     'batch_size', 'gradient_accumulation_steps', 'weight_decay', 'max_grad_norm', 'seed')
    data_keys = ('text_column', 'label_column', 'positive_label', 'max_length', 'prompt_template')
    config_hash, tokenizer_hash = model_tokenizer_fingerprints(model, tokenizer)
    return {'training': {k: settings.get(k) for k in training_keys},
            'data': {k: data.get(k) for k in data_keys}, 'train_csv_sha256': data_hash,
            'task': task, 'labels': labels, 'device_type': device.type,
            'precision': effective_precision(c, device), 'model_config_sha256': config_hash,
            'tokenizer_sha256': tokenizer_hash, 'model_commit': getattr(model.config, '_commit_hash', None)}


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.checkpoint-', suffix='.pt', delete=False) as handle:
            tmp = Path(handle.name)
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def train(c, out, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    data, settings = c['data'], c['train']
    model_root = MODEL_ROOT / validate_model_name(settings['output_model_name']) / 'weights'
    model_dir = model_root / 'final'
    if (model_root.exists() and not settings.get('resume_from')) or model_dir.exists() or (model_root / 'config.json').exists():
        raise FileExistsError(f'Trained model directory already exists: {model_root}; choose a different output_model_name')
    LOG.info('Reading training data: %s', data['train_csv'])
    _, rows = read_csv(data['train_csv'], data['text_column'], data['label_column'])
    train_hash = sha256(data['train_csv'])
    classes = sorted({r[data['label_column']].strip() for r in rows})
    labels = None
    evaluation = None
    if data.get('eval_csv'):
        _, evaluation = read_csv(data['eval_csv'], data['text_column'], data['label_column'])
        unknown = {r[data['label_column']].strip() for r in evaluation} - set(classes)
        if unknown:
            raise ValueError(f'Evaluation CSV has labels absent from training: {sorted(unknown)}')
    seed = settings['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    repo, opts = resolve_model_source(c['train']['model_source'], c['train']['model_revision'], c['train']['cache_dir'])
    opts['trust_remote_code'] = c['train'].get('trust_remote_code', False)
    task = select_task(repo, opts, c['train'].get('task', 'auto'))
    LOG.info('Loading training model %s (task: %s)', repo, task)
    if task == 'classification':
        if len(classes) != 2 or data['positive_label'] not in classes:
            raise ValueError(f'Classifier training requires exactly two classes including positive_label; found {classes}')
        negative = next(x for x in classes if x != data['positive_label'])
        labels = {negative: 0, data['positive_label']: 1}
    tokenizer = AutoTokenizer.from_pretrained(repo, **opts)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError('Model tokenizer has no padding or EOS token; choose a compatible sequence-classification model')
        tokenizer.pad_token = tokenizer.eos_token
    if task == 'classification':
        model = AutoModelForSequenceClassification.from_pretrained(repo, num_labels=2, id2label={0: negative, 1: data['positive_label']}, label2id=labels, **opts)
    else:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
        cls = AutoModelForCausalLM if task == 'causal_lm' else AutoModelForSeq2SeqLM
        model = cls.from_pretrained(repo, **opts)
    model.config.phishing_task = task
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    loader = make_loader(rows, tokenizer, c, labels, shuffle=True) if task == 'classification' else make_generative_loader(rows, tokenizer, c, task, model, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    accum = settings['gradient_accumulation_steps']
    steps = math.ceil(len(loader) / accum) * settings['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=steps)
    precision = effective_precision(c, device)
    LOG.info('Training precision: %s', precision)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and precision == 'fp16'))
    optimizer.zero_grad(set_to_none=True)
    identity = checkpoint_identity(c, model, tokenizer, task, device, labels, train_hash)
    start_epoch = 0
    if settings.get('resume_from'):
        checkpoint = torch.load(settings['resume_from'], map_location='cpu', weights_only=False)
        if checkpoint.get('format_version') != 1 or checkpoint.get('identity') != identity:
            raise ValueError('Checkpoint does not match the model, tokenizer, data, device, or training settings')
        start_epoch = checkpoint['epoch']
        if not isinstance(start_epoch, int) or not 0 < start_epoch <= settings['epochs']:
            raise ValueError('Checkpoint epoch is invalid for this training configuration')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        random.setstate(checkpoint['python_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        LOG.info('Resuming after epoch %d from %s', start_epoch, settings['resume_from'])
    LOG.info('Starting training: %d records, %d batches per epoch, %d epochs', len(rows), len(loader), settings['epochs'])
    for epoch in range(start_epoch, settings['epochs']):
        LOG.info('Epoch %d/%d started', epoch + 1, settings['epochs'])
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            group_size = min(accum, len(loader) - (i // accum) * accum)
            with autocast(device, precision):
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
            if i == 0 or (i + 1) % settings.get('progress_every', 1) == 0 or i + 1 == len(loader):
                LOG.info('Epoch %d/%d batch %d/%d loss %.6f', epoch + 1, settings['epochs'], i + 1, len(loader), loss.item())
        LOG.info('Epoch %d/%d mean batch loss %.6f', epoch + 1, settings['epochs'], total_loss / len(loader))
        checkpoint_path = model_root / 'checkpoints' / out.name / f'epoch-{epoch + 1:03d}.pt'
        save_checkpoint(checkpoint_path, {'format_version': 1, 'epoch': epoch + 1, 'identity': identity,
                        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                        'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None})
        LOG.info('Saved checkpoint: %s', checkpoint_path)
    if evaluation and task != 'classification':
        validation_config = dict(c)
        validation_config['data'] = dict(data, inference_csv=data['eval_csv'], selected_dataset='validation')
        validation_dir = out / f'validation-epoch-{settings["epochs"]:03d}'
        validation_dir.mkdir()
        fields, _ = read_csv(data['eval_csv'], data['text_column'], data['label_column'])
        predictions = predict_generative(validation_config, validation_dir, device, settings['model_source'], repo, opts, tokenizer, fields, evaluation, task, loaded_model=model)
        save_prediction_metrics(validation_config, validation_dir, predictions)
    if evaluation and task == 'classification':
        tp = fp = tn = fn = 0
        model.eval()
        with torch.inference_mode():
            for batch in make_loader(evaluation, tokenizer, c, labels):
                batch = {k: v.to(device) for k, v in batch.items()}
                target = batch.pop('labels')
                with autocast(device, precision): logits = model(**batch).logits
                predicted = torch.softmax(logits.float(), dim=-1)[:, 1] >= c.get('validation_threshold', c['inference']['threshold'])
                positive = target == 1
                tp += (predicted & positive).sum().item()
                fp += (predicted & ~positive).sum().item()
                tn += (~predicted & ~positive).sum().item()
                fn += (~predicted & positive).sum().item()
        metrics = calculate_metrics(tp, fp, tn, fn)
        metrics.update({'positive_label': data['positive_label'], 'threshold': c.get('validation_threshold', c['inference']['threshold']), 'checkpoint': f'epoch-{settings["epochs"]:03d}'})
        LOG.info('Validation metrics: %s', json.dumps(metrics))
        (out / f'evaluation-epoch-{settings["epochs"]:03d}.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    model_root.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(exist_ok=False)
    try:
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    except Exception:
        shutil.rmtree(model_dir)
        raise
    metadata = common_metadata(c, device)
    metadata.update({'resolved_model_revision': getattr(model.config, '_commit_hash', None), 'resolved_tokenizer_revision': tokenizer.init_kwargs.get('_commit_hash'), 'task': task, 'labels': labels if labels is not None else classes, 'trained_model_dir': str(model_dir), 'training_csv_sha256': train_hash, 'eval_csv_sha256': sha256(data['eval_csv']) if data.get('eval_csv') else None})
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    LOG.info('Saved trained model to %s', model_dir)


def prediction_path(c, out):
    """Place each prediction CSV inside its model/checkpoint run folder."""
    return out / 'predictions.csv'


def load_epoch_model(checkpoint_path, c):
    """Restore epoch weights for inference from the checkpoint and its base model."""
    from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('format_version') != 1 or not isinstance(checkpoint.get('identity'), dict):
        raise ValueError('Unrecognized checkpoint format')
    identity = checkpoint['identity']
    task = identity['task']
    requested = c['inference'].get('task', 'auto')
    if requested not in ('auto', task):
        raise ValueError('Configured inference task does not match checkpoint task')
    training = identity['training']
    reference, opts = resolve_model_source(training['model_source'], training['model_revision'], c['inference']['cache_dir'])
    opts['trust_remote_code'] = c['inference'].get('trust_remote_code', False)
    if not opts.get('local_files_only') and identity.get('model_commit'):
        opts['revision'] = identity['model_commit']
    tokenizer = AutoTokenizer.from_pretrained(reference, **opts)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError('Checkpoint tokenizer has no padding or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    if task == 'classification':
        labels = identity['labels']
        if not isinstance(labels, dict) or set(labels.values()) != {0, 1}:
            raise ValueError('Checkpoint does not contain a binary label mapping')
        id2label = {index: label for label, index in labels.items()}
        model = AutoModelForSequenceClassification.from_pretrained(reference, num_labels=2, id2label=id2label, label2id=labels, **opts)
    elif task in ('causal_lm', 'seq2seq'):
        cls = AutoModelForCausalLM if task == 'causal_lm' else AutoModelForSeq2SeqLM
        model = cls.from_pretrained(reference, **opts)
    else:
        raise ValueError('Unsupported checkpoint task')
    model.config.phishing_task = task
    model.config.pad_token_id = tokenizer.pad_token_id
    config_hash, tokenizer_hash = model_tokenizer_fingerprints(model, tokenizer)
    if config_hash != identity['model_config_sha256'] or tokenizer_hash != identity['tokenizer_sha256']:
        raise ValueError('Checkpoint base model or tokenizer has changed; use the original revision')
    model.load_state_dict(checkpoint['model'])
    return model, tokenizer, task, reference, opts


def predict_generative(c, out, device, model_source, reference, opts, tokenizer, fields, rows, task, loaded_model=None, checkpoint_path=None):
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
    if 'model_response' in fields:
        raise ValueError('Inference CSV already contains model_response')
    cls = AutoModelForCausalLM if task == 'causal_lm' else AutoModelForSeq2SeqLM
    if loaded_model is None:
        model, info = cls.from_pretrained(reference, output_loading_info=True, **opts)
        if info['missing_keys'] or info['mismatched_keys'] or info['error_msgs']:
            raise ValueError('Inference model weights are incomplete')
    else:
        model = loaded_model
    model = model.to(device).eval()
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError('Inference tokenizer has no padding or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    if task == 'causal_lm':
        tokenizer.padding_side = 'left'
    inf = c['inference']
    settings = {'do_sample': inf.get('do_sample', False), 'max_new_tokens': inf.get('max_new_tokens', 32), 'pad_token_id': tokenizer.pad_token_id}
    if settings['do_sample']:
        settings.update(temperature=inf.get('temperature', 0.7), top_p=inf.get('top_p', 0.9), top_k=0)
    # Use a fresh configuration rather than inheriting model-specific sampling defaults.
    from transformers import GenerationConfig
    settings['eos_token_id'] = model.generation_config.eos_token_id
    if task == 'seq2seq':
        settings['decoder_start_token_id'] = model.generation_config.decoder_start_token_id
        settings['bos_token_id'] = model.generation_config.bos_token_id
    gen = GenerationConfig(**settings)
    seed = c['train']['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    output = prediction_path(c, out)
    with output.open('x', newline='', encoding='utf-8') as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=fields + ['model_response'])
        writer.writeheader()
        size = inf['batch_size']
        for offset in range(0, len(rows), size):
            group = rows[offset:offset + size]
            inputs = tokenizer([prompt_for(c, r[c['data']['text_column']]) for r in group], padding=True, truncation=True, max_length=c['data']['max_length'], return_tensors='pt').to(device)
            with autocast(device, effective_precision(c, device)):
                generated = model.generate(**inputs, generation_config=gen)
            if task == 'causal_lm': generated = generated[:, inputs['input_ids'].shape[1]:]
            outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, response in zip(group, outputs):
                writer.writerow({**row, 'model_response': response})
    meta = common_metadata(c, device)
    meta.update({'inference_model_source': model_source, 'inference_model_resolved': reference, 'inference_model_revision': opts.get('revision') if not opts.get('local_files_only') else None, 'inference_model_commit': getattr(model.config, '_commit_hash', None), 'task': task, 'checkpoint': str(checkpoint_path) if checkpoint_path else 'final', 'generation_settings': settings, 'predictions_csv': str(output), 'test_csv': c['data']['inference_csv'], 'inference_csv_sha256': sha256(c['data']['inference_csv']), 'count': len(rows)})
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    LOG.info('Saved %d generated responses to %s', len(rows), output)
    return output


def predict(c, out, device, model_source, checkpoint_path=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    data = c['data']
    fields, rows = read_csv(data['inference_csv'], data['text_column'])
    loaded_model = None
    if checkpoint_path:
        loaded_model, tokenizer, task, reference, opts = load_epoch_model(checkpoint_path, c)
    else:
        reference, opts = resolve_model_source(model_source, c['inference'].get('model_revision', 'main'), c['inference']['cache_dir'])
        opts['trust_remote_code'] = c['inference'].get('trust_remote_code', False)
        task = select_task(reference, opts, c['inference'].get('task', 'auto'))
        tokenizer = AutoTokenizer.from_pretrained(reference, **opts)
    if task != 'classification':
        return predict_generative(c, out, device, model_source, reference, opts, tokenizer, fields, rows, task, loaded_model, checkpoint_path)
    extra = ['predicted_label', 'phishing_probability']
    if any(x in fields for x in extra):
        raise ValueError(f'Inference CSV already contains an output column: {extra}')
    if loaded_model is not None:
        model = loaded_model
    else:
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(reference, output_loading_info=True, **opts)
        if loading_info['missing_keys'] or loading_info['mismatched_keys'] or loading_info['error_msgs']:
            raise ValueError('Inference model did not load all classifier weights; choose an already trained binary classifier')
    model = model.to(device)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError('Inference tokenizer has no padding or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    labels = {int(k): v for k, v in model.config.id2label.items()}
    if set(labels) != {0, 1}:
        raise ValueError('Inference model must be a binary sequence classifier')
    positive = c['inference'].get('positive_label_index')
    if positive is None:
        matches = [index for index, label in labels.items() if label == data['positive_label']]
        if len(matches) != 1:
            raise ValueError('Model labels do not identify positive_label; set inference.positive_label_index explicitly')
        positive = matches[0]
    negative = 1 - positive
    loader = make_loader(rows, tokenizer, c, batch_size=c['inference']['batch_size'])
    model.eval()
    output = prediction_path(c, out)
    offset = 0
    with output.open('x', newline='', encoding='utf-8') as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=fields + extra)
        writer.writeheader()
        for batch in loader:
            with autocast(device, effective_precision(c, device)):
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            probs = torch.softmax(logits.float(), dim=-1)[:, positive].cpu().tolist()
            for prob in probs:
                writer.writerow({**rows[offset], 'predicted_label': labels[positive if prob >= c['inference']['threshold'] else negative], 'phishing_probability': prob})
                offset += 1
    meta = common_metadata(c, device)
    meta.update({'inference_model_source': model_source, 'inference_model_resolved': reference, 'inference_model_revision': opts.get('revision') if not opts.get('local_files_only') else None, 'inference_model_commit': getattr(model.config, '_commit_hash', None), 'task': task, 'checkpoint': str(checkpoint_path) if checkpoint_path else 'final', 'positive_label_index': positive, 'predictions_csv': str(output), 'test_csv': data['inference_csv'], 'inference_csv_sha256': sha256(data['inference_csv']), 'count': offset, 'threshold': c['inference']['threshold']})
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    LOG.info('Saved %d predictions to %s', offset, output)
    return output


def save_prediction_metrics(c, out, path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        fields = csv.DictReader(f).fieldnames or []
    if c['data']['label_column'] not in fields:
        LOG.info('No ground-truth label column; metrics were not calculated')
        return
    try:
        metrics = analyze_predictions(path, c['data']['label_column'], c['data']['positive_label'])
    except ValueError as exc:
        LOG.warning('Metrics not calculated: %s', exc)
        return
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    LOG.info('Saved metrics: %s', json.dumps(metrics))


def main(forced_action=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    if forced_action is None:
        sub = parser.add_subparsers(dest='action', required=True)
        sub.add_parser('train', help='train and validate the model')
        prediction = sub.add_parser('test', help='test the model on data/test.csv')
        prediction.add_argument('--checkpoint', help='test an epoch checkpoint')
        prediction.add_argument('--csv', help='test CSV path (default: project data/test.csv)')
    elif forced_action == 'train':
        parser.add_argument('--resume', help='resume training from an epoch checkpoint')
        parser.add_argument('--check', action='store_true', help='validate configuration and inspect local hardware')
    elif forced_action in ('predict', 'test'):
        forced_action = 'test'
        parser.add_argument('--checkpoint', help='test an epoch checkpoint')
        parser.add_argument('--csv', help='test CSV path (default: project data/test.csv)')
    args = parser.parse_args()
    if forced_action is not None:
        args.action = 'check' if forced_action == 'train' and args.check else forced_action
    try:
        c = read_config(args.config, getattr(args, 'train_csv', None))
        if getattr(args, 'resume', None):
            c['train']['resume_from'] = args.resume
        device = configure_device(c)
        check_optional_vision_dependency()
        if args.action == 'check':
            print(f'Configuration valid. Device: {device}; model: {c["train"]["model_source"]}; cache: {c["train"]["cache_dir"]}')
            return 0
        if args.action == 'test':
            test_csv = Path(args.csv).expanduser() if args.csv else ROOT / 'data' / 'test.csv'
            c['data']['inference_csv'] = str(test_csv.resolve())
            c['data']['selected_dataset'] = 'test'
            args.model_dir = c['inference'].get('model_source') or c['inference'].get('model_dir')
            if not isinstance(args.model_dir, str) or not args.model_dir.strip():
                if getattr(args, 'checkpoint', None) or c['inference'].get('checkpoint'):
                    args.model_dir = 'checkpoint'
                else:
                    raise ValueError('Set test.model_source in config.yaml')
        checkpoint = (getattr(args, 'checkpoint', None) or c['inference'].get('checkpoint')) if args.action == 'test' else None
        out = setup_run(c, 'predict' if args.action == 'test' else args.action, getattr(args, 'model_dir', None), checkpoint)
        try:
            LOG.info('Device: %s%s; precision: %s', device, f' ({torch.cuda.get_device_name(device)})' if device.type == 'cuda' else ' (CUDA unavailable; using CPU)', effective_precision(c, device))
            if args.action == 'train': train(c, out, device)
            else:
                LOG.info('Testing data: %s', c['data']['inference_csv'])
                predictions = predict(c, out, device, args.model_dir, checkpoint)
                save_prediction_metrics(c, out, predictions)
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
