import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DistilBertTokenizerFast, DistilBertConfig, GPT2Config, T5Config, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import finetune as p


class PhishingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.auto-commit-test-', dir=p.ROOT)
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.weights_patch = patch.object(p, 'WEIGHTS_ROOT', self.root / 'weights')
        self.weights_patch.start()
        self.addCleanup(self.weights_patch.stop)
        self.c = yaml.safe_load((p.ROOT / 'config.yaml').read_text())
        self.c['data'].update(train_csv=str(self.root / 'train' / 'train.csv'), inference_csv=str(self.root / 'infer' / 'data.csv'), eval_csv=None, text_column='text', positive_label='phishing', max_length=16)
        self.c['train']['task'] = 'classification'
        self.c['inference']['task'] = 'classification'
        self.c['train'].update(output_model_name='Model-1', epochs=1, batch_size=2, gradient_accumulation_steps=2)
        self.c['train']['output_dir'] = self.c['inference']['output_dir'] = str(self.root / 'out')
        for name in ('train', 'infer'):
            (self.root / name).mkdir()
        (self.root / 'train' / 'train.csv').write_text('text,label\nclick now,phishing\nhello friend,legitimate\nverify now,phishing\n')
        (self.root / 'infer' / 'data.csv').write_text('text,id,label\nhello friend,1,legitimate\nclick now,2,phishing\n')

    def test_three_config_sections_and_device_selection(self):
        self.assertEqual(set(self.c), {'train', 'data', 'inference'})
        self.c['train']['precision'] = 'fp32'
        with patch.object(torch.cuda, 'is_available', return_value=True):
            self.assertEqual(p.configure_device(self.c).type, 'cuda')
        with patch.object(torch.cuda, 'is_available', return_value=False):
            self.assertEqual(p.configure_device(self.c).type, 'cpu')

    def test_automatic_precision(self):
        self.c['train']['precision'] = 'auto'
        self.assertEqual(p.effective_precision(self.c, torch.device('cpu')), 'fp32')
        with patch.object(torch.cuda, 'is_bf16_supported', return_value=True):
            self.assertEqual(p.effective_precision(self.c, torch.device('cuda')), 'bf16')
        with patch.object(torch.cuda, 'is_bf16_supported', return_value=False):
            self.assertEqual(p.effective_precision(self.c, torch.device('cuda')), 'fp16')

    def test_inference_never_feeds_label_field(self):
        vocab = self.root / 'vocab.txt'
        vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nfriend\n')
        tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab))
        rows = [{'text': 'hello friend', 'label': 'SENTINEL_LABEL'}]
        dataset = p.TextDataset(rows, tokenizer, 'text', 16, labels=None, label_col='label')
        self.assertEqual(dataset[0], dict(tokenizer('hello friend', truncation=True, max_length=16)))
        self.assertNotIn('labels', dataset[0])
        self.assertNotIn('SENTINEL_LABEL', p.prompt_for(self.c, rows[0]['text']))
        self.c['data']['text_column'] = self.c['data']['label_column']
        config = self.root / 'invalid.yaml'
        config.write_text(yaml.safe_dump(self.c))
        with self.assertRaisesRegex(ValueError, 'prevent label leakage'):
            p.read_config(config)

    def test_url_parsing(self):
        self.assertEqual(p.repository_id('https://huggingface.co/example/model/'), 'example/model')
        with self.assertRaises(ValueError): p.repository_id('https://example.com/model')
        with self.assertRaises(ValueError): p.repository_id('https://huggingface.co/a/b/tree/main')

    def test_same_folder_allowed_but_same_file_rejected(self):
        self.c['data']['inference_csv'] = str(self.root / 'train' / 'infer.csv')
        config = self.root / 'test.yaml'
        config.write_text(yaml.safe_dump(self.c))
        p.read_config(config)
        self.c['data']['inference_csv'] = self.c['data']['train_csv']
        config.write_text(yaml.safe_dump(self.c))
        with self.assertRaisesRegex(ValueError, 'different files'):
            p.read_config(config)

    def test_csv_validation(self):
        bad = self.root / 'bad.csv'
        bad.write_text('text,label\n,phishing\n')
        with self.assertRaisesRegex(ValueError, 'missing text'): p.read_csv(bad, 'text', 'label')

    def test_epoch_checkpoint_resume_matches_uninterrupted(self):
        vocab = self.root / 'vocab.txt'
        vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nclick\nnow\nhello\nfriend\nverify\n')
        tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab))
        base = self.root / 'resume-base'
        base.mkdir()
        tokenizer.save_pretrained(base)
        cfg = DistilBertConfig(vocab_size=len(tokenizer), dim=16, hidden_dim=32, n_layers=1, n_heads=2, max_position_embeddings=32, num_labels=2)
        torch.manual_seed(123)
        AutoModelForSequenceClassification.from_config(cfg).save_pretrained(base)
        self.c['train'].update(model_source=str(base), task='classification', epochs=2, output_model_name='baseline')
        self.c['data']['max_length'] = 16
        baseline = self.root / 'baseline-run'
        baseline.mkdir()
        p.train(self.c, baseline, torch.device('cpu'))
        self.assertEqual(len(list((baseline / 'checkpoints').glob('*.pt'))), 2)
        self.c['train']['output_model_name'] = 'resumed'
        interrupted = self.root / 'interrupted-run'
        interrupted.mkdir()
        original_save = p.save_checkpoint
        def interrupt_after_save(path, state):
            original_save(path, state)
            if state['epoch'] == 1:
                raise RuntimeError('simulated interruption')
        with patch.object(p, 'save_checkpoint', side_effect=interrupt_after_save):
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                p.train(self.c, interrupted, torch.device('cpu'))
        self.c['train']['resume_from'] = str(interrupted / 'checkpoints' / 'epoch-001.pt')
        resumed = self.root / 'resumed-run'
        resumed.mkdir()
        p.train(self.c, resumed, torch.device('cpu'))
        first = AutoModelForSequenceClassification.from_pretrained(p.WEIGHTS_ROOT / 'baseline').state_dict()
        second = AutoModelForSequenceClassification.from_pretrained(p.WEIGHTS_ROOT / 'resumed').state_dict()
        self.assertTrue(all(torch.equal(first[k], second[k]) for k in first))
        self.assertEqual([x.name for x in (resumed / 'checkpoints').glob('*.pt')], ['epoch-002.pt'])
        self.c['train'].update(seed=99, output_model_name='mismatch')
        with self.assertRaisesRegex(ValueError, 'Checkpoint does not match'):
            p.train(self.c, resumed, torch.device('cpu'))

    def test_generative_models_train_and_predict(self):
        vocab = self.root / 'vocab.txt'
        vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nclick\nnow\nhello\nfriend\nverify\nphishing\nlegitimate\n')
        tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab))
        self.c['data']['max_length'] = 16
        self.c['data']['prompt_template'] = '{text}'
        self.c['inference'].update(max_new_tokens=2, do_sample=False)
        for task in ('causal_lm', 'seq2seq'):
            with self.subTest(task=task):
                base = self.root / ('base-' + task)
                base.mkdir()
                tokenizer.save_pretrained(base)
                if task == 'causal_lm':
                    cfg = GPT2Config(vocab_size=len(tokenizer), n_positions=32, n_ctx=32, n_embd=16, n_layer=1, n_head=2, bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.sep_token_id, pad_token_id=tokenizer.pad_token_id)
                    model = AutoModelForCausalLM.from_config(cfg)
                else:
                    cfg = T5Config(vocab_size=len(tokenizer), d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, decoder_start_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.sep_token_id, pad_token_id=tokenizer.pad_token_id)
                    model = AutoModelForSeq2SeqLM.from_config(cfg)
                model.save_pretrained(base)
                self.c['train'].update(model_source=str(base), task=task)
                self.c['train']['output_model_name'] = task
                self.c['inference']['task'] = 'auto'
                run = self.root / ('run-' + task)
                run.mkdir()
                p.train(self.c, run, torch.device('cpu'))
                out = self.root / ('pred-' + task)
                out.mkdir()
                p.predict(self.c, out, torch.device('cpu'), str(p.WEIGHTS_ROOT / task))
                with (out / 'predictions.csv').open(newline='') as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 2)
                self.assertIn('model_response', rows[0])
                self.assertEqual(json.loads((out / 'metadata.json').read_text())['task'], task)

    def test_training_and_inference_offline(self):
        vocab = self.root / 'vocab.txt'
        vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nclick\nnow\nhello\nfriend\nverify\n')
        tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab))
        model_cfg = DistilBertConfig(vocab_size=10, dim=16, hidden_dim=32, n_layers=1, n_heads=2, max_position_embeddings=32, num_labels=2)
        model_cfg.id2label = {0: 'legitimate', 1: 'phishing'}
        model_cfg.label2id = {'legitimate': 0, 'phishing': 1}
        model = AutoModelForSequenceClassification.from_config(model_cfg)
        validation = self.root / 'validation.csv'
        validation.write_text('text,label\nhello friend,legitimate\nclick now,phishing\n')
        self.c['data']['eval_csv'] = str(validation)
        train_dir = self.root / 'trained'
        train_dir.mkdir()
        p.setup_run(self.c, 'test')
        with patch.object(AutoTokenizer, 'from_pretrained', return_value=tokenizer), patch.object(AutoModelForSequenceClassification, 'from_pretrained', return_value=model):
            p.train(self.c, train_dir, torch.device('cpu'))
        validation_metrics = json.loads((train_dir / 'evaluation.json').read_text())
        self.assertEqual(validation_metrics['count'], 2)
        self.assertTrue({'accuracy', 'precision', 'recall', 'f1'} <= set(validation_metrics))
        pred_dir = self.root / 'predicted'
        pred_dir.mkdir()
        p.predict(self.c, pred_dir, torch.device('cpu'), str(p.WEIGHTS_ROOT / 'Model-1'))
        with (pred_dir / 'predictions.csv').open(newline='') as f: rows = list(csv.DictReader(f))
        self.assertEqual([r['id'] for r in rows], ['1', '2'])
        p.save_prediction_metrics(self.c, pred_dir)
        self.assertEqual(json.loads((pred_dir / 'metrics.json').read_text())['count'], 2)
        self.assertTrue(all(r['predicted_label'] in ('legitimate', 'phishing') and 0 <= float(r['phishing_probability']) <= 1 for r in rows))
        self.assertTrue((p.WEIGHTS_ROOT / 'Model-1' / 'config.json').exists())
        self.assertEqual(p.resolve_model_source('local:Model-1', 'main', 'cache')[0], str((p.WEIGHTS_ROOT / 'Model-1').resolve()))
        with self.assertRaises(FileExistsError):
            p.train(self.c, train_dir, torch.device('cpu'))
        # Train again from the already saved local weights, with no Hub download.
        self.c['train']['model_source'] = 'local:Model-1'
        self.c['train']['output_model_name'] = 'Model-2'
        p.train(self.c, train_dir, torch.device('cpu'))
        self.assertTrue((p.WEIGHTS_ROOT / 'Model-2' / 'config.json').exists())

        # A separately selected Hub classifier can use a different positive index.
        model.config.id2label = {0: 'phishing', 1: 'legitimate'}
        self.c['inference'].update(model_source='other/classifier', model_revision='revision123', positive_label_index=0)
        alternate_dir = self.root / 'alternate'
        alternate_dir.mkdir()
        info = {'missing_keys': [], 'mismatched_keys': [], 'error_msgs': []}
        with patch.object(AutoTokenizer, 'from_pretrained', return_value=tokenizer) as tokenizer_load, patch.object(AutoModelForSequenceClassification, 'from_pretrained', return_value=(model, info)) as model_load:
            p.predict(self.c, alternate_dir, torch.device('cpu'), self.c['inference']['model_source'])
        self.assertEqual(model_load.call_args.args[0], 'other/classifier')
        self.assertEqual(tokenizer_load.call_args.kwargs['revision'], 'revision123')
        metadata = json.loads((alternate_dir / 'metadata.json').read_text())
        self.assertEqual(metadata['positive_label_index'], 0)
        self.assertEqual(metadata['inference_model_source'], 'other/classifier')



if __name__ == '__main__': unittest.main()
