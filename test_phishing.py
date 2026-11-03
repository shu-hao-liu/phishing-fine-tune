import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DistilBertTokenizerFast, DistilBertConfig
import finetune as p


class PhishingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=p.ROOT)
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.c = yaml.safe_load((p.ROOT / 'config.yaml').read_text())
        self.c['data'].update(train_csv=str(self.root / 'train' / 'train.csv'), inference_csv=str(self.root / 'infer' / 'data.csv'), max_length=16)
        self.c['training'].update(epochs=1, batch_size=2, gradient_accumulation_steps=2)
        self.c['output_dir'] = str(self.root / 'out')
        for name in ('train', 'infer'):
            (self.root / name).mkdir()
        (self.root / 'train' / 'train.csv').write_text('text,label\nclick now,phishing\nhello friend,legitimate\nverify now,phishing\n')
        (self.root / 'infer' / 'data.csv').write_text('text,id\nhello friend,1\nclick now,2\n')

    def test_url_parsing(self):
        self.assertEqual(p.repository_id('https://huggingface.co/example/model/'), 'example/model')
        with self.assertRaises(ValueError): p.repository_id('https://example.com/model')
        with self.assertRaises(ValueError): p.repository_id('https://huggingface.co/a/b/tree/main')

    def test_separate_paths(self):
        self.c['data']['inference_csv'] = str(self.root / 'train' / 'infer.csv')
        config = self.root / 'test.yaml'
        config.write_text(yaml.safe_dump(self.c))
        with self.assertRaisesRegex(ValueError, 'different directories'): p.read_config(config)

    def test_csv_validation(self):
        bad = self.root / 'bad.csv'
        bad.write_text('text,label\n,phishing\n')
        with self.assertRaisesRegex(ValueError, 'missing text'): p.read_csv(bad, 'text', 'label')

    def test_training_and_inference_offline(self):
        vocab = self.root / 'vocab.txt'
        vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nclick\nnow\nhello\nfriend\nverify\n')
        tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab))
        model_cfg = DistilBertConfig(vocab_size=10, dim=16, hidden_dim=32, n_layers=1, n_heads=2, max_position_embeddings=32, num_labels=2)
        model_cfg.id2label = {0: 'legitimate', 1: 'phishing'}
        model_cfg.label2id = {'legitimate': 0, 'phishing': 1}
        model = AutoModelForSequenceClassification.from_config(model_cfg)
        train_dir = self.root / 'trained'
        train_dir.mkdir()
        p.setup_run(self.c, 'test')
        with patch.object(AutoTokenizer, 'from_pretrained', return_value=tokenizer), patch.object(AutoModelForSequenceClassification, 'from_pretrained', return_value=model):
            p.train(self.c, train_dir, torch.device('cpu'))
        pred_dir = self.root / 'predicted'
        pred_dir.mkdir()
        p.predict(self.c, pred_dir, torch.device('cpu'), str(train_dir / 'model'))
        with (pred_dir / 'predictions.csv').open(newline='') as f: rows = list(csv.DictReader(f))
        self.assertEqual([r['id'] for r in rows], ['1', '2'])
        self.assertTrue(all(r['predicted_label'] in ('legitimate', 'phishing') and 0 <= float(r['phishing_probability']) <= 1 for r in rows))
        self.assertTrue((train_dir / 'model' / 'config.json').exists())


if __name__ == '__main__': unittest.main()
