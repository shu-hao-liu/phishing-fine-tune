import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from bootstrap_test import bootstrap_metrics, load_counts


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.auto-commit-test-', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_confusion_count_bootstrap_matches_row_resampling_distribution(self):
        counts = (3, 1, 4, 2)
        result = bootstrap_metrics(counts, resamples=5000, confidence=.95, seed=42)
        self.assertEqual(result, bootstrap_metrics(counts, resamples=5000, confidence=.95, seed=42))
        self.assertEqual(result['point_metrics']['accuracy'], .7)
        self.assertEqual(result['point_metrics']['precision'], .75)
        self.assertEqual(result['point_metrics']['recall'], .6)
        self.assertAlmostEqual(result['point_metrics']['f1'], 2 / 3)
        # Exact binomial confidence quantiles for accuracy with 7/10 correct.
        probabilities = [__import__('math').comb(10, i) * .7**i * .3**(10-i) for i in range(11)]
        cdf = np.cumsum(probabilities)
        exact = [np.searchsorted(cdf, q) / 10 for q in (.025, .975)]
        np.testing.assert_allclose(result['metrics']['accuracy']['confidence_interval'], exact, atol=.1)

    def test_undefined_metrics_are_explicit(self):
        result = bootstrap_metrics((0, 0, 10, 0), resamples=100, seed=42)
        for metric in ('precision', 'recall', 'f1'):
            self.assertIsNone(result['metrics'][metric]['estimate'])
            self.assertIsNone(result['metrics'][metric]['confidence_interval'])
            self.assertEqual(result['metrics'][metric]['undefined_resamples'], 100)
        self.assertEqual(result['metrics']['accuracy']['confidence_interval'], [1, 1])

    def test_csv_and_cli_with_custom_column_labels(self):
        path = self.root / 'predictions.csv'
        path.write_text('truth,pred\nphish,phish\nsafe,phish\nsafe,safe\nphish,safe\n')
        self.assertEqual(load_counts(path, 'truth', 'phish', 'pred')[0], (1, 1, 1, 1))
        output = self.root / 'result.json'
        cmd = [sys.executable, str(Path(__file__).parent / 'bootstrap_test.py'), str(path),
               '--label-column', 'truth', '--prediction-column', 'pred', '--positive-label', 'phish',
               '--resamples', '100', '--seed', '7', '--output', str(output)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        saved = json.loads(output.read_text())
        self.assertEqual(saved['point_metrics']['count'], 4)
        self.assertEqual(saved['point_metrics']['accuracy'], .5)
        self.assertEqual(saved['positive_label'], 'phish')
        self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)

    def test_invalid_input_and_configuration(self):
        path = self.root / 'bad.csv'
        path.write_text('label,predicted_label\n1.0,maybe\n0.0,0.0\n')
        with self.assertRaisesRegex(ValueError, 'binary labels'):
            load_counts(path, 'label', '1.0')
        with self.assertRaisesRegex(ValueError, 'confidence'):
            bootstrap_metrics((1, 1, 1, 1), confidence=1)
        with self.assertRaisesRegex(ValueError, 'resamples'):
            bootstrap_metrics((1, 1, 1, 1), resamples=0)


if __name__ == '__main__':
    unittest.main()
