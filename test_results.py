import csv
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

from analyze_results import analyze_predictions, calculate_metrics
from split_validation_test import split


class ResultsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='.auto-commit-test-', dir=Path(__file__).parent)
        self.root = Path(temp.name)
        self.addCleanup(temp.cleanup)

    def test_metrics_and_zero_denominator(self):
        m = calculate_metrics(3, 1, 4, 2)
        self.assertEqual(m['accuracy'], 0.7)
        self.assertEqual(m['precision'], 0.75)
        self.assertEqual(m['recall'], 0.6)
        self.assertAlmostEqual(m['f1'], 2 / 3)
        self.assertIsNone(calculate_metrics(0, 0, 3, 2)['precision'])
        self.assertEqual(calculate_metrics(0, 0, 3, 2)['f1'], 0.0)

    def test_asymmetric_confusion_matrix_and_cli(self):
        path = self.root / 'predictions.csv'
        path.write_text('label,predicted_label\n' + '1.0,1.0\n' * 3 + '0.0,1.0\n' + '0.0,0.0\n' * 4 + '1.0,0.0\n' * 2)
        metrics = analyze_predictions(path, 'label', '1.0')
        self.assertEqual((metrics['true_positives'], metrics['false_positives'], metrics['true_negatives'], metrics['false_negatives']), (3, 1, 4, 2))
        self.assertEqual((metrics['accuracy'], metrics['precision'], metrics['recall']), (0.7, 0.75, 0.6))
        self.assertAlmostEqual(metrics['f1'], 2 / 3)
        result = subprocess.run([sys.executable, str(Path(__file__).parent / 'analyze_results.py'), str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.root / 'metrics.json').read_text()), metrics)

    def test_analyzer_rejects_unknown_labels(self):
        path = self.root / 'predictions.csv'
        path.write_text('label,predicted_label\n1.0,1.0\n0.0,1.0\n1.0,0.0\n0.0,0.0\n')
        m = analyze_predictions(path, 'label', '1.0')
        self.assertEqual((m['accuracy'], m['precision'], m['recall'], m['f1']), (0.5, 0.5, 0.5, 0.5))
        path.write_text('label,model_response\n1.0,phishing maybe\n0.0,safe\n')
        with self.assertRaisesRegex(ValueError, 'match the ground-truth'):
            analyze_predictions(path, 'label', '1.0')

    def test_split_is_disjoint_stratified_and_preserves_source(self):
        src, val, test = [self.root / n for n in ('inference.csv', 'validation.csv', 'test.csv')]
        content = 'id,label\n' + ''.join(f'{i},{i % 2}\n' for i in range(11))
        src.write_text(content)
        counts = split(src, val, test, seed=42)
        self.assertEqual(src.read_text(), content)
        self.assertEqual((sum(counts['validation'].values()), sum(counts['test'].values())), (5, 6))
        with val.open(newline='') as f: a = list(csv.DictReader(f))
        with test.open(newline='') as f: b = list(csv.DictReader(f))
        self.assertFalse({r['id'] for r in a} & {r['id'] for r in b})
        self.assertEqual({r['id'] for r in a+b}, {str(i) for i in range(11)})
        with self.assertRaises(FileExistsError): split(src, val, test)


if __name__ == '__main__': unittest.main()
