import csv
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from roc_auc_test import calculate_roc_auc, read_scores


class RocAucTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='.auto-commit-test-', dir=Path(__file__).parent)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_standard_example_and_curve(self):
        auc, curve = calculate_roc_auc([(0.1, False), (0.4, False), (0.35, True), (0.8, True)])
        self.assertEqual(auc, .75)
        self.assertEqual((curve[0]['false_positive_rate'], curve[0]['true_positive_rate']), (0, 0))
        self.assertEqual((curve[-1]['false_positive_rate'], curve[-1]['true_positive_rate']), (1, 1))
        self.assertEqual([point['threshold'] for point in curve], [None, .8, .4, .35, .1])

    def test_ties_match_pairwise_ranking(self):
        generator = random.Random(29)
        for _ in range(20):
            positive = [generator.randrange(5) for _ in range(11)]
            negative = [generator.randrange(5) for _ in range(7)]
            expected = sum(1 if x > y else .5 if x == y else 0 for x in positive for y in negative) / (len(positive) * len(negative))
            actual, _ = calculate_roc_auc([(x, True) for x in positive] + [(y, False) for y in negative])
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(calculate_roc_auc([(1, True), (1, False)])[0], .5)
        self.assertEqual(calculate_roc_auc([(0, True), (1, False)])[0], 0)

    def test_input_validation(self):
        path = self.root / 'data.csv'
        for contents, message in [('label,phishing_probability\n1.0,nan\n0.0,0.5\n', 'Non-finite'),
                                  ('label,phishing_probability\n1.0,0.8\n', 'two label classes'),
                                  ('label,score\n1.0,0.9\n0.0,0.2\n', 'missing')]:
            path.write_text(contents)
            with self.assertRaisesRegex(ValueError, message):
                read_scores(path)

    def test_cli_and_output_safety(self):
        source, output, curve = self.root / 'data.csv', self.root / 'result.json', self.root / 'curve.csv'
        source.write_text('truth,score\nphish,0.8\nsafe,0.4\nphish,0.35\nsafe,0.1\n')
        cmd = [sys.executable, str(Path(__file__).parent / 'roc_auc_test.py'), str(source), '--label-column', 'truth', '--score-column', 'score', '--positive-label', 'phish', '--output', str(output), '--curve-csv', str(curve)]
        process = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(output.read_text())
        self.assertEqual(result['roc_auc'], .75)
        self.assertEqual((result['positive_count'], result['negative_count']), (2, 2))
        with curve.open(newline='') as f:
            self.assertEqual(len(list(csv.DictReader(f))), 5)
        before = output.read_bytes()
        self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
        self.assertEqual(output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
