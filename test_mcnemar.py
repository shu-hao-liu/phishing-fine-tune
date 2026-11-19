import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mcnemar_test import compare_csvs, exact_mcnemar


class McNemarTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='.auto-commit-test-', dir=Path(__file__).parent)
        self.root = Path(temp.name)
        self.addCleanup(temp.cleanup)

    def test_exact_binomial_and_degenerate_cases(self):
        self.assertEqual(exact_mcnemar(0, 0)[0], 1.0)
        self.assertAlmostEqual(exact_mcnemar(1, 9)[0], 22 / 1024)
        self.assertAlmostEqual(exact_mcnemar(9, 1)[0], 22 / 1024)
        self.assertAlmostEqual(exact_mcnemar(0, 4)[0], .125)
        self.assertAlmostEqual(exact_mcnemar(20, 20)[0], 1.0)
        p, logp = exact_mcnemar(0, 2000)
        self.assertEqual(p, 0.0)
        self.assertTrue(math.isfinite(logp) and logp < -300)
        with self.assertRaises(ValueError):
            exact_mcnemar(-1, 2)

    def test_paired_counts_and_cli(self):
        a, b = self.root / 'a.csv', self.root / 'b.csv'
        a.write_text('id,label,predicted_label\na,1,1\nb,0,0\nc,1,0\nd,0,1\ne,1,1\n')
        b.write_text('id,label,predicted_label\na,1,1\nb,0,1\nc,1,1\nd,0,1\ne,1,1\n')
        result = compare_csvs(a, b)
        self.assertEqual((result['both_correct'], result['only_a_correct'], result['only_b_correct'], result['both_incorrect']), (2, 1, 1, 1))
        self.assertAlmostEqual(result['accuracy_a'], .6)
        self.assertAlmostEqual(result['accuracy_b'], .6)
        self.assertAlmostEqual(result['p_value'], 1)
        output = self.root / 'result.json'
        cmd = [sys.executable, str(Path(__file__).parent / 'mcnemar_test.py'), str(a), str(b), '--output', str(output)]
        process = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(output.read_text())['only_a_correct'], 1)
        self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)

    def test_rejects_misaligned_rows_and_incompatible_predictions(self):
        a, b = self.root / 'a.csv', self.root / 'b.csv'
        a.write_text('id,label,predicted_label\na,1,1\nb,0,0\n')
        b.write_text('id,label,predicted_label\nb,0,0\na,1,1\n')
        with self.assertRaisesRegex(ValueError, 'not aligned'):
            compare_csvs(a, b)
        b.write_text('id,label,predicted_label\na,1,maybe\nb,0,0\n')
        with self.assertRaisesRegex(ValueError, 'binary labels'):
            compare_csvs(a, b)
        b.write_text('id,label,predicted_label\na,0,1\nb,0,0\n')
        with self.assertRaisesRegex(ValueError, 'labels differ'):
            compare_csvs(a, b)


if __name__ == '__main__':
    unittest.main()
