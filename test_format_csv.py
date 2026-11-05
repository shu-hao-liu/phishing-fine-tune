import csv
from pathlib import Path
import tempfile
import unittest

from format_csv import format_csv


class FormatterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.auto-commit-test-', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.csv'
        self.output = self.root / 'output.csv'
        self.kwargs = dict(mode='train', text_source='message', label_source='category', text_target='body', label_target='label', positive_label='1.0', negative_label='0.0')

    def test_mapping_and_preservation(self):
        self.source.write_text('message,category,subject,extra\n"click, now",phish,hello,discard\n"hello\nfriend",safe,world,discard\n')
        counts = format_csv(self.source, self.output, positive_source='phish', negative_source='safe', **self.kwargs)
        self.assertEqual(dict(counts), {'1.0': 1, '0.0': 1})
        with self.output.open(newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, ['subject', 'body', 'urls', 'label', 'language', 'source'])
        self.assertEqual([(r['body'], r['label']) for r in rows], [('click, now', '1.0'), ('hello\nfriend', '0.0')])
        self.assertEqual(rows[0]['subject'], 'hello')
        self.assertEqual(rows[0]['urls'], '')
        with self.assertRaises(FileExistsError):
            format_csv(self.source, self.output, **self.kwargs)

    def test_rejects_invalid_without_writing_output(self):
        self.source.write_text('message,category\nhello,phish\n')
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            format_csv(self.source, self.output, **self.kwargs)
        self.assertFalse(self.output.exists())
        self.source.write_text('message,category\nhello,1.0\n,0.0\n')
        with self.assertRaisesRegex(ValueError, 'Empty text'):
            format_csv(self.source, self.output, **self.kwargs)
        self.assertFalse(self.output.exists())

    def test_unlabeled_test(self):
        self.source.write_text('message\nhello\n')
        self.kwargs['mode'] = 'test'
        format_csv(self.source, self.output, unlabeled=True, **self.kwargs)
        with self.output.open(newline='') as f:
            row = next(csv.DictReader(f))
        self.assertEqual(row['body'], 'hello')
        self.assertNotIn('label', row)


if __name__ == '__main__':
    unittest.main()
