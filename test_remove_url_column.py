import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import remove_url_column as tool


class RemoveUrlColumnTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='.auto-commit-test-', dir=Path(__file__).parent)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_backups_and_values_preserved_for_all_files(self):
        first, second = self.root / 'a.csv', self.root / 'b.csv'
        first.write_bytes(b'\xef\xbb\xbflabel,urls,body,source\r\n1.0,http://x,"hello,\r\nworld",x\r\n')
        second.write_text('urls,body,label\n,hello,0.0\n')
        originals = {p: p.read_bytes() for p in (first, second)}
        replace = tool.os.replace
        def verify_backups_exist(source, destination):
            for path in (first, second):
                backup = path.with_name(path.stem + '.before-url-removal.csv')
                self.assertEqual(backup.read_bytes(), originals[path])
            return replace(source, destination)
        with patch.object(tool.os, 'replace', side_effect=verify_backups_exist):
            plans, skipped = tool.remove_url_column([first, second])
        self.assertEqual(len(plans), 2)
        self.assertFalse(skipped)
        with first.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows, [['label', 'body', 'source'], ['1.0', 'hello,\r\nworld', 'x']])
        with second.open(encoding='utf-8-sig', newline='') as f:
            self.assertEqual(list(csv.reader(f)), [['body', 'label'], ['hello', '0.0']])
        self.assertTrue(first.read_bytes().startswith(b'\xef\xbb\xbf'))

    def test_existing_backup_and_bad_row_leave_source_untouched(self):
        source = self.root / 'source.csv'
        original = b'body,urls,label\nhello,http://x,1.0\n'
        source.write_bytes(original)
        backup = self.root / 'source.before-url-removal.csv'
        backup.write_bytes(b'do not replace')
        with self.assertRaises(FileExistsError):
            tool.remove_url_column([source])
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(backup.read_bytes(), b'do not replace')
        backup.unlink()
        source.write_bytes(b'body,urls,label\nmissing,columns\n')
        with self.assertRaisesRegex(ValueError, 'malformed row'):
            tool.remove_url_column([source])
        self.assertEqual(source.read_bytes(), b'body,urls,label\nmissing,columns\n')
        self.assertEqual(backup.read_bytes(), source.read_bytes())

    def test_remove_multiple_selected_columns_and_preserve_the_rest(self):
        source = self.root / 'source.csv'
        original = 'subject,body,label,language,source,extra\nTitle,"Hi, there",1.0,en,example,kept\n'
        source.write_text(original)
        plans, skipped = tool.remove_url_column([source], ('subject', 'language', 'source'))
        self.assertEqual(len(plans), 1)
        self.assertFalse(skipped)
        self.assertEqual((self.root / 'source.before-column-removal.csv').read_text(), original)
        with source.open(newline='') as f:
            self.assertEqual(list(csv.reader(f)), [['body', 'label', 'extra'], ['Hi, there', '1.0', 'kept']])
        with self.assertRaisesRegex(ValueError, 'every column'):
            tool.remove_url_column([source], ('body', 'label', 'extra'))

    def test_no_url_column_is_skipped(self):
        source = self.root / 'source.csv'
        source.write_text('content,label\nhello,1\n')
        plans, skipped = tool.remove_url_column([source])
        self.assertEqual(plans, [])
        self.assertEqual(skipped, [source.resolve()])
        self.assertFalse((self.root / 'source.before-url-removal.csv').exists())


if __name__ == '__main__':
    unittest.main()
