import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import auto_commit as a


class AutoCommitTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='.auto-commit-test-', dir=a.ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.git('init', '-b', 'main')
        self.git('config', 'user.name', 'Test')
        self.git('config', 'user.email', 'test@example.invalid')
        (self.root / 'example.txt').write_text('original')
        self.git('add', 'example.txt')
        self.git('commit', '-m', 'initial')
        self.root_patch = patch.object(a, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.root), *args], check=True, text=True, capture_output=True)

    def test_commit_and_preserve_user_staging(self):
        worker = a.AutoCommitter(push=False)
        (self.root / 'example.txt').write_text('updated')
        worker.sync()
        self.assertEqual(self.git('show', 'HEAD:example.txt').stdout, 'updated')
        (self.root / 'example.txt').write_text('staged')
        self.git('add', 'example.txt')
        with self.assertRaisesRegex(RuntimeError, 'Staged changes'): worker.sync()
        self.assertEqual(self.git('show', ':example.txt').stdout, 'staged')

    def test_retry_failed_commit(self):
        worker = a.AutoCommitter(push=False)
        hook = self.root / '.git' / 'hooks' / 'pre-commit'
        hook.write_text('#!/bin/sh\nexit 1\n')
        hook.chmod(0o755)
        (self.root / 'example.txt').write_text('retry')
        with self.assertRaises(subprocess.CalledProcessError): worker.sync()
        hook.unlink()
        worker.sync()
        self.assertEqual(self.git('show', 'HEAD:example.txt').stdout, 'retry')

    def test_changed_staging_after_failure_is_not_committed(self):
        worker = a.AutoCommitter(push=False)
        hook = self.root / '.git' / 'hooks' / 'pre-commit'
        hook.write_text('#!/bin/sh\nexit 1\n')
        hook.chmod(0o755)
        (self.root / 'example.txt').write_text('retry')
        with self.assertRaises(subprocess.CalledProcessError): worker.sync()
        (self.root / 'other.txt').write_text('manual')
        self.git('add', 'other.txt')
        with self.assertRaisesRegex(RuntimeError, 'Staged changes'): worker.sync()

    def test_multiple_instances_are_rejected(self):
        with a.instance_lock():
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                with a.instance_lock(): pass


if __name__ == '__main__': unittest.main()
