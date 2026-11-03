#!/usr/bin/env python3
"""Periodically commit changed project files and push the current branch."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def git(*args, check=True):
    env = os.environ.copy()
    env['GIT_TERMINAL_PROMPT'] = '0'
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True,
                          check=check, env=env, timeout=60)


def index_fingerprint():
    diff = git('diff', '--cached', '--binary', '--no-ext-diff').stdout
    return hashlib.sha256(diff.encode('utf-8')).hexdigest()


@contextmanager
def instance_lock():
    """Prevent concurrent instances, including --once, in this checkout."""
    git_dir = Path(git('rev-parse', '--git-dir').stdout.strip())
    if not git_dir.is_absolute():
        git_dir = ROOT / git_dir
    with (git_dir / 'auto-commit.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another auto-committer is already running in this checkout') from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class AutoCommitter:
    def __init__(self, push=True):
        self.push = push
        self.pending_index = None

    def sync(self):
        root = Path(git('rev-parse', '--show-toplevel').stdout.strip()).resolve()
        if root != ROOT:
            raise RuntimeError('The project directory must be the Git repository root')
        branch = git('symbolic-ref', '--quiet', '--short', 'HEAD', check=False)
        if branch.returncode:
            raise RuntimeError('A checked-out branch is required')
        staged = git('diff', '--cached', '--quiet', '--exit-code', check=False)
        if staged.returncode not in (0, 1):
            raise RuntimeError('Unable to inspect staged changes')
        if staged.returncode == 1:
            if self.pending_index is None or index_fingerprint() != self.pending_index:
                raise RuntimeError('Staged changes already exist; commit or unstage them before using auto-commit')
        else:
            self.pending_index = None
            git('add', '-A', '--', '.')
            staged = git('diff', '--cached', '--quiet', '--exit-code', check=False)
            if staged.returncode not in (0, 1):
                raise RuntimeError('Unable to inspect staged changes')
            if staged.returncode == 1:
                self.pending_index = index_fingerprint()
        if staged.returncode == 1:
            stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            git('commit', '-m', f'Auto-commit {stamp}')
            self.pending_index = None
            print(f'Committed changes on {branch.stdout.strip()}.', flush=True)
        if self.push:
            upstream = git('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}', check=False)
            if upstream.returncode:
                git('push', '--set-upstream', 'origin', 'HEAD')
                print('Pushed and configured the upstream branch.', flush=True)
            else:
                ahead = int(git('rev-list', '--count', '@{upstream}..HEAD').stdout.strip())
                if ahead:
                    git('push')
                    print(f'Pushed to {upstream.stdout.strip()}.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=float, default=2, help='seconds between checks (default: 2)')
    parser.add_argument('--once', action='store_true', help='check once and exit')
    parser.add_argument('--no-push', action='store_true', help='commit locally without pushing')
    args = parser.parse_args()
    if not 0 < args.interval < float('inf'):
        parser.error('--interval must be a positive finite number')
    try:
        with instance_lock():
            worker = AutoCommitter(push=not args.no_push)
            while True:
                try:
                    worker.sync()
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, RuntimeError, ValueError) as exc:
                    detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
                    print(f'Auto-commit failed: {detail}', file=sys.stderr, flush=True)
                    if args.once:
                        return 1
                if args.once:
                    return 0
                time.sleep(args.interval)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, RuntimeError) as exc:
        print(f'Auto-commit failed: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nAuto-committer stopped.')
