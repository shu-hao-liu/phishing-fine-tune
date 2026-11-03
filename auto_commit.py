#!/usr/bin/env python3
"""Periodically commit changed project files and push the current branch."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def git(*args, check=True):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, check=check)


def sync(push=True):
    root = Path(git('rev-parse', '--show-toplevel').stdout.strip()).resolve()
    if root != ROOT:
        raise RuntimeError('The project directory must be the Git repository root')
    branch = git('symbolic-ref', '--quiet', '--short', 'HEAD', check=False)
    if branch.returncode:
        raise RuntimeError('A checked-out branch is required')
    staged = git('diff', '--cached', '--quiet', '--exit-code', check=False)
    if staged.returncode == 1:
        raise RuntimeError('Staged changes already exist; commit or unstage them before using auto-commit')
    if staged.returncode != 0:
        raise RuntimeError('Unable to inspect staged changes')
    git('add', '-A', '--', '.')
    staged = git('diff', '--cached', '--quiet', '--exit-code', check=False)
    if staged.returncode == 1:
        stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        git('commit', '-m', f'Auto-commit {stamp}')
        print(f'Committed changes on {branch.stdout.strip()}.', flush=True)
    elif staged.returncode != 0:
        raise RuntimeError('Unable to inspect staged changes')
    if push:
        upstream = git('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}', check=False)
        if upstream.returncode:
            # A normal push never overwrites a non-fast-forward remote.
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
    while True:
        try:
            sync(push=not args.no_push)
        except (subprocess.CalledProcessError, OSError, RuntimeError, ValueError) as exc:
            detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
            print(f'Auto-commit failed: {detail}', file=sys.stderr, flush=True)
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nAuto-committer stopped.')
