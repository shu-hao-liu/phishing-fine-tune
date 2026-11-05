#!/usr/bin/env python3
"""Test a model on data/test.csv using the shared configuration."""
import sys
from pathlib import Path

# Resolve symlinks so /workspace/test.py can point to the project entry point.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune import main

if __name__ == '__main__':
    sys.exit(main('test'))
