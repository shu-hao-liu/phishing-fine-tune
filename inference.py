#!/usr/bin/env python3
"""Test a model on data/test.csv using the shared configuration."""
import sys
from finetune import main

if __name__ == '__main__':
    sys.exit(main('test'))
