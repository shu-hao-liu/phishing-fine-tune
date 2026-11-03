#!/usr/bin/env python3
"""Classify the inference CSV using the shared configuration."""
import sys
from finetune import main

if __name__ == '__main__':
    sys.exit(main('predict'))
