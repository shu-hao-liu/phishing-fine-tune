#!/usr/bin/env python3
"""Classify the inference CSV using the shared configuration."""
import sys
from phishing import main

if __name__ == '__main__':
    sys.exit(main('predict'))
