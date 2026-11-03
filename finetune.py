#!/usr/bin/env python3
"""Fine-tune the model using the shared configuration."""
import sys
from phishing import main

if __name__ == '__main__':
    sys.exit(main('train'))
