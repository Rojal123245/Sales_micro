#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run tests
echo "Running tests..."
python -m unittest discover -s tests
