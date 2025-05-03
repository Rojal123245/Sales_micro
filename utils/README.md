# Utilities

This directory contains utility functions and helpers for the Sales Automation Microservice.

## Components

### helpers.py

The `helpers.py` module provides utility functions for:

- Setting up logging
- Creating directories
- Generating date ranges
- Calculating evaluation metrics
- Formatting metrics for display
- Exporting data to CSV

## Usage

```python
from utils.helpers import setup_logging, create_directory_if_not_exists, calculate_metrics

# Set up logging
setup_logging(log_level=logging.INFO, log_file='logs/app.log')

# Create directory
create_directory_if_not_exists('output/plots')

# Calculate metrics
metrics = calculate_metrics(actual_values, predicted_values)
```
