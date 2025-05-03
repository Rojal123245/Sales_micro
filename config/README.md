# Configuration

This directory contains configuration settings for the Sales Automation Microservice.

## Components

### config.py

The `config.py` module provides configuration settings for:

- Data paths and file names
- Model parameters
- API settings
- Feature engineering settings
- Visualization settings

## Usage

```python
from config import config

# Access configuration settings
data_path = config.DATA_PATH
train_test_split = config.TRAIN_TEST_SPLIT
api_port = config.API_PORT
features = config.FEATURES
```

## Environment Variables

The following environment variables can be used to override default settings:

- `DATA_DIR`: Directory containing data files
- `DATA_FILE`: Name of the data file
- `API_HOST`: Host for the API server
- `API_PORT`: Port for the API server
- `DEBUG_MODE`: Whether to run the API server in debug mode
