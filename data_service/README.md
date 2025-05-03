# Data Service

This service is responsible for loading, preprocessing, and feature engineering of sales data.

## Components

### DataLoader

The `DataLoader` class provides functionality for:

- Loading data from CSV files
- Initial data preprocessing
- Date component extraction

### DataProcessor

The `DataProcessor` class provides functionality for:

- Feature engineering
- Train-test data splitting
- Time series data preparation

## Usage

```python
from data_service.data_loader import DataLoader
from data_service.data_processor import DataProcessor
from config import config

# Initialize services
data_loader = DataLoader(config.DATA_PATH)
data_processor = DataProcessor(config)

# Load data
data = data_loader.get_data()

# Engineer features
processed_data = data_processor.engineer_features(data)

# Prepare train-test data
X_train, X_test, y_train, y_test = data_processor.prepare_train_test_data(processed_data)

# Prepare time series data
train_data, test_data = data_processor.prepare_time_series_data(processed_data)
```
