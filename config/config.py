"""
Configuration settings for the Sales Automation Microservice
"""

import os

# Data paths
DATA_DIR = os.environ.get('DATA_DIR', 'data')
DATA_FILE = os.environ.get('DATA_FILE', 'retail_store_data_with_real_items.csv')
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

# Model parameters
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42
ARIMA_ORDER = (2, 1, 2)

# API settings
API_HOST = os.environ.get('API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('API_PORT', 8080))
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'True').lower() == 'true'

# Feature engineering settings
FEATURES = [
    'dayofweek', 'Month', 'quarter', 'is_weekend', 'stock_ratio',
    'price_bins', 'sales_ratio', 'sales_lag1', 'sales_lag7',
    'sales_ma_7d', 'sales_ma_30d', 'price_stock_ratio', 'sales_price_ratio'
]

# Visualization settings
PLOT_STYLE = 'ggplot'
FIGURE_SIZE = (12, 6)
