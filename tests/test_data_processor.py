"""
Tests for the DataProcessor class
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_service.data_processor import DataProcessor
from config import config

class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = config
        self.processor = DataProcessor(self.config)
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=100)
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Item Name': ['Batteries'] * 100,
            'Sales': np.random.randint(50, 150, size=100),
            'Price': np.random.uniform(10, 50, size=100),
            'Total Stock': np.random.randint(200, 500, size=100),
            'Stock Left': np.random.randint(50, 200, size=100),
            'Revenue': np.random.uniform(500, 2000, size=100)
        })
        
        # Add date components
        self.test_data['Year'] = self.test_data['Date'].dt.year
        self.test_data['Month'] = self.test_data['Date'].dt.month
        self.test_data['Day'] = self.test_data['Date'].dt.day
        self.test_data['dayofweek'] = self.test_data['Date'].dt.dayofweek
        self.test_data['quarter'] = self.test_data['Date'].dt.quarter
        self.test_data['is_weekend'] = self.test_data['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    def test_engineer_features(self):
        """Test engineer_features method"""
        # Engineer features
        processed_data = self.processor.engineer_features(self.test_data)
        
        # Check that features are created correctly
        self.assertTrue('stock_ratio' in processed_data.columns)
        self.assertTrue('price_bins' in processed_data.columns)
        self.assertTrue('sales_ratio' in processed_data.columns)
        self.assertTrue('price_stock_ratio' in processed_data.columns)
        self.assertTrue('sales_price_ratio' in processed_data.columns)
        self.assertTrue('sales_lag1' in processed_data.columns)
        self.assertTrue('sales_lag7' in processed_data.columns)
        self.assertTrue('sales_ma_7d' in processed_data.columns)
        self.assertTrue('sales_ma_30d' in processed_data.columns)
        
        # Check that there are no missing values
        self.assertEqual(processed_data.isnull().sum().sum(), 0)
    
    def test_prepare_train_test_data(self):
        """Test prepare_train_test_data method"""
        # Engineer features first
        processed_data = self.processor.engineer_features(self.test_data)
        
        # Prepare train-test data
        X_train, X_test, y_train, y_test = self.processor.prepare_train_test_data(processed_data)
        
        # Check shapes
        self.assertEqual(len(X_train) + len(X_test), len(processed_data))
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_test), len(X_test))
        
        # Check that features are correct
        for feature in self.config.FEATURES:
            self.assertTrue(feature in X_train.columns)
            self.assertTrue(feature in X_test.columns)
    
    def test_prepare_time_series_data(self):
        """Test prepare_time_series_data method"""
        # Engineer features first
        processed_data = self.processor.engineer_features(self.test_data)
        
        # Prepare time series data
        train_data, test_data = self.processor.prepare_time_series_data(processed_data)
        
        # Check shapes
        self.assertEqual(len(train_data) + len(test_data), len(processed_data.groupby('Date').size()))
        
        # Check that features are correct
        self.assertTrue('Sales' in train_data.columns)
        self.assertTrue('Revenue' in train_data.columns)
        self.assertTrue('dayofweek' in train_data.columns)
        self.assertTrue('month' in train_data.columns)
        self.assertTrue('quarter' in train_data.columns)
        self.assertTrue('is_weekend' in train_data.columns)
        self.assertTrue('stock_ratio' in train_data.columns)
        self.assertTrue('price_bins' in train_data.columns)
        self.assertTrue('sales_ratio' in train_data.columns)
        self.assertTrue('sales_lag1' in train_data.columns)
        self.assertTrue('sales_lag7' in train_data.columns)
        self.assertTrue('sales_ma_7d' in train_data.columns)
        self.assertTrue('sales_ma_30d' in train_data.columns)
        self.assertTrue('price_stock_ratio' in train_data.columns)
        self.assertTrue('sales_price_ratio' in train_data.columns)

if __name__ == '__main__':
    unittest.main()
