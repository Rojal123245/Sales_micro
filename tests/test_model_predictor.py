"""
Tests for the ModelPredictor class
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_service.model_predictor import ModelPredictor
from config import config

class TestModelPredictor(unittest.TestCase):
    """Test cases for ModelPredictor"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = config
        self.predictor = ModelPredictor(self.config)
        
        # Create a mock model
        class MockModel:
            def forecast(self, steps, exog=None):
                return np.random.randint(50, 150, size=steps)
        
        self.mock_model = MockModel()
        self.predictor.model = self.mock_model
        
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
        
        # Add other features
        self.test_data['stock_ratio'] = self.test_data['Stock Left'] / self.test_data['Total Stock']
        self.test_data['price_bins'] = pd.qcut(self.test_data['Price'], q=5, labels=False)
        self.test_data['sales_ratio'] = self.test_data['Sales'] / self.test_data['Total Stock']
        self.test_data['price_stock_ratio'] = self.test_data['Price'] / self.test_data['Total Stock']
        self.test_data['sales_price_ratio'] = self.test_data['Sales'] / self.test_data['Price']
        
        # Add lag features
        self.test_data['sales_lag1'] = self.test_data['Sales'].shift(1)
        self.test_data['sales_lag7'] = self.test_data['Sales'].shift(7)
        self.test_data['sales_ma_7d'] = self.test_data['Sales'].rolling(window=7).mean()
        self.test_data['sales_ma_30d'] = self.test_data['Sales'].rolling(window=30).mean()
        
        # Fill missing values
        self.test_data = self.test_data.fillna(method='bfill')
    
    def test_prepare_forecast_features(self):
        """Test prepare_forecast_features method"""
        start_date = datetime.now()
        periods = 30
        
        # Test without historical data
        features = self.predictor.prepare_forecast_features(start_date, periods)
        self.assertEqual(len(features), periods)
        self.assertTrue('Date' in features.columns)
        self.assertTrue('dayofweek' in features.columns)
        self.assertTrue('month' in features.columns)
        
        # Test with historical data
        features = self.predictor.prepare_forecast_features(start_date, periods, self.test_data)
        self.assertEqual(len(features), periods)
        self.assertTrue('sales_lag1' in features.columns)
        self.assertTrue('sales_lag7' in features.columns)
    
    def test_forecast(self):
        """Test forecast method"""
        periods = 30
        start_date = datetime.now()
        
        # Test forecast without historical data
        forecast = self.predictor.forecast(periods, start_date)
        self.assertEqual(len(forecast), periods)
        self.assertTrue('Date' in forecast.columns)
        self.assertTrue('Forecast' in forecast.columns)
        
        # Test forecast with historical data
        forecast = self.predictor.forecast(periods, start_date, self.test_data)
        self.assertEqual(len(forecast), periods)
        self.assertTrue('Date' in forecast.columns)
        self.assertTrue('Forecast' in forecast.columns)
    
    def test_forecast_by_item(self):
        """Test forecast_by_item method"""
        periods = 30
        start_date = datetime.now()
        item_name = 'Batteries'
        
        # Test forecast by item
        forecast = self.predictor.forecast_by_item(item_name, periods, start_date, self.test_data)
        self.assertEqual(len(forecast), periods)
        self.assertTrue('Date' in forecast.columns)
        self.assertTrue('Forecast' in forecast.columns)
        self.assertTrue('Item Name' in forecast.columns)
        self.assertEqual(forecast['Item Name'].iloc[0], item_name)
        
        # Test with invalid item name
        with self.assertRaises(ValueError):
            self.predictor.forecast_by_item('NonExistentItem', periods, start_date, self.test_data)
    
    def test_load_model(self):
        """Test load_model method"""
        # Create a temporary model file
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'test_model.pkl')
        
        # Save the mock model
        with open(model_path, 'wb') as f:
            pickle.dump(self.mock_model, f)
        
        # Test loading the model
        self.predictor.load_model(model_path)
        self.assertIsNotNone(self.predictor.model)
        
        # Test loading non-existent model
        with self.assertRaises(FileNotFoundError):
            self.predictor.load_model('non_existent_model.pkl')
        
        # Clean up
        os.remove(model_path)

if __name__ == '__main__':
    unittest.main()
