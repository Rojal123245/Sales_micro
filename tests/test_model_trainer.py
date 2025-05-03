"""
Tests for the ModelTrainer class
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_service.model_trainer import ModelTrainer
from config import config

class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = config
        self.trainer = ModelTrainer(self.config)
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=100)
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Sales': np.random.randint(50, 150, size=100)
        })
        
        # Create test features
        self.test_features = pd.DataFrame({
            'dayofweek': np.random.randint(0, 7, size=100),
            'month': np.random.randint(1, 13, size=100),
            'quarter': np.random.randint(1, 5, size=100),
            'is_weekend': np.random.randint(0, 2, size=100),
            'stock_ratio': np.random.uniform(0, 1, size=100),
            'price_bins': np.random.randint(0, 5, size=100),
            'sales_ratio': np.random.uniform(0, 1, size=100),
            'sales_lag1': np.random.randint(50, 150, size=100),
            'sales_lag7': np.random.randint(50, 150, size=100),
            'sales_ma_7d': np.random.uniform(50, 150, size=100),
            'sales_ma_30d': np.random.uniform(50, 150, size=100),
            'price_stock_ratio': np.random.uniform(0, 1, size=100),
            'sales_price_ratio': np.random.uniform(0, 1, size=100)
        })
    
    def test_grid_search_arima(self):
        """Test grid_search_arima method"""
        # Define parameter ranges
        p_values = range(0, 2)
        d_values = range(0, 2)
        q_values = range(0, 2)
        
        # Perform grid search
        best_order, best_model = self.trainer.grid_search_arima(
            self.test_data['Sales'],
            p_values,
            d_values,
            q_values
        )
        
        # Check that best_order is a tuple of 3 integers
        self.assertIsInstance(best_order, tuple)
        self.assertEqual(len(best_order), 3)
        self.assertIsInstance(best_order[0], int)
        self.assertIsInstance(best_order[1], int)
        self.assertIsInstance(best_order[2], int)
        
        # Check that best_model is not None
        self.assertIsNotNone(best_model)
    
    def test_train_arima_model(self):
        """Test train_arima_model method"""
        # Train model
        model = self.trainer.train_arima_model(
            self.test_data['Sales'],
            self.test_features
        )
        
        # Check that model is not None
        self.assertIsNotNone(model)
        
        # Check that model has forecast method
        self.assertTrue(hasattr(model, 'forecast'))
    
    def test_evaluate_model(self):
        """Test evaluate_model method"""
        # Train model
        model = self.trainer.train_arima_model(
            self.test_data['Sales'][:80],
            self.test_features[:80]
        )
        
        # Evaluate model
        metrics, forecast = self.trainer.evaluate_model(
            model,
            self.test_data['Sales'][80:],
            self.test_features[80:]
        )
        
        # Check that metrics is a dictionary
        self.assertIsInstance(metrics, dict)
        self.assertTrue('rmse' in metrics)
        self.assertTrue('mae' in metrics)
        
        # Check that forecast is an array-like object
        self.assertEqual(len(forecast), len(self.test_data['Sales'][80:]))
    
    def test_save_and_load_model(self):
        """Test save_model and load_model methods"""
        # Train model
        model = self.trainer.train_arima_model(
            self.test_data['Sales'],
            self.test_features
        )
        
        # Create a temporary model file
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'test_model.pkl')
        
        # Save model
        self.trainer.save_model(model, model_path)
        
        # Check that model file exists
        self.assertTrue(os.path.exists(model_path))
        
        # Load model
        loaded_model = self.trainer.load_model(model_path)
        
        # Check that loaded model is not None
        self.assertIsNotNone(loaded_model)
        
        # Check that loaded model has forecast method
        self.assertTrue(hasattr(loaded_model, 'forecast'))
        
        # Clean up
        os.remove(model_path)

if __name__ == '__main__':
    unittest.main()
