"""
Tests for the Visualizer class
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization_service.visualizer import Visualizer
from config import config

class TestVisualizer(unittest.TestCase):
    """Test cases for Visualizer"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = config
        self.visualizer = Visualizer(self.config)
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=100)
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Item Name': ['Batteries'] * 50 + ['Soap'] * 50,
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
        
        # Create forecast data
        forecast_dates = pd.date_range(start='2023-04-10', periods=30)
        self.forecast_data = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': np.random.randint(50, 150, size=30)
        })
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove output directory if it exists
        output_dir = self.visualizer.output_dir
        if os.path.exists(output_dir):
            # Remove all files in the directory
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    
    def test_plot_sales_trend(self):
        """Test plot_sales_trend method"""
        # Generate plot
        plot_path = self.visualizer.plot_sales_trend(self.test_data)
        
        # Check that plot file exists
        self.assertTrue(os.path.exists(plot_path))
        
        # Check file extension
        self.assertTrue(plot_path.endswith('.png'))
    
    def test_plot_item_comparison(self):
        """Test plot_item_comparison method"""
        # Generate plot
        plot_path = self.visualizer.plot_item_comparison(self.test_data, ['Batteries', 'Soap'])
        
        # Check that plot file exists
        self.assertTrue(os.path.exists(plot_path))
        
        # Check file extension
        self.assertTrue(plot_path.endswith('.png'))
        
        # Test with invalid item
        with self.assertRaises(ValueError):
            self.visualizer.plot_item_comparison(self.test_data, ['NonExistentItem'])
    
    def test_plot_forecast(self):
        """Test plot_forecast method"""
        # Generate plot without item name
        plot_path = self.visualizer.plot_forecast(self.test_data, self.forecast_data)
        
        # Check that plot file exists
        self.assertTrue(os.path.exists(plot_path))
        
        # Check file extension
        self.assertTrue(plot_path.endswith('.png'))
        
        # Generate plot with item name
        plot_path = self.visualizer.plot_forecast(self.test_data, self.forecast_data, 'Batteries')
        
        # Check that plot file exists
        self.assertTrue(os.path.exists(plot_path))
        
        # Check file extension
        self.assertTrue(plot_path.endswith('.png'))
    
    def test_plot_sales_by_category(self):
        """Test plot_sales_by_category method"""
        # Generate plot
        plot_path = self.visualizer.plot_sales_by_category(self.test_data, 'Item Name')
        
        # Check that plot file exists
        self.assertTrue(os.path.exists(plot_path))
        
        # Check file extension
        self.assertTrue(plot_path.endswith('.png'))
    
    def test_plot_seasonal_patterns(self):
        """Test plot_seasonal_patterns method"""
        # Generate plot
        plot_path = self.visualizer.plot_seasonal_patterns(self.test_data)
        
        # Check that plot file exists
        self.assertTrue(os.path.exists(plot_path))
        
        # Check file extension
        self.assertTrue(plot_path.endswith('.png'))

if __name__ == '__main__':
    unittest.main()
