"""
Tests for the DataLoader class
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_service.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a test data directory
        self.test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_data')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a test CSV file
        self.test_file = os.path.join(self.test_dir, 'test_data.csv')
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=100)
        test_data = pd.DataFrame({
            'Date': dates,
            'Item Name': ['Batteries'] * 100,
            'Sales': np.random.randint(50, 150, size=100),
            'Price': np.random.uniform(10, 50, size=100),
            'Total Stock': np.random.randint(200, 500, size=100),
            'Stock Left': np.random.randint(50, 200, size=100),
            'Revenue': np.random.uniform(500, 2000, size=100)
        })
        
        # Save test data to CSV
        test_data.to_csv(self.test_file, index=False)
        
        # Initialize DataLoader with test file
        self.data_loader = DataLoader(self.test_file)
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove test file and directory
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    def test_load_data(self):
        """Test load_data method"""
        # Load data
        data = self.data_loader.load_data()
        
        # Check that data is loaded correctly
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 100)
        self.assertTrue('Date' in data.columns)
        self.assertTrue('Item Name' in data.columns)
        self.assertTrue('Sales' in data.columns)
    
    def test_preprocess_dates(self):
        """Test preprocess_dates method"""
        # Load data
        data = self.data_loader.load_data()
        
        # Preprocess dates
        processed_data = self.data_loader.preprocess_dates(data)
        
        # Check that date components are extracted correctly
        self.assertTrue('Year' in processed_data.columns)
        self.assertTrue('Month' in processed_data.columns)
        self.assertTrue('Day' in processed_data.columns)
        self.assertTrue('dayofweek' in processed_data.columns)
        self.assertTrue('quarter' in processed_data.columns)
        self.assertTrue('is_weekend' in processed_data.columns)
        
        # Check values
        self.assertEqual(processed_data['Year'].iloc[0], 2023)
        self.assertEqual(processed_data['Month'].iloc[0], 1)
        self.assertEqual(processed_data['Day'].iloc[0], 1)
    
    def test_get_data(self):
        """Test get_data method"""
        # Get data
        data = self.data_loader.get_data()
        
        # Check that data is loaded and preprocessed correctly
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 100)
        self.assertTrue('Year' in data.columns)
        self.assertTrue('Month' in data.columns)
        self.assertTrue('Day' in data.columns)
    
    def test_file_not_found(self):
        """Test handling of non-existent file"""
        # Initialize DataLoader with non-existent file
        data_loader = DataLoader('non_existent_file.csv')
        
        # Check that FileNotFoundError is raised
        with self.assertRaises(FileNotFoundError):
            data_loader.load_data()

if __name__ == '__main__':
    unittest.main()
