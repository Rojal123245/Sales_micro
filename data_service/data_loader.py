"""
Data Loader module for loading and initial processing of sales data
"""

import os
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class for loading and initial processing of sales data
    """
    
    def __init__(self, data_path):
        """
        Initialize the DataLoader with the path to the data file
        
        Args:
            data_path (str): Path to the data file
        """
        self.data_path = data_path
        logger.info(f"DataLoader initialized with data path: {data_path}")
    
    def load_data(self):
        """
        Load data from the specified path
        
        Returns:
            pandas.DataFrame: Loaded data
        """
        if not os.path.exists(self.data_path):
            logger.error(f"Data file not found at {self.data_path}")
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        try:
            logger.info(f"Loading data from {self.data_path}")
            data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully with {len(data)} rows")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_dates(self, data):
        """
        Convert date columns to datetime and extract date components
        
        Args:
            data (pandas.DataFrame): Input data
            
        Returns:
            pandas.DataFrame: Data with processed date columns
        """
        try:
            logger.info("Processing date columns")
            # Convert Date column to datetime
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data = data.dropna(subset=['Date'])
            
            # Extract date components
            data['Year'] = data['Date'].dt.year
            data['Month'] = data['Date'].dt.month
            data['Day'] = data['Date'].dt.day
            data['dayofweek'] = data['Date'].dt.dayofweek
            data['quarter'] = data['Date'].dt.quarter
            data['is_weekend'] = data['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
            
            logger.info("Date columns processed successfully")
            return data
        except Exception as e:
            logger.error(f"Error processing date columns: {str(e)}")
            raise
    
    def get_data(self):
        """
        Load and preprocess the data
        
        Returns:
            pandas.DataFrame: Loaded and preprocessed data
        """
        data = self.load_data()
        data = self.preprocess_dates(data)
        return data
