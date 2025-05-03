"""
Helper functions for Sales Automation Microservice
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level
        log_file: Path to log file (optional)
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            filename=log_file,
            filemode='a'
        )
    else:
        logging.basicConfig(
            level=log_level,
            format=log_format
        )
    
    # Reduce verbosity of third-party libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('pandas').setLevel(logging.WARNING)
    logging.getLogger('statsmodels').setLevel(logging.WARNING)
    
    logger.info("Logging configured")

def create_directory_if_not_exists(directory_path):
    """
    Create directory if it doesn't exist
    
    Args:
        directory_path: Path to directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def generate_date_range(start_date, periods):
    """
    Generate a date range
    
    Args:
        start_date: Start date
        periods: Number of periods
        
    Returns:
        list: List of dates
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    return [start_date + timedelta(days=i) for i in range(periods)]

def calculate_metrics(actual, predicted):
    """
    Calculate evaluation metrics
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        dict: Dictionary of metrics
    """
    # Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # Mean Squared Error
    mse = np.mean((actual - predicted) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # R-squared
    ss_total = np.sum((actual - np.mean(actual)) ** 2)
    ss_residual = np.sum((actual - predicted) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

def format_metrics(metrics):
    """
    Format metrics for display
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        str: Formatted metrics string
    """
    return (
        f"MAE: {metrics['mae']:.2f}\n"
        f"MSE: {metrics['mse']:.2f}\n"
        f"RMSE: {metrics['rmse']:.2f}\n"
        f"MAPE: {metrics['mape']:.2f}%\n"
        f"R²: {metrics['r2']:.4f}"
    )

def export_to_csv(data, filename, directory=None):
    """
    Export data to CSV file
    
    Args:
        data: Data to export
        filename: Filename
        directory: Directory (optional)
        
    Returns:
        str: Path to saved file
    """
    if directory:
        create_directory_if_not_exists(directory)
        file_path = os.path.join(directory, filename)
    else:
        file_path = filename
    
    data.to_csv(file_path, index=False)
    logger.info(f"Data exported to {file_path}")
    
    return file_path
