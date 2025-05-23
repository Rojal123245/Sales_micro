"""
Data Processor module for feature engineering and data preparation
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class for feature engineering and data preparation
    """

    def __init__(self, config):
        """
        Initialize the DataProcessor with configuration

        Args:
            config: Configuration object containing parameters
        """
        self.config = config
        self.scaler = StandardScaler()
        logger.info("DataProcessor initialized")

    def engineer_features(self, data):
        """
        Create additional features for the model

        Args:
            data (pandas.DataFrame): Input data

        Returns:
            pandas.DataFrame: Data with engineered features
        """
        try:
            logger.info("Engineering features")

            data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
            data['Total Stock'] = pd.to_numeric(data['Total Stock'], errors='coerce')
            data['Sales'] = pd.to_numeric(data['Sales'], errors='coerce')
            data['Stock Left'] = pd.to_numeric(data['Stock Left'], errors='coerce')
            data['Revenue'] = pd.to_numeric(data['Revenue'], errors='coerce')

            # Calculate stock ratio
            data['stock_ratio'] = data['Stock Left'] / data['Total Stock']
            data['stock_ratio'] = data['stock_ratio'].fillna(0)

            # Create price bins
            try:
                data['price_bins'] = pd.qcut(data['Price'], q=5, labels=False)
            except ValueError:
                # Handle case with duplicate values
                logger.info("Using quantile bins with duplicates='drop'")
                data['price_bins'] = pd.qcut(data['Price'], q=5, labels=False, duplicates='drop')
            data['price_bins'] = pd.to_numeric(data['price_bins'], errors='coerce').fillna(0)

            # Calculate sales ratio
            data['sales_ratio'] = data['Sales'] / data['Total Stock']
            data['sales_ratio'] = data['sales_ratio'].fillna(0)

            # Calculate price-stock ratio
            data['price_stock_ratio'] = data['Price'] * data['stock_ratio']
            data['price_stock_ratio'] = data['price_stock_ratio'].fillna(0)

            # Calculate sales-price ratio
            data['sales_price_ratio'] = data['Sales'] * data['Price']
            data['sales_price_ratio'] = data['sales_price_ratio'].fillna(0)

            # Rolling features (grouped by itemname) -- moving average
            for window in [7, 30]:
                data[f'sales_ma_{window}d'] = data.groupby('Item Name')['Sales'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean())
                data[f'revenue_ma_{window}d'] = data.groupby('Item Name')['Revenue'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean())
                data[f'stock_ma_{window}d'] = data.groupby('Item Name')['Stock Left'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean())

            data['sales_lag1'] = data.groupby('Item Name')['Sales'].shift(1)
            data['sales_lag7'] = data.groupby('Item Name')['Sales'].shift(7)

            # Fill missing values
            data = data.fillna(method='bfill')

            logger.info("Features engineered successfully")
            return data
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            raise

    def prepare_train_test_data(self, data):
        """
        Split data into training and testing sets

        Args:
            data (pandas.DataFrame): Input data

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            logger.info("Preparing train-test data")

            # Select features and target
            features = self.config.FEATURES
            X = data[features]
            y = data['Sales']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                train_size=self.config.TRAIN_TEST_SPLIT,
                random_state=self.config.RANDOM_STATE
            )

            logger.info(f"Data split into train ({len(X_train)} rows) and test ({len(X_test)} rows)")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error preparing train-test data: {str(e)}")
            raise

    def prepare_time_series_data(self, data):
        """
        Prepare data for time series analysis

        Args:
            data (pandas.DataFrame): Input data

        Returns:
            tuple: (train_data, test_data) with both target and exogenous variables
        """
        try:
            logger.info("Preparing time series data")

            # Sort by date
            data = data.sort_values('Date')

            daily_data = data.copy()

            # Group by date for aggregated metrics
            # daily_data = data.groupby('Date').agg({
            #     'Sales': 'sum',
            #     'Revenue': 'sum',
            #     'Total Stock': 'sum',
            #     'Stock Left': 'sum',
            #     'dayofweek': 'first',
            #     'Month': 'first',
            #     'quarter': 'first',
            #     'is_weekend': 'first',
            #     'sales_lag1': 'mean',
            #     'sales_lag7': 'mean',
            #     'sales_ma_7d': 'mean',
            #     'sales_ma_30d': 'mean',
            #     'stock_ratio': 'mean',
            #     'price_bins': 'mean',
            #     'sales_ratio': 'mean',
            #     'price_stock_ratio': 'mean',
            #     'sales_price_ratio': 'mean'
            # }).reset_index()

            # # Create additional time-based features
            # daily_data['date_of_month'] = daily_data['Date'].dt.day
            # daily_data['day_of_year'] = daily_data['Date'].dt.dayofyear

            # # Create seasonal features using sine and cosine transformations
            # daily_data['month_sin'] = np.sin(2 * np.pi * daily_data['Month']/12)
            # daily_data['month_cos'] = np.cos(2 * np.pi * daily_data['Month']/12)
            # daily_data['day_sin'] = np.sin(2 * np.pi * daily_data['date_of_month']/31)
            # daily_data['day_cos'] = np.cos(2 * np.pi * daily_data['date_of_month']/31)

            # # Fill any missing values
            # daily_data = daily_data.fillna(method='ffill').fillna(method='bfill').fillna(0)

            # # Ensure all numeric columns have proper data types
            # numeric_columns = daily_data.select_dtypes(include=['float', 'int']).columns
            # for col in numeric_columns:
            #     daily_data[col] = pd.to_numeric(daily_data[col], errors='coerce')
            #     daily_data[col] = daily_data[col].fillna(0)

            # Split into train and test
            train_size = int(len(daily_data) * self.config.TRAIN_TEST_SPLIT)
            train_data = daily_data[:train_size]
            test_data = daily_data[train_size:]

            logger.info(f"Time series data prepared with {len(train_data)} train and {len(test_data)} test samples")
            return train_data, test_data
        except Exception as e:
            logger.error(f"Error preparing time series data: {str(e)}")
            raise
