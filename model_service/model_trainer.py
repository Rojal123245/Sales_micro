"""
Model Trainer module for training sales forecasting models
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Class for training sales forecasting models
    """

    def __init__(self, config):
        """
        Initialize the ModelTrainer with configuration

        Args:
            config: Configuration object containing parameters
        """
        self.config = config
        logger.info("ModelTrainer initialized")

    def grid_search_arima(self, data, p_values, d_values, q_values):
        """
        Perform grid search to find the best ARIMA parameters

        Args:
            data (pandas.Series): Time series data
            p_values (list): List of p values to try
            d_values (list): List of d values to try
            q_values (list): List of q values to try

        Returns:
            tuple: (best_order, best_model)
        """
        try:
            logger.info("Starting ARIMA grid search")
            best_aic = float('inf')
            best_order = None
            best_model = None

            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            model = ARIMA(data, order=(p, d, q))
                            model_fit = model.fit()
                            aic = model_fit.aic

                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                                best_model = model_fit

                            logger.debug(f"ARIMA({p},{d},{q}) AIC: {aic}")
                        except Exception as e:
                            logger.debug(f"ARIMA({p},{d},{q}) failed: {str(e)}")
                            continue

            logger.info(f"Best ARIMA order: {best_order} with AIC: {best_aic}")
            return best_order, best_model
        except Exception as e:
            logger.error(f"Error in ARIMA grid search: {str(e)}")
            raise

    def train_arima_model(self, train_data, exog_variables=None):
        """
        Train an ARIMA model

        Args:
            train_data (pandas.Series): Training data
            exog_variables (pandas.DataFrame, optional): Exogenous variables

        Returns:
            statsmodels.tsa.arima.model.ARIMAResults: Trained model
        """
        try:
            logger.info("Training ARIMA model")

            # Use predefined order or perform grid search
            if hasattr(self.config, 'ARIMA_ORDER'):
                order = self.config.ARIMA_ORDER
                logger.info(f"Using predefined ARIMA order: {order}")
            else:
                # Default grid search parameters
                p_values = range(0, 3)
                d_values = range(0, 2)
                q_values = range(0, 3)
                order, _ = self.grid_search_arima(train_data, p_values, d_values, q_values)

            # Train the model
            try:
                # First try without trend
                model = ARIMA(endog=train_data, exog=exog_variables, order=order, trend='n')
                model_fit = model.fit()
            except Exception as e:
                logger.warning(f"Error training ARIMA with no trend: {str(e)}")
                # If that fails, try with default trend
                model = ARIMA(endog=train_data, exog=exog_variables, order=order)
                model_fit = model.fit()

            logger.info("ARIMA model trained successfully")
            return model_fit
        except Exception as e:
            logger.error(f"Error training ARIMA model: {str(e)}")
            raise

    def evaluate_model(self, model, test_data, exog_variables=None):
        """
        Evaluate the trained model

        Args:
            model: Trained model
            test_data (pandas.Series): Test data
            exog_variables (pandas.DataFrame, optional): Exogenous variables

        Returns:
            dict: Evaluation metrics
        """
        try:
            logger.info("Evaluating model")

            # Generate forecasts
            forecast = model.forecast(steps=len(test_data), exog=exog_variables)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(test_data, forecast))
            mae = mean_absolute_error(test_data, forecast)

            metrics = {
                'rmse': rmse,
                'mae': mae
            }

            logger.info(f"Model evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}")
            return metrics, forecast
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def save_model(self, model, path):
        """
        Save the trained model to disk

        Args:
            model: Trained model
            path (str): Path to save the model
        """
        try:
            logger.info(f"Saving model to {path}")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save the model
            with open(path, 'wb') as f:
                pickle.dump(model, f)

            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path):
        """
        Load a trained model from disk

        Args:
            path (str): Path to the saved model

        Returns:
            object: Loaded model
        """
        try:
            logger.info(f"Loading model from {path}")

            if not os.path.exists(path):
                logger.error(f"Model file not found at {path}")
                raise FileNotFoundError(f"Model file not found at {path}")

            # Load the model
            with open(path, 'rb') as f:
                model = pickle.load(f)

            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
