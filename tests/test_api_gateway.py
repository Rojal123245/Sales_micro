"""
Tests for the API Gateway
"""

import os
import sys
import unittest
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_gateway.app import app

class TestAPIGateway(unittest.TestCase):
    """Test cases for API Gateway"""
    
    def setUp(self):
        """Set up test environment"""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.app.get('/health')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue('timestamp' in data)
    
    @patch('api_gateway.app.data_loader')
    def test_load_data(self, mock_data_loader):
        """Test load data endpoint"""
        # Mock the get_data method
        mock_data = MagicMock()
        mock_data.head.return_value.to_dict.return_value = [{'Date': '2023-01-01', 'Sales': 100}]
        mock_data.__len__.return_value = 100
        mock_data_loader.get_data.return_value = mock_data
        
        # Call the endpoint
        response = self.app.get('/data/load')
        data = json.loads(response.data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertTrue('sample' in data)
    
    @patch('api_gateway.app.data_loader')
    @patch('api_gateway.app.data_processor')
    def test_process_data(self, mock_data_processor, mock_data_loader):
        """Test process data endpoint"""
        # Mock the get_data method
        mock_data = MagicMock()
        mock_data_loader.get_data.return_value = mock_data
        
        # Mock the engineer_features method
        mock_processed_data = MagicMock()
        mock_processed_data.head.return_value.to_dict.return_value = [{'Date': '2023-01-01', 'Sales': 100}]
        mock_processed_data.__len__.return_value = 100
        mock_processed_data.columns = ['Date', 'Sales', 'Price']
        mock_data_processor.engineer_features.return_value = mock_processed_data
        
        # Call the endpoint
        response = self.app.get('/data/process')
        data = json.loads(response.data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertTrue('features' in data)
        self.assertTrue('sample' in data)
    
    @patch('api_gateway.app.data_loader')
    @patch('api_gateway.app.data_processor')
    @patch('api_gateway.app.model_trainer')
    def test_train_model(self, mock_model_trainer, mock_data_processor, mock_data_loader):
        """Test train model endpoint"""
        # Mock the get_data method
        mock_data = MagicMock()
        mock_data_loader.get_data.return_value = mock_data
        
        # Mock the engineer_features method
        mock_processed_data = MagicMock()
        mock_data_processor.engineer_features.return_value = mock_processed_data
        
        # Mock the prepare_time_series_data method
        mock_train_data = MagicMock()
        mock_test_data = MagicMock()
        mock_data_processor.prepare_time_series_data.return_value = (mock_train_data, mock_test_data)
        
        # Mock the train_arima_model method
        mock_model = MagicMock()
        mock_model_trainer.train_arima_model.return_value = mock_model
        
        # Mock the evaluate_model method
        mock_metrics = {'rmse': 10.5, 'mae': 8.2}
        mock_forecast = MagicMock()
        mock_model_trainer.evaluate_model.return_value = (mock_metrics, mock_forecast)
        
        # Mock the save_model method
        mock_model_trainer.save_model.return_value = None
        
        # Call the endpoint
        response = self.app.post('/model/train', json={})
        data = json.loads(response.data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertTrue('metrics' in data)
        self.assertTrue('model_path' in data)
    
    @patch('api_gateway.app.data_loader')
    @patch('api_gateway.app.data_processor')
    @patch('api_gateway.app.model_predictor')
    def test_forecast(self, mock_model_predictor, mock_data_processor, mock_data_loader):
        """Test forecast endpoint"""
        # Mock the get_data method
        mock_data = MagicMock()
        mock_data_loader.get_data.return_value = mock_data
        
        # Mock the engineer_features method
        mock_processed_data = MagicMock()
        mock_data_processor.engineer_features.return_value = mock_processed_data
        
        # Mock the forecast method
        mock_forecast_result = MagicMock()
        mock_forecast_result.to_dict.return_value = [{'Date': '2023-01-01', 'Forecast': 100}]
        mock_model_predictor.forecast.return_value = mock_forecast_result
        
        # Call the endpoint
        response = self.app.post('/model/forecast', json={'periods': 30})
        data = json.loads(response.data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertTrue('forecast' in data)
    
    @patch('api_gateway.app.data_loader')
    @patch('api_gateway.app.data_processor')
    @patch('api_gateway.app.visualizer')
    def test_sales_trend(self, mock_visualizer, mock_data_processor, mock_data_loader):
        """Test sales trend visualization endpoint"""
        # Mock the get_data method
        mock_data = MagicMock()
        mock_data_loader.get_data.return_value = mock_data
        
        # Mock the engineer_features method
        mock_processed_data = MagicMock()
        mock_data_processor.engineer_features.return_value = mock_processed_data
        
        # Mock the plot_sales_trend method
        mock_plot_path = '/path/to/plot.png'
        mock_visualizer.plot_sales_trend.return_value = mock_plot_path
        
        # Call the endpoint
        response = self.app.get('/visualization/sales_trend')
        data = json.loads(response.data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertTrue('plot_path' in data)
        self.assertEqual(data['plot_path'], mock_plot_path)

if __name__ == '__main__':
    unittest.main()
