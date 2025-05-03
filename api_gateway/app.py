"""
API Gateway for Sales Automation Microservice
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from data_service.data_loader import DataLoader
from data_service.data_processor import DataProcessor
from model_service.model_trainer import ModelTrainer
from model_service.model_predictor import ModelPredictor
from visualization_service.visualizer import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize services
data_loader = DataLoader(config.DATA_PATH)
data_processor = DataProcessor(config)
model_trainer = ModelTrainer(config)
model_predictor = ModelPredictor(config)
visualizer = Visualizer(config)

@app.route('/', methods=['GET'])
def index():
    """Home page"""
    return render_template('index.html')

# Load model if exists
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'arima_model.pkl')
if os.path.exists(model_path):
    try:
        model_predictor.load_model(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/data/load', methods=['GET'])
def load_data():
    """Load data endpoint"""
    try:
        data = data_loader.get_data()
        return jsonify({
            'status': 'success',
            'message': f'Data loaded successfully with {len(data)} rows',
            'sample': data.head(5).to_dict(orient='records')
        })
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/data/process', methods=['GET'])
def process_data():
    """Process data endpoint"""
    try:
        # Load data
        data = data_loader.get_data()

        # Engineer features
        processed_data = data_processor.engineer_features(data)

        return jsonify({
            'status': 'success',
            'message': f'Data processed successfully with {len(processed_data)} rows',
            'features': list(processed_data.columns),
            'sample': processed_data.head(5).to_dict(orient='records')
        })
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/model/train', methods=['POST'])
def train_model():
    """Train model endpoint"""
    try:
        # Get parameters from request
        params = request.get_json() or {}

        # Get ARIMA order from parameters or use default
        p = params.get('p', config.ARIMA_ORDER[0])
        d = params.get('d', config.ARIMA_ORDER[1])
        q = params.get('q', config.ARIMA_ORDER[2])
        arima_order = (p, d, q)

        logger.info(f"Training model with ARIMA order {arima_order}")

        # Load data
        data = data_loader.get_data()

        # Engineer features
        processed_data = data_processor.engineer_features(data)

        # Prepare time series data
        train_data, test_data = data_processor.prepare_time_series_data(processed_data)

        # Define exogenous variables for ARIMA model
        exog_features = [
            'dayofweek', 'Month', 'quarter', 'is_weekend',
            'stock_ratio', 'sales_ratio', 'sales_lag1', 'sales_lag7',
            'sales_ma_7d', 'sales_ma_30d', 'price_stock_ratio',
            'sales_price_ratio', 'month_sin', 'month_cos',
            'day_sin', 'day_cos'
        ]

        # Filter features that exist in the data
        available_features = [f for f in exog_features if f in train_data.columns]

        # Log the features being used
        logger.info(f"Using {len(available_features)} features for ARIMA model: {available_features}")

        # Temporarily override the ARIMA order in config
        original_order = config.ARIMA_ORDER
        config.ARIMA_ORDER = arima_order

        try:
            # Train model with exogenous variables
            model = model_trainer.train_arima_model(
                train_data['Sales'],
                train_data[available_features]
            )

            # Evaluate model with exogenous variables
            metrics, forecast = model_trainer.evaluate_model(
                model,
                test_data['Sales'],
                test_data[available_features]
            )

            # Save model
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'arima_model.pkl')
            model_trainer.save_model(model, model_path)

            # Update model predictor
            model_predictor.model = model

            # Restore original ARIMA order
            config.ARIMA_ORDER = original_order

            # Test the model with a simple forecast
            test_forecast = model_predictor.forecast(
                periods=7,
                historical_data=processed_data
            )

            # Check if forecast is all zeros
            if test_forecast['Forecast'].sum() == 0:
                logger.warning("Test forecast is all zeros, model may need adjustment")

            return jsonify({
                'status': 'success',
                'message': 'Model trained successfully',
                'metrics': metrics,
                'model_path': model_path,
                'arima_order': arima_order,
                'test_forecast': test_forecast.to_dict(orient='records')
            })
        except Exception as model_error:
            # Restore original ARIMA order
            config.ARIMA_ORDER = original_order
            raise model_error
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/model/forecast', methods=['POST'])
def forecast():
    """Generate forecast endpoint"""
    try:
        # Get parameters from request
        params = request.get_json() or {}
        periods = params.get('periods', 30)
        start_date = params.get('start_date', None)
        item_name = params.get('item_name', None)

        if start_date:
            start_date = datetime.fromisoformat(start_date)
        else:
            start_date = datetime.now()

        # Load data for historical context
        data = data_loader.get_data()
        processed_data = data_processor.engineer_features(data)

        # Generate forecast
        if item_name:
            forecast_result = model_predictor.forecast_by_item(
                item_name,
                periods,
                start_date,
                processed_data
            )
        else:
            forecast_result = model_predictor.forecast(
                periods,
                start_date,
                processed_data
            )

        # Convert to JSON-serializable format
        # Only include the requested number of periods
        forecast_dict = forecast_result.head(periods).to_dict(orient='records')

        return jsonify({
            'status': 'success',
            'message': f'Forecast generated successfully for {periods} periods',
            'forecast': forecast_dict
        })
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/visualization/sales_trend', methods=['GET'])
def sales_trend():
    """Generate sales trend visualization endpoint"""
    try:
        # Load data
        data = data_loader.get_data()

        # Engineer features
        processed_data = data_processor.engineer_features(data)

        # Generate visualization
        plot_path = visualizer.plot_sales_trend(processed_data)

        return jsonify({
            'status': 'success',
            'message': 'Sales trend visualization generated successfully',
            'plot_path': plot_path
        })
    except Exception as e:
        logger.error(f"Error generating sales trend visualization: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/visualization/item_comparison', methods=['POST'])
def item_comparison():
    """Generate item comparison visualization endpoint"""
    try:
        # Get parameters from request
        params = request.get_json() or {}
        items = params.get('items', [])

        if not items:
            return jsonify({
                'status': 'error',
                'message': 'No items specified for comparison'
            }), 400

        # Load data
        data = data_loader.get_data()

        # Generate visualization
        plot_path = visualizer.plot_item_comparison(data, items)

        return jsonify({
            'status': 'success',
            'message': 'Item comparison visualization generated successfully',
            'plot_path': plot_path
        })
    except Exception as e:
        logger.error(f"Error generating item comparison visualization: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/visualization/forecast_plot', methods=['POST'])
def forecast_plot():
    """Generate forecast visualization endpoint"""
    try:
        # Get parameters from request
        params = request.get_json() or {}
        periods = params.get('periods', 30)
        start_date = params.get('start_date', None)
        item_name = params.get('item_name', None)

        if start_date:
            start_date = datetime.fromisoformat(start_date)
        else:
            start_date = datetime.now()

        # Load data for historical context
        data = data_loader.get_data()
        processed_data = data_processor.engineer_features(data)

        # Generate forecast
        if item_name:
            forecast_result = model_predictor.forecast_by_item(
                item_name,
                periods,
                start_date,
                processed_data
            )
            historical_data = processed_data[processed_data['Item Name'] == item_name]
        else:
            forecast_result = model_predictor.forecast(
                periods,
                start_date,
                processed_data
            )
            historical_data = processed_data

        # Generate visualization
        plot_path = visualizer.plot_forecast(historical_data, forecast_result, item_name)

        return jsonify({
            'status': 'success',
            'message': 'Forecast visualization generated successfully',
            'plot_path': plot_path
        })
    except Exception as e:
        logger.error(f"Error generating forecast visualization: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG_MODE
    )
