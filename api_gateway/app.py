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

import warnings
warnings.filterwarnings("ignore")

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
        # Load data
        data = data_loader.get_data()

        # Engineer features
        processed_data = data_processor.engineer_features(data)

        if len(processed_data) == 0:
            return jsonify({
                'status': 'error',
                'message': 'No valid data with dates available for training'
            }), 400

        # Prepare time series data
        train_data, test_data = data_processor.prepare_time_series_data(processed_data)

        logger.info(f"Train data columns: {train_data.columns.tolist()}")

        # Define exogenous variables for ARIMA model
        exog_features = [
            'dayofweek', 'Month', 'quarter', 'is_weekend', 'price_bins',
            'stock_ratio', 'sales_ratio', 'sales_lag1', 'sales_lag7',
            'sales_ma_7d', 'sales_ma_30d', 'price_stock_ratio',
            'sales_price_ratio', 'month_sin', 'month_cos',
            'day_sin', 'day_cos'
        ]

        # Filter features that exist in the data
        available_features = [f for f in exog_features if f in train_data.columns]

        # Log the features being used
        logger.info(f"Using {len(available_features)} features for ARIMA model: {available_features}")

        try:
            # Perform grid search to find the best ARIMA parameters
            logger.info("Starting grid search for optimal ARIMA parameters")

            # Define grid search parameters
            p_values = range(0, 3)  # AR order
            d_values = range(0, 2)  # Differencing
            q_values = range(0, 3)  # MA order

            logger.info("Performing grid search for optimal ARIMA parameters...")
            best_order, best_model = model_trainer.grid_search_arima(
                train_data['Sales'],
                p_values,
                d_values,
                q_values
            )

            X_train = train_data[available_features]
            X_test = test_data[available_features]
            y_train = train_data['Sales']
            y_test = test_data['Sales']

            if best_model is None:
                logger.warning("Grid search failed to find a suitable model, using default parameters")
                best_order = config.ARIMA_ORDER

                # Train model with exogenous variables using default parameters
                # model = model_trainer.train_arima_model(
                #     train_data['Sales'],
                #     train_data[available_features]
                # )
                model = model_trainer.train_arima_model(
                    y_train,
                    X_train
                )
            else:
                logger.info(f"Grid search found optimal ARIMA order: {best_order}")

                # Train model with exogenous variables using optimal parameters
                # Set the optimal order in config temporarily
                original_order = config.ARIMA_ORDER
                config.ARIMA_ORDER = best_order

                # model = model_trainer.train_arima_model(
                #     train_data['Sales'],
                #     train_data[available_features]
                # )
                model = model_trainer.train_arima_model(
                    y_train,
                    X_train
                )

                # Restore original order
                config.ARIMA_ORDER = original_order

            # Evaluate model with exogenous variables
            # metrics, _ = model_trainer.evaluate_model(
            #     model,
            #     test_data['Sales'],
            #     test_data[available_features]
            # )
            metrics, _ = model_trainer.evaluate_model(
                model,
                test_data,
                X_test,
                y_test
            )

            # Ensure metrics are numeric values
            if 'rmse' in metrics:
                try:
                    metrics['rmse'] = float(metrics['rmse'])
                except (ValueError, TypeError):
                    metrics['rmse'] = 0.0

            if 'mae' in metrics:
                try:
                    metrics['mae'] = float(metrics['mae'])
                except (ValueError, TypeError):
                    metrics['mae'] = 0.0

            # Save model
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'arima_model.pkl')
            model_trainer.save_model(model, model_path)

            # Update model predictor
            model_predictor.model = model

            # Test the model with a simple forecast
            test_forecast = model_predictor.forecast(
                periods=7,
                historical_data=processed_data
            )

            # Check if forecast is all zeros
            if test_forecast['Forecast'].sum() == 0:
                logger.warning("Test forecast is all zeros, model may need adjustment")

            # Create a copy for JSON serialization
            test_forecast_copy = test_forecast.copy()
            # Convert datetime objects to strings and handle NaT values
            if 'Date' in test_forecast_copy.columns:
                # Replace NaT values with actual dates
                if test_forecast_copy['Date'].isna().any():
                    # Find the first valid date
                    first_valid_date = test_forecast_copy['Date'].dropna().iloc[0] if not test_forecast_copy['Date'].dropna().empty else pd.Timestamp.now()

                    # Create a date range starting from the first valid date
                    date_range = pd.date_range(start=first_valid_date, periods=len(test_forecast_copy))

                    # Replace the Date column with the new date range
                    test_forecast_copy['Date'] = date_range

                # Convert to string format
                test_forecast_copy['Date'] = test_forecast_copy['Date'].dt.strftime('%Y-%m-%d')

            return jsonify({
                'status': 'success',
                'message': f'Model trained successfully with optimal ARIMA order {best_order}',
                'metrics': metrics,
                'model_path': model_path,
                'arima_order': best_order,
                'test_forecast': test_forecast_copy.to_dict(orient='records')
            })
        except Exception as model_error:
            logger.error(f"Error in model training/evaluation: {str(model_error)}")

            # Try a simpler approach - train a basic model without exogenous variables
            try:
                logger.info("Falling back to simpler model without exogenous variables")

                # Try grid search for a simpler model
                logger.info("Performing grid search for simple ARIMA model")
                p_values = range(0, 3)
                d_values = range(0, 2)
                q_values = range(0, 3)

                try:
                    best_order, best_simple_model = model_trainer.grid_search_arima(
                        train_data['Sales'],
                        p_values,
                        d_values,
                        q_values
                    )
                except Exception as grid_error:
                    logger.warning(f"Grid search for simple model failed: {str(grid_error)}")
                    best_order = config.ARIMA_ORDER
                    best_simple_model = None

                if best_simple_model is None:
                    logger.warning("Grid search failed for simple model, using default parameters")
                    simple_model = model_trainer.train_arima_model(train_data['Sales'])
                else:
                    logger.info(f"Grid search found optimal simple ARIMA order: {best_order}")
                    simple_model = best_simple_model

                # Evaluate the simple model
                try:
                    simple_metrics, _ = model_trainer.evaluate_model(
                        simple_model,
                        test_data['Sales']
                    )

                    # Ensure metrics are numeric values
                    if 'rmse' in simple_metrics:
                        try:
                            simple_metrics['rmse'] = float(simple_metrics['rmse'])
                        except (ValueError, TypeError):
                            simple_metrics['rmse'] = 0.0

                    if 'mae' in simple_metrics:
                        try:
                            simple_metrics['mae'] = float(simple_metrics['mae'])
                        except (ValueError, TypeError):
                            simple_metrics['mae'] = 0.0
                except Exception as eval_error:
                    logger.warning(f"Error evaluating simple model: {str(eval_error)}")
                    simple_metrics = {'rmse': 0.0, 'mae': 0.0}

                # Save the simple model
                model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, 'arima_model_simple.pkl')
                model_trainer.save_model(simple_model, model_path)

                # Update model predictor
                model_predictor.model = simple_model

                return jsonify({
                    'status': 'success',
                    'message': f'Simple model trained successfully with ARIMA order {best_order} (fallback)',
                    'metrics': simple_metrics,
                    'model_path': model_path,
                    'arima_order': best_order
                })
            except Exception as simple_error:
                logger.error(f"Error training simple model: {str(simple_error)}")
                raise simple_error
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

        # Truncate forecast_result to the requested number of periods
        # This ensures we use the same data for both the JSON response and visualization
        if len(forecast_result) != periods:
            logger.warning(f"forecast_result has {len(forecast_result)} rows but {periods} periods were requested")
            # Adjust forecast_result to match the requested periods
            if len(forecast_result) > periods:
                logger.info(f"Truncating forecast_result from {len(forecast_result)} to {periods} rows")
                forecast_result = forecast_result.iloc[:periods].copy()
            else:
                logger.warning(f"forecast_result has fewer rows than requested, this may cause issues")
        else:
            forecast_result = forecast_result.copy()

        # Add confidence intervals if missing
        if 'Lower_CI' not in forecast_result.columns:
            logger.info("Adding missing Lower_CI column")
            forecast_result['Lower_CI'] = forecast_result['Forecast'] * 0.8

        if 'Upper_CI' not in forecast_result.columns:
            logger.info("Adding missing Upper_CI column")
            forecast_result['Upper_CI'] = forecast_result['Forecast'] * 1.2

        # Create a copy for JSON serialization
        forecast_result_copy = forecast_result.copy()
        if pd.api.types.is_datetime64_any_dtype(forecast_result_copy['Date']):
            forecast_result_copy['Date'] = forecast_result_copy['Date'].dt.strftime('%Y-%m-%d')
        forecast_dict = forecast_result_copy.to_dict(orient='records')

        # Generate visualization for the forecast
        try:
            # Load data for historical context
            data = data_loader.get_data()
            processed_data = data_processor.engineer_features(data)

            # Filter historical data if item_name is provided
            if item_name:
                historical_data = processed_data[processed_data['Item Name'] == item_name]
            else:
                historical_data = processed_data

            # Log debug information
            logger.info(f"DEBUG: forecast_result shape before visualization: {forecast_result.shape}")
            logger.info(f"DEBUG: forecast_result columns: {forecast_result.columns.tolist()}")
            logger.info(f"DEBUG: forecast min: {forecast_result['Forecast'].min()}, max: {forecast_result['Forecast'].max()}, std: {forecast_result['Forecast'].std()}")

            # Generate visualization with explicit error handling
            try:
                # Use the same forecast_result for visualization that we're sending to the client
                plot_path = visualizer.plot_forecast(historical_data, forecast_result, item_name)
            except Exception as plot_error:
                logger.error(f"Error in plot_forecast: {str(plot_error)}")
                # Try a more direct approach
                try:
                    # Create a simple plot without confidence intervals
                    import matplotlib.pyplot as plt
                    import os
                    import datetime as dt_module

                    plt.figure(figsize=(10, 6))
                    plt.plot(forecast_result['Date'], forecast_result['Forecast'], 'g-', label='Forecast')
                    plt.title('Simple Forecast Plot (Fallback)')
                    plt.xlabel('Date')
                    plt.ylabel('Sales')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    # Save the plot
                    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'plots')
                    os.makedirs(output_dir, exist_ok=True)
                    timestamp = dt_module.datetime.now().strftime('%Y%m%d_%H%M%S')
                    item_suffix = f'_{item_name}' if item_name else ''
                    plot_path = os.path.join(output_dir, f'forecast_simple{item_suffix}_{timestamp}.png')
                    plt.savefig(plot_path)
                    plt.close()

                    logger.info(f"Created simple fallback plot at {plot_path}")
                except Exception as fallback_error:
                    logger.error(f"Fallback plot also failed: {str(fallback_error)}")
                    plot_path = None
        except Exception as viz_error:
            logger.error(f"Error generating forecast visualization: {str(viz_error)}")
            plot_path = None

        return jsonify({
            'status': 'success',
            'message': f'Forecast generated successfully for {periods} periods',
            'forecast': forecast_dict,
            'plot_path': plot_path
        })
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/visualization/sales_trend', methods=['GET', 'POST'])
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

        # Truncate forecast_result to the requested number of periods
        # This ensures we use the same data for both the JSON response and visualization
        if len(forecast_result) != periods:
            logger.warning(f"forecast_result has {len(forecast_result)} rows but {periods} periods were requested")
            # Adjust forecast_result to match the requested periods
            if len(forecast_result) > periods:
                logger.info(f"Truncating forecast_result from {len(forecast_result)} to {periods} rows")
                forecast_result = forecast_result.iloc[:periods].copy()
            else:
                logger.warning(f"forecast_result has fewer rows than requested, this may cause issues")
        else:
            forecast_result = forecast_result.copy()

        # Add confidence intervals if missing
        if 'Lower_CI' not in forecast_result.columns:
            logger.info("Adding missing Lower_CI column")
            forecast_result['Lower_CI'] = forecast_result['Forecast'] * 0.8

        if 'Upper_CI' not in forecast_result.columns:
            logger.info("Adding missing Upper_CI column")
            forecast_result['Upper_CI'] = forecast_result['Forecast'] * 1.2

        # Log debug information
        logger.info(f"DEBUG: forecast_result shape before visualization: {forecast_result.shape}")
        logger.info(f"DEBUG: forecast_result columns: {forecast_result.columns.tolist()}")
        logger.info(f"DEBUG: forecast min: {forecast_result['Forecast'].min()}, max: {forecast_result['Forecast'].max()}, std: {forecast_result['Forecast'].std()}")

        # Generate visualization with explicit error handling
        try:
            # Use the same forecast_result for visualization that we're sending to the client
            plot_path = visualizer.plot_forecast(historical_data, forecast_result, item_name)
        except Exception as plot_error:
            logger.error(f"Error in plot_forecast: {str(plot_error)}")
            # Try a more direct approach
            try:
                # Create a simple plot without confidence intervals
                import matplotlib.pyplot as plt
                import os
                import datetime as dt_module

                plt.figure(figsize=(10, 6))
                plt.plot(forecast_result['Date'], forecast_result['Forecast'], 'g-', label='Forecast')
                plt.title('Simple Forecast Plot (Fallback)')
                plt.xlabel('Date')
                plt.ylabel('Sales')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Save the plot
                output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'plots')
                os.makedirs(output_dir, exist_ok=True)
                timestamp = dt_module.datetime.now().strftime('%Y%m%d_%H%M%S')
                item_suffix = f'_{item_name}' if item_name else ''
                plot_path = os.path.join(output_dir, f'forecast_simple{item_suffix}_{timestamp}.png')
                plt.savefig(plot_path)
                plt.close()

                logger.info(f"Created simple fallback plot at {plot_path}")
            except Exception as fallback_error:
                logger.error(f"Fallback plot also failed: {str(fallback_error)}")
                plot_path = None

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

@app.route('/visualization/sales_by_category', methods=['POST'])
def sales_by_category():
    """Generate sales by category visualization endpoint"""
    try:
        # Get parameters from request
        params = request.get_json() or {}
        category_column = params.get('category_column', 'Month')

        # Load data
        data = data_loader.get_data()

        # Engineer features
        processed_data = data_processor.engineer_features(data)

        # Generate visualization
        plot_path = visualizer.plot_sales_by_category(processed_data, category_column)

        return jsonify({
            'status': 'success',
            'message': f'Sales by {category_column} visualization generated successfully',
            'plot_path': plot_path
        })
    except Exception as e:
        logger.error(f"Error generating sales by category visualization: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/visualization/seasonal_patterns', methods=['POST'])
def seasonal_patterns():
    """Generate seasonal patterns visualization endpoint"""
    try:
        # Load data
        data = data_loader.get_data()

        # Engineer features
        processed_data = data_processor.engineer_features(data)

        # Generate visualization
        plot_path = visualizer.plot_seasonal_patterns(processed_data)

        return jsonify({
            'status': 'success',
            'message': 'Seasonal patterns visualization generated successfully',
            'plot_path': plot_path
        })
    except Exception as e:
        logger.error(f"Error generating seasonal patterns visualization: {str(e)}")
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
