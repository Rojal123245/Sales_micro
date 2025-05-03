# Model Service

This service is responsible for training, evaluating, and making predictions with sales forecasting models.

## Components

### ModelTrainer

The `ModelTrainer` class provides functionality for:

- Training ARIMA models for time series forecasting
- Grid search for optimal ARIMA parameters
- Model evaluation
- Saving and loading models

### ModelPredictor

The `ModelPredictor` class provides functionality for:

- Loading trained models
- Generating forecasts for future periods
- Item-specific forecasting
- Feature preparation for forecasting

## Usage

```python
from model_service.model_trainer import ModelTrainer
from model_service.model_predictor import ModelPredictor
from config import config

# Initialize services
trainer = ModelTrainer(config)
predictor = ModelPredictor(config)

# Train a model
model = trainer.train_arima_model(train_data['Sales'], train_data[config.FEATURES])

# Evaluate the model
metrics, forecast = trainer.evaluate_model(model, test_data['Sales'], test_data[config.FEATURES])

# Save the model
trainer.save_model(model, 'models/arima_model.pkl')

# Load the model
predictor.load_model('models/arima_model.pkl')

# Generate a forecast
forecast_result = predictor.forecast(periods=30, start_date=datetime.now(), historical_data=data)
```
