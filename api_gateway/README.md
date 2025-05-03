# API Gateway

This service provides a REST API for interacting with the Sales Automation Microservice.

## Endpoints

### Health Check

```
GET /health
```

Returns the health status of the API.

### Data Endpoints

```
GET /data/load
```

Loads data from the configured data source.

```
GET /data/process
```

Processes data and performs feature engineering.

### Model Endpoints

```
POST /model/train
```

Trains a sales forecasting model.

```
POST /model/forecast
```

Generates a sales forecast.

Parameters:

- `periods` (optional): Number of periods to forecast (default: 30)
- `start_date` (optional): Start date for forecasting (default: current date)
- `item_name` (optional): Item name for item-specific forecasting

### Visualization Endpoints

```
GET /visualization/sales_trend
```

Generates a sales trend visualization.

```
POST /visualization/item_comparison
```

Generates an item comparison visualization.

Parameters:

- `items`: List of item names to compare

```
POST /visualization/forecast_plot
```

Generates a forecast visualization.

Parameters:

- `periods` (optional): Number of periods to forecast (default: 30)
- `start_date` (optional): Start date for forecasting (default: current date)
- `item_name` (optional): Item name for item-specific forecasting

## Usage

Start the API server:

```
python -m api_gateway.app
```

The API will be available at http://localhost:8080.
