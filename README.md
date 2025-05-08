# Sales Automation Microservice

A microservice architecture for sales data analysis and forecasting.

## Overview

This project provides a microservice-based solution for analyzing retail sales data and generating forecasts. It is designed to be modular, scalable, and easy to deploy.

## Architecture

The system is composed of the following microservices:

1. **Data Service**: Handles data loading, preprocessing, and feature engineering
2. **Model Service**: Manages model training, evaluation, and prediction
3. **API Gateway**: Provides REST API endpoints for client applications
4. **Visualization Service**: Generates visualizations and reports
5. **Configuration Service**: Manages configuration settings

## Project Structure

```
Sales-Automation-Micro/
├── config/                  # Configuration settings
│   ├── README.md
│   └── config.py
├── data_service/            # Data loading and processing
│   ├── README.md
│   ├── __init__.py
│   ├── data_loader.py
│   └── data_processor.py
├── model_service/           # Model training and prediction
│   ├── README.md
│   ├── __init__.py
│   ├── model_trainer.py
│   └── model_predictor.py
├── api_gateway/             # REST API endpoints
│   ├── README.md
│   ├── __init__.py
│   └── app.py
├── visualization_service/   # Data visualization
│   ├── README.md
│   ├── __init__.py
│   └── visualizer.py
├── utils/                   # Utility functions
│   ├── README.md
│   ├── __init__.py
│   └── helpers.py
├── tests/                   # Unit tests
│   ├── README.md
│   ├── __init__.py
│   ├── test_api_gateway.py
│   ├── test_data_loader.py
│   ├── test_data_processor.py
│   ├── test_model_predictor.py
│   ├── test_model_trainer.py
│   └── test_visualizer.py
├── data/                    # Data directory
│   └── retail_store_data_with_real_items.csv
├── models/                  # Saved models
├── output/                  # Output files
│   └── plots/               # Generated visualizations
├── logs/                    # Log files
├── client.py                # Client script
├── main.py                  # Main entry point
├── streamlit_app.py         # Streamlit UI application
├── run.sh                   # Script to run the application
├── run_streamlit.sh         # Script to run Streamlit UI
├── run_tests.sh             # Script to run tests
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
├── .gitignore               # Git ignore file
├── README.md                # Main documentation
└── streamlit_README.md      # Streamlit UI documentation
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/Sales-Automation-Micro.git
   cd Sales-Automation-Micro
   ```

2. Create a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Place your data file in the `data` directory (create it if it doesn't exist).

## Usage

### Running the Application

You can run the application using the provided script:

```bash
./run.sh
```

This will:

1. Create a virtual environment if it doesn't exist
2. Install dependencies
3. Create necessary directories
4. Start the application

Alternatively, you can run the application directly:

```bash
python main.py
```

The API will be available at http://localhost:8080.

### Running Tests

You can run the tests using the provided script:

```bash
./run_tests.sh
```

This will run all the unit tests in the `tests` directory.

### Using the Client

The `client.py` script provides a command-line interface for interacting with the API:

```bash
# Check API health
python client.py --action health

# Load data
python client.py --action load

# Process data
python client.py --action process

# Train a model
python client.py --action train

# Generate a forecast
python client.py --action forecast --periods 30 --item "Coffee"

# Generate a sales trend visualization
python client.py --action sales_trend

# Generate an item comparison visualization
python client.py --action item_comparison --items "Coffee" "Tea" "Milk"

# Generate a forecast visualization
python client.py --action forecast_plot --periods 30 --item "Coffee"
```

### Docker Deployment

Build the Docker image:

```bash
docker build -t sales-automation-micro .
```

Run the container:

```bash
docker run -p 8080:8080 -v $(pwd)/data:/app/data sales-automation-micro
```

Or use Docker Compose:

```bash
docker-compose up
```

## API Endpoints

### Health Check

- `GET /health`: Health check endpoint

### Data Endpoints

- `GET /data/load`: Load data from the configured data source
- `GET /data/process`: Process data and perform feature engineering

### Model Endpoints

- `POST /model/train`: Train a sales forecasting model
- `POST /model/forecast`: Generate a sales forecast with parameters:
  - `periods` (optional): Number of periods to forecast (default: 30)
  - `start_date` (optional): Start date for forecasting (default: current date)
  - `item_name` (optional): Item name for item-specific forecasting

### Visualization Endpoints

- `GET /visualization/sales_trend`: Generate a sales trend visualization
- `POST /visualization/item_comparison`: Generate an item comparison visualization with parameters:
  - `items`: List of item names to compare
- `POST /visualization/forecast_plot`: Generate a forecast visualization with parameters:
  - `periods` (optional): Number of periods to forecast (default: 30)
  - `start_date` (optional): Start date for forecasting (default: current date)
  - `item_name` (optional): Item name for item-specific forecasting

## Example API Requests

### Generate a Forecast

```bash
curl -X POST http://localhost:8080/model/forecast \
  -H "Content-Type: application/json" \
  -d '{"periods": 30, "item_name": "Coffee"}'
```

### Train a Model

```bash
curl -X POST http://localhost:8080/model/train \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Microservice Architecture

This project follows a microservice architecture, with each service having a specific responsibility:

### Data Service

Responsible for data loading, preprocessing, and feature engineering. It provides clean, structured data for the other services.

### Model Service

Handles model training, evaluation, and prediction. It uses time series forecasting techniques to predict future sales.

### API Gateway

Provides a unified REST API for client applications to interact with the system. It routes requests to the appropriate services.

### Visualization Service

Generates visualizations of sales data and forecasts. It helps users understand trends and patterns in the data.

### Configuration Service

Manages configuration settings for the entire system. It provides a centralized way to configure the application.

## Streamlit UI

The project includes a Streamlit-based user interface that provides a web-based dashboard for interacting with the Sales Automation Microservice. The UI allows users to:

- Upload sales data files
- Process data and engineer features
- Train forecasting models
- Generate sales forecasts
- Create visualizations
- Export results to PDF

### Running the Streamlit UI

You can run the Streamlit UI using the provided script:

```bash
./run_streamlit.sh
```

This will:

1. Start the API service if it's not already running
2. Launch the Streamlit application

Alternatively, you can run the Streamlit application directly:

```bash
streamlit run streamlit_app.py
```

The Streamlit UI will be available at http://localhost:8501.

For more information about the Streamlit UI, see the [Streamlit UI documentation](streamlit_README.md).

## Future Enhancements

- Add authentication and authorization
- Implement real-time data processing
- Add support for more forecasting models
- Enhance the Streamlit UI with more features
- Implement caching for improved performance
- Add support for distributed training

## License

This project is licensed under the MIT License - see the LICENSE file for details.
