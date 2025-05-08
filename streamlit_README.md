# Sales Automation Streamlit UI

This is a Streamlit-based user interface for the Sales Automation Microservice. It provides a web-based dashboard for uploading sales data, processing it, training forecasting models, generating forecasts, and visualizing the results.

## Features

- **Data Upload**: Upload your own CSV files or use existing sample data
- **Data Processing**: Process data and engineer features for forecasting
- **Model Training**: Train ARIMA models with customizable parameters
- **Forecasting**: Generate sales forecasts with various parameters
- **Visualizations**: Create different types of visualizations for sales data
- **PDF Export**: Export visualizations and results to a PDF report

## Prerequisites

- Python 3.8+
- Sales Automation Microservice API running

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the Sales Automation API is running:

```bash
python api_gateway/app.py
```

## Usage

1. Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

3. Follow the steps in the UI:
   - Upload or select data
   - Process the data
   - Train a forecasting model
   - Generate forecasts
   - Create visualizations
   - Export results to PDF

## Workflow

1. **Upload Data**: Start by uploading your sales data CSV file or using the sample data
2. **Process Data**: Process the data to engineer features for forecasting
3. **Train Model**: Train an ARIMA model with your preferred parameters
4. **Generate Forecast**: Create a forecast for future sales
5. **Visualizations**: Explore different visualizations of your data
6. **Export to PDF**: Generate a PDF report with all visualizations and results

## API Integration

The Streamlit UI integrates with the Sales Automation Microservice API to perform data processing, model training, and forecasting. Make sure the API is running before using the UI.

## Troubleshooting

- If you see an error about the API not being available, make sure the API service is running
- If you encounter issues with data upload, ensure your CSV file has the required columns (Date, Item Name, Price, Total Stock, Stock Left, Sales, Revenue)
- For PDF export issues, ensure you have the necessary permissions to create temporary files
