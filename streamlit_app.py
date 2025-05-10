"""
Streamlit UI for Sales Automation Microservice
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tempfile
from PIL import Image
import io
import base64
from fpdf import FPDF

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config import config

# Set page configuration
st.set_page_config(
    page_title="Sales Automation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = "http://0.0.0.0:8085"
print(f"Connecting to API at {API_BASE_URL}")

# Function to check API health
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Function to load data from API
def load_data_from_api():
    response = requests.get(f"{API_BASE_URL}/data/load")
    return response.json()

# Function to process data via API
def process_data_via_api():
    response = requests.get(f"{API_BASE_URL}/data/process")
    return response.json()

# Function to train model via API
def train_model_via_api(params=None):
    if params is None:
        params = {}
    response = requests.post(f"{API_BASE_URL}/model/train", json=params)
    return response.json()

# Function to generate forecast via API
def generate_forecast_via_api(params=None):
    if params is None:
        params = {}
    response = requests.post(f"{API_BASE_URL}/model/forecast", json=params)
    return response.json()

# Function to generate visualization via API
def generate_visualization_via_api(viz_type, params=None):
    if params is None:
        params = {}
    response = requests.post(f"{API_BASE_URL}/visualization/{viz_type}", json=params)
    return response.json()

# Function to create PDF from visualizations and data
def create_pdf(images, data_summary, forecast_summary=None):
    pdf = FPDF()
    pdf.add_page()

    # Add title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(190, 10, 'Sales Automation Report', 0, 1, 'C')
    pdf.ln(10)

    # Add timestamp
    pdf.set_font('Arial', '', 10)
    pdf.cell(190, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
    pdf.ln(5)

    # Add data summary
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 10, 'Data Summary', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    for key, value in data_summary.items():
        if key != 'sample':
            pdf.cell(190, 10, f'{key}: {value}', 0, 1, 'L')
    pdf.ln(5)

    # Add forecast summary if available
    if forecast_summary:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 10, 'Forecast Summary', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        for key, value in forecast_summary.items():
            if key != 'forecast_data':
                pdf.cell(190, 10, f'{key}: {value}', 0, 1, 'L')
        pdf.ln(5)

    # Add visualizations
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 10, 'Visualizations', 0, 1, 'L')

    for i, img_path in enumerate(images):
        try:
            # Check if the image path is a URL or local path
            if img_path.startswith('http'):
                response = requests.get(img_path)
                img = Image.open(io.BytesIO(response.content))
            else:
                img = Image.open(img_path)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
                img.save(temp.name, format='JPEG')
                pdf.image(temp.name, x=10, y=None, w=190)
                os.unlink(temp.name)  # Delete the temporary file

            pdf.ln(5)
        except Exception as e:
            st.error(f"Error adding image to PDF: {str(e)}")

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp:
        pdf_path = temp.name
        pdf.output(pdf_path)

    return pdf_path

# Function to get download link for a file
def get_download_link(file_path, link_text):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="sales_report.pdf">{link_text}</a>'
    return href

# Main application
def main():
    # Sidebar
    st.sidebar.title("Sales Automation")

    # Check API health
    api_healthy = check_api_health()
    if api_healthy:
        st.sidebar.success("✅ API is running")
    else:
        st.sidebar.error("❌ API is not available")
        st.error("The Sales Automation API is not available. Please start the API service and refresh this page.")
        return

    # Navigation
    page = st.sidebar.radio("Navigation", ["Upload Data", "Process Data", "Train Model", "Generate Forecast", "Visualizations"])

    # Session state initialization
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = None
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = []

    # Upload Data Page
    if page == "Upload Data":
        st.title("Upload Sales Data")

        # File upload option
        st.header("Option 1: Upload your own data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.uploaded_data = data
                st.success(f"Data uploaded successfully with {len(data)} rows")

                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(data.head())

                # Display data statistics
                st.subheader("Data Statistics")
                st.dataframe(data.describe())

            except Exception as e:
                st.error(f"Error uploading data: {str(e)}")

        # Use existing data option
        st.header("Option 2: Use existing data")
        if st.button("Load Sample Data"):
            try:
                response = load_data_from_api()
                if response['status'] == 'success':
                    # Convert sample data to DataFrame
                    sample_data = pd.DataFrame(response['sample'])
                    st.session_state.uploaded_data = sample_data

                    st.success(response['message'])

                    # Display data preview
                    st.subheader("Data Preview")
                    st.dataframe(sample_data)

                else:
                    st.error(f"Error loading data: {response['message']}")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    # Process Data Page
    elif page == "Process Data":
        st.title("Process Data")

        if st.session_state.uploaded_data is not None:
            st.info("Data is loaded and ready for processing")

            if st.button("Process Data"):
                with st.spinner("Processing data..."):
                    try:
                        response = process_data_via_api()
                        if response['status'] == 'success':
                            # Convert processed data to DataFrame
                            processed_data = pd.DataFrame(response['sample'])
                            st.session_state.processed_data = processed_data

                            st.success(response['message'])

                            # Display processed data preview
                            st.subheader("Processed Data Preview")
                            st.dataframe(processed_data)

                            # Display engineered features
                            st.subheader("Engineered Features")
                            st.write(response['features'])

                        else:
                            st.error(f"Error processing data: {response['message']}")
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
        else:
            st.warning("Please upload data first")
            if st.button("Go to Upload Data"):
                st.session_state.page = "Upload Data"
                st.experimental_rerun()

    # Train Model Page
    elif page == "Train Model":
        st.title("Train Forecasting Model")

        if st.session_state.processed_data is not None:
            st.info("Data is processed and ready for model training")

            # Information about grid search
            st.subheader("Automatic Parameter Optimization")
            st.write("""
            The system will automatically find the best ARIMA parameters using grid search.
            This process evaluates multiple combinations of parameters to find the optimal model.
            """)

            # Display grid search ranges
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**AR Order (p)**: 0-2")
            with col2:
                st.write("**Differencing (d)**: 0-1")
            with col3:
                st.write("**MA Order (q)**: 0-2")

            st.write("The grid search will test all combinations of these parameters and select the best model based on AIC (Akaike Information Criterion).")

            if st.button("Train Model with Grid Search"):
                with st.spinner("Training model with grid search... This may take several minutes"):
                    try:
                        # No parameters needed as grid search will find the best ones
                        response = train_model_via_api({})
                        if response['status'] == 'success':
                            st.session_state.model_trained = True

                            # Get the optimal ARIMA order from the response
                            arima_order = response.get('arima_order', 'N/A')

                            st.success(response['message'])

                            # Display the optimal parameters
                            st.subheader("Optimal ARIMA Parameters")
                            if isinstance(arima_order, (list, tuple)) and len(arima_order) == 3:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("AR Order (p)", arima_order[0])
                                with col2:
                                    st.metric("Differencing (d)", arima_order[1])
                                with col3:
                                    st.metric("MA Order (q)", arima_order[2])
                            else:
                                st.write(f"Optimal ARIMA Order: {arima_order}")

                            # Display model metrics
                            st.subheader("Model Evaluation Metrics")
                            metrics = response.get('metrics', {})
                            col1, col2 = st.columns(2)

                            # Handle RMSE metric - ensure it's a number before formatting
                            rmse = metrics.get('rmse', 'N/A')
                            if rmse != 'N/A':
                                try:
                                    rmse_display = f"{float(rmse):.4f}"
                                except (ValueError, TypeError):
                                    rmse_display = str(rmse)
                            else:
                                rmse_display = 'N/A'

                            # Handle MAE metric - ensure it's a number before formatting
                            mae = metrics.get('mae', 'N/A')
                            if mae != 'N/A':
                                try:
                                    mae_display = f"{float(mae):.4f}"
                                except (ValueError, TypeError):
                                    mae_display = str(mae)
                            else:
                                mae_display = 'N/A'

                            with col1:
                                st.metric("RMSE", rmse_display)
                            with col2:
                                st.metric("MAE", mae_display)

                            # Add explanation of metrics
                            st.info("""
                            **RMSE (Root Mean Square Error)**: Lower values indicate better model accuracy.

                            **MAE (Mean Absolute Error)**: Lower values indicate better model accuracy.
                            """)

                        else:
                            st.error(f"Error training model: {response['message']}")
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            st.warning("Please process data first")
            if st.button("Go to Process Data"):
                st.session_state.page = "Process Data"
                st.experimental_rerun()

    # Generate Forecast Page
    elif page == "Generate Forecast":
        st.title("Generate Sales Forecast")

        if st.session_state.model_trained:
            st.info("Model is trained and ready for forecasting")

            # Forecast parameters
            st.subheader("Forecast Parameters")
            col1, col2 = st.columns(2)
            with col1:
                periods = st.slider("Forecast Periods", 7, 90, 30)
                start_date = st.date_input("Start Date", datetime.now())
            with col2:
                # Get unique item names if available
                if st.session_state.uploaded_data is not None and 'Item Name' in st.session_state.uploaded_data.columns:
                    item_names = ['All Items'] + list(st.session_state.uploaded_data['Item Name'].unique())
                    item_name = st.selectbox("Select Item", item_names)
                    if item_name == 'All Items':
                        item_name = None
                else:
                    item_name = st.text_input("Item Name (optional)")
                    if item_name == '':
                        item_name = None

            if st.button("Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    try:
                        params = {
                            'periods': periods,
                            'start_date': start_date.isoformat(),
                            'item_name': item_name
                        }
                        response = generate_forecast_via_api(params)
                        if response['status'] == 'success':
                            st.session_state.forecast_data = response

                            st.success(response['message'])

                            # Display forecast plot
                            if 'plot_path' in response:
                                plot_path = response['plot_path']
                                st.subheader("Forecast Visualization")

                                # Add to visualizations list for PDF export
                                if plot_path not in st.session_state.visualizations:
                                    st.session_state.visualizations.append(plot_path)

                                # Display the image
                                try:
                                    # Check if file exists
                                    if not os.path.exists(plot_path):
                                        st.error(f"Plot file not found at path: {plot_path}")
                                    else:
                                        # Check file size to ensure it's valid
                                        file_size = os.path.getsize(plot_path)
                                        if file_size == 0:
                                            st.error("Plot file exists but is empty")
                                        else:
                                            # Open and display the image
                                            image = Image.open(plot_path)
                                            st.image(image, caption="Sales Forecast", use_column_width=True)
                                            st.success(f"Forecast visualization loaded successfully")
                                except Exception as e:
                                    st.error(f"Error displaying forecast plot: {str(e)}")
                                    st.info("Detailed error information to help debugging: " + str(e.__class__.__name__))

                        else:
                            st.error(f"Error generating forecast: {response['message']}")
                    except Exception as e:
                        st.error(f"Error generating forecast: {str(e)}")
        else:
            st.warning("Please train a model first")
            if st.button("Go to Train Model"):
                st.session_state.page = "Train Model"
                st.experimental_rerun()

    # Visualizations Page
    elif page == "Visualizations":
        st.title("Data Visualizations")

        if st.session_state.processed_data is not None:
            st.info("Data is processed and ready for visualization")

            # Visualization options
            st.subheader("Select Visualization Type")
            viz_type = st.selectbox(
                "Visualization Type",
                ["Sales Trend", "Item Comparison", "Sales by Category", "Seasonal Patterns"]
            )

            if viz_type == "Sales Trend":
                if st.button("Generate Sales Trend"):
                    with st.spinner("Generating visualization..."):
                        try:
                            response = generate_visualization_via_api('sales_trend')
                            if response['status'] == 'success':
                                st.success(response['message'])

                                # Display plot
                                if 'plot_path' in response:
                                    plot_path = response['plot_path']

                                    # Add to visualizations list for PDF export
                                    if plot_path not in st.session_state.visualizations:
                                        st.session_state.visualizations.append(plot_path)

                                    # Display the image
                                    try:
                                        image = Image.open(plot_path)
                                        st.image(image, caption="Sales Trend")
                                    except Exception as e:
                                        st.error(f"Error displaying plot: {str(e)}")
                            else:
                                st.error(f"Error generating visualization: {response['message']}")
                        except Exception as e:
                            st.error(f"Error generating visualization: {str(e)}")

            elif viz_type == "Item Comparison":
                # Get unique item names if available
                if st.session_state.uploaded_data is not None and 'Item Name' in st.session_state.uploaded_data.columns:
                    all_items = list(st.session_state.uploaded_data['Item Name'].unique())
                    selected_items = st.multiselect("Select Items to Compare", all_items)

                    if selected_items and st.button("Generate Item Comparison"):
                        with st.spinner("Generating visualization..."):
                            try:
                                params = {'items': selected_items}
                                response = generate_visualization_via_api('item_comparison', params)
                                if response['status'] == 'success':
                                    st.success(response['message'])

                                    # Display plot
                                    if 'plot_path' in response:
                                        plot_path = response['plot_path']

                                        # Add to visualizations list for PDF export
                                        if plot_path not in st.session_state.visualizations:
                                            st.session_state.visualizations.append(plot_path)

                                        # Display the image
                                        try:
                                            image = Image.open(plot_path)
                                            st.image(image, caption="Item Comparison")
                                        except Exception as e:
                                            st.error(f"Error displaying plot: {str(e)}")
                                else:
                                    st.error(f"Error generating visualization: {response['message']}")
                            except Exception as e:
                                st.error(f"Error generating visualization: {str(e)}")
                else:
                    st.warning("Item data not available for comparison")

            elif viz_type == "Sales by Category":
                # Get column names for category selection
                if st.session_state.processed_data is not None:
                    categorical_columns = [col for col in st.session_state.processed_data.columns
                                          if st.session_state.processed_data[col].dtype == 'object'
                                          or col in ['Month', 'Year', 'quarter', 'dayofweek', 'price_bins']]

                    if categorical_columns:
                        selected_category = st.selectbox("Select Category", categorical_columns)

                        if st.button("Generate Sales by Category"):
                            with st.spinner("Generating visualization..."):
                                try:
                                    params = {'category_column': selected_category}
                                    response = generate_visualization_via_api('sales_by_category', params)
                                    if response['status'] == 'success':
                                        st.success(response['message'])

                                        # Display plot
                                        if 'plot_path' in response:
                                            plot_path = response['plot_path']

                                            # Add to visualizations list for PDF export
                                            if plot_path not in st.session_state.visualizations:
                                                st.session_state.visualizations.append(plot_path)

                                            # Display the image
                                            try:
                                                image = Image.open(plot_path)
                                                st.image(image, caption=f"Sales by {selected_category}")
                                            except Exception as e:
                                                st.error(f"Error displaying plot: {str(e)}")
                                    else:
                                        st.error(f"Error generating visualization: {response['message']}")
                                except Exception as e:
                                    st.error(f"Error generating visualization: {str(e)}")
                    else:
                        st.warning("No categorical columns available for analysis")
                else:
                    st.warning("Processed data not available")

            elif viz_type == "Seasonal Patterns":
                if st.button("Generate Seasonal Patterns"):
                    with st.spinner("Generating visualization..."):
                        try:
                            response = generate_visualization_via_api('seasonal_patterns')
                            if response['status'] == 'success':
                                st.success(response['message'])

                                # Display plot
                                if 'plot_path' in response:
                                    plot_path = response['plot_path']

                                    # Add to visualizations list for PDF export
                                    if plot_path not in st.session_state.visualizations:
                                        st.session_state.visualizations.append(plot_path)

                                    # Display the image
                                    try:
                                        image = Image.open(plot_path)
                                        st.image(image, caption="Seasonal Patterns")
                                    except Exception as e:
                                        st.error(f"Error displaying plot: {str(e)}")
                            else:
                                st.error(f"Error generating visualization: {response['message']}")
                        except Exception as e:
                            st.error(f"Error generating visualization: {str(e)}")

            # Export to PDF section
            if st.session_state.visualizations:
                st.subheader("Export Report to PDF")

                if st.button("Generate PDF Report"):
                    with st.spinner("Generating PDF report..."):
                        try:
                            # Get data summary
                            data_summary = {
                                'Total Rows': len(st.session_state.uploaded_data) if st.session_state.uploaded_data is not None else 'N/A',
                                'Date Range': f"{st.session_state.uploaded_data['Date'].min()} to {st.session_state.uploaded_data['Date'].max()}" if st.session_state.uploaded_data is not None and 'Date' in st.session_state.uploaded_data.columns else 'N/A',
                                'Number of Items': len(st.session_state.uploaded_data['Item Name'].unique()) if st.session_state.uploaded_data is not None and 'Item Name' in st.session_state.uploaded_data.columns else 'N/A',
                                'Total Sales': f"{st.session_state.uploaded_data['Sales'].sum():,.2f}" if st.session_state.uploaded_data is not None and 'Sales' in st.session_state.uploaded_data.columns else 'N/A',
                                'Total Revenue': f"${st.session_state.uploaded_data['Revenue'].sum():,.2f}" if st.session_state.uploaded_data is not None and 'Revenue' in st.session_state.uploaded_data.columns else 'N/A'
                            }

                            # Get forecast summary if available
                            forecast_summary = None
                            if st.session_state.forecast_data is not None:
                                forecast_summary = {
                                    'Forecast Periods': st.session_state.forecast_data.get('periods', 'N/A'),
                                    'Start Date': st.session_state.forecast_data.get('start_date', 'N/A'),
                                    'Item': st.session_state.forecast_data.get('item_name', 'All Items'),
                                    'Model Type': 'ARIMA'
                                }

                            # Create PDF
                            pdf_path = create_pdf(st.session_state.visualizations, data_summary, forecast_summary)

                            # Display download link
                            st.markdown(get_download_link(pdf_path, "Download PDF Report"), unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"Error generating PDF report: {str(e)}")
        else:
            st.warning("Please process data first")
            if st.button("Go to Process Data"):
                st.session_state.page = "Process Data"
                st.experimental_rerun()

if __name__ == "__main__":
    main()
