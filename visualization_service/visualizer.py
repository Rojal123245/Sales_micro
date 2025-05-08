"""
Visualizer module for generating visualizations of sales data and forecasts
"""

import os
import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend to avoid GUI issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import datetime as dt_module  # Add an alternative import for datetime module

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Class for generating visualizations of sales data and forecasts
    """

    def __init__(self, config):
        """
        Initialize the Visualizer with configuration

        Args:
            config: Configuration object containing parameters
        """
        self.config = config
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'plots')
        os.makedirs(self.output_dir, exist_ok=True)

        # Set plot style
        if hasattr(self.config, 'PLOT_STYLE'):
            plt.style.use(self.config.PLOT_STYLE)

        logger.info("Visualizer initialized")

    def plot_sales_trend(self, data):
        """
        Plot sales trend over time

        Args:
            data (pandas.DataFrame): Sales data

        Returns:
            str: Path to the saved plot
        """
        try:
            logger.info("Generating sales trend plot")

            # Group by date and calculate daily sales
            daily_sales = data.groupby('Date')['Sales'].sum().reset_index()

            # Create figure
            plt.figure(figsize=self.config.FIGURE_SIZE)

            # Plot sales trend
            plt.plot(daily_sales['Date'], daily_sales['Sales'], marker='o', linestyle='-', alpha=0.7)

            # Add trend line
            z = np.polyfit(range(len(daily_sales)), daily_sales['Sales'], 1)
            p = np.poly1d(z)
            plt.plot(daily_sales['Date'], p(range(len(daily_sales))), "r--", alpha=0.7)

            # Add labels and title
            plt.title('Daily Sales Trend', fontsize=15)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Total Sales', fontsize=12)
            plt.grid(True, alpha=0.3)

            # Rotate x-axis labels
            plt.xticks(rotation=45)

            # Tight layout
            plt.tight_layout()

            # Save plot
            timestamp = dt_module.datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.output_dir, f'sales_trend_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close()

            logger.info(f"Sales trend plot saved to {plot_path}")
            return plot_path
        except Exception as e:
            logger.error(f"Error generating sales trend plot: {str(e)}")
            raise

    def plot_item_comparison(self, data, items):
        """
        Plot sales comparison for specific items

        Args:
            data (pandas.DataFrame): Sales data
            items (list): List of item names to compare

        Returns:
            str: Path to the saved plot
        """
        try:
            logger.info(f"Generating item comparison plot for {len(items)} items")

            # Filter data for selected items
            filtered_data = data[data['Item Name'].isin(items)]

            if filtered_data.empty:
                logger.error("No data found for the specified items")
                raise ValueError("No data found for the specified items")

            # Group by date and item name
            item_sales = filtered_data.groupby(['Date', 'Item Name'])['Sales'].sum().reset_index()

            # Create figure
            plt.figure(figsize=self.config.FIGURE_SIZE)

            # Plot sales for each item
            for item in items:
                item_data = item_sales[item_sales['Item Name'] == item]
                plt.plot(item_data['Date'], item_data['Sales'], marker='o', linestyle='-', alpha=0.7, label=item)

            # Add labels and title
            plt.title('Sales Comparison by Item', fontsize=15)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Sales', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Rotate x-axis labels
            plt.xticks(rotation=45)

            # Tight layout
            plt.tight_layout()

            # Save plot
            timestamp = dt_module.datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.output_dir, f'item_comparison_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close()

            logger.info(f"Item comparison plot saved to {plot_path}")
            return plot_path
        except Exception as e:
            logger.error(f"Error generating item comparison plot: {str(e)}")
            raise

    def plot_forecast(self, historical_data, forecast_data, item_name=None):
        """
        Plot forecasted sales with historical data and confidence intervals

        Args:
            historical_data (pandas.DataFrame): Historical sales data
            forecast_data (pandas.DataFrame): Forecast data with Date, Forecast, Lower_CI, Upper_CI columns
            item_name (str, optional): Name of the item being forecasted

        Returns:
            str: Path to the saved plot
        """
        try:
            logger.info(f"Generating forecast plot{' for ' + item_name if item_name else ''}")

            # Create figure
            plt.figure(figsize=self.config.FIGURE_SIZE)

            # Ensure forecast_data has the required columns
            required_columns = ['Date', 'Forecast']
            for col in required_columns:
                if col not in forecast_data.columns:
                    logger.error(f"Missing required column {col} in forecast_data")
                    raise ValueError(f"Missing required column {col} in forecast_data")

            # Add confidence intervals if missing
            if 'Lower_CI' not in forecast_data.columns:
                logger.info("Adding missing Lower_CI column")
                forecast_data['Lower_CI'] = forecast_data['Forecast'] * 0.8

            if 'Upper_CI' not in forecast_data.columns:
                logger.info("Adding missing Upper_CI column")
                forecast_data['Upper_CI'] = forecast_data['Forecast'] * 1.2

            # Prepare historical data
            if historical_data is not None and not historical_data.empty:
                # If item_name is provided, filter historical data for that item
                if item_name:
                    if 'Item Name' in historical_data.columns:
                        historical_data = historical_data[historical_data['Item Name'] == item_name]
                    else:
                        logger.warning(f"Item Name column not found in historical data, using all data")

                # Group by date to get daily sales
                if 'Date' in historical_data.columns and 'Sales' in historical_data.columns:
                    daily_sales = historical_data.groupby('Date')['Sales'].sum().reset_index()

                    # Split historical data into training and actual (for visualization)
                    # Use the last 30% of the data as "actual" for comparison
                    split_idx = int(len(daily_sales) * 0.7)
                    training_data = daily_sales.iloc[:split_idx]
                    actual_data = daily_sales.iloc[split_idx:]

                    # Plot training data
                    plt.plot(training_data['Date'], training_data['Sales'], 'b-',
                             alpha=0.7, label='Training')

                    # Plot actual data
                    plt.plot(actual_data['Date'], actual_data['Sales'], 'orange',
                             alpha=0.7, label='Actual')
                else:
                    logger.warning("Date or Sales columns not found in historical data")

            # Ensure Date column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(forecast_data['Date']):
                logger.info("Converting Date column to datetime")
                forecast_data['Date'] = pd.to_datetime(forecast_data['Date'], errors='coerce')
                # Drop rows with NaT in Date column
                forecast_data = forecast_data.dropna(subset=['Date'])

            # Plot forecast data
            try:
                # Ensure Date and Forecast columns have the same length
                dates = forecast_data['Date'].values
                forecasts = forecast_data['Forecast'].values

                # Log array shapes and statistics for debugging
                logger.info(f"DEBUG: dates shape: {dates.shape}, forecasts shape: {forecasts.shape}")
                logger.info(f"DEBUG: forecast min: {forecast_data['Forecast'].min()}, max: {forecast_data['Forecast'].max()}, std: {forecast_data['Forecast'].std()}")

                # Make sure arrays have the same length
                min_length = min(len(dates), len(forecasts))
                if min_length < len(dates) or min_length < len(forecasts):
                    logger.warning(f"Array length mismatch. Truncating to min length: {min_length}")
                    dates = dates[:min_length]
                    forecasts = forecasts[:min_length]

                # Check if any array is empty
                if min_length > 0:
                    # Use a more visible line style and marker to highlight variability
                    plt.plot(dates, forecasts, 'g-', alpha=0.8, linewidth=2, label='Forecast')

                    # Add markers to make the variability more visible
                    plt.plot(dates, forecasts, 'go', alpha=0.6, markersize=4)
                else:
                    logger.warning("Cannot plot forecast: arrays are empty")
            except Exception as forecast_error:
                logger.error(f"Error plotting forecast: {str(forecast_error)}")
                raise

            # Plot confidence intervals if available
            if 'Lower_CI' in forecast_data.columns and 'Upper_CI' in forecast_data.columns:
                try:
                    # Ensure all arrays have the same length
                    dates = forecast_data['Date'].values
                    lower_ci = forecast_data['Lower_CI'].values
                    upper_ci = forecast_data['Upper_CI'].values

                    # Log array shapes for debugging
                    logger.info(f"DEBUG: dates shape: {dates.shape}, lower_ci shape: {lower_ci.shape}, upper_ci shape: {upper_ci.shape}")

                    # Make sure all arrays have the same length
                    min_length = min(len(dates), len(lower_ci), len(upper_ci))
                    if min_length < len(dates) or min_length < len(lower_ci) or min_length < len(upper_ci):
                        logger.warning(f"Array length mismatch. Truncating to min length: {min_length}")
                        dates = dates[:min_length]
                        lower_ci = lower_ci[:min_length]
                        upper_ci = upper_ci[:min_length]

                    # Check if any array is empty
                    if min_length > 0:
                        # Use a more visible fill for confidence intervals
                        plt.fill_between(dates, lower_ci, upper_ci, color='g', alpha=0.3)

                        # Add lines for upper and lower bounds to make them more visible
                        plt.plot(dates, lower_ci, 'g--', alpha=0.5, linewidth=1, label='Lower CI')
                        plt.plot(dates, upper_ci, 'g--', alpha=0.5, linewidth=1, label='Upper CI')
                    else:
                        logger.warning("Cannot plot confidence intervals: arrays are empty")
                except Exception as ci_error:
                    logger.error(f"Error plotting confidence intervals: {str(ci_error)}")
                    logger.info("Continuing without confidence intervals")

            # Add title and labels
            title = 'ARIMA Forecast with Seasonal Patterns'
            if item_name:
                title = f'ARIMA Forecast for {item_name} with Seasonal Patterns'

            plt.title(title, fontsize=15)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Sales', fontsize=12)

            # Add a more visible grid to help see patterns
            plt.grid(True, alpha=0.4, linestyle='--')

            # Add a more visible legend
            plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)

            # Add a note about seasonal patterns
            plt.figtext(0.5, 0.01,
                       "Note: Forecast includes day-of-week seasonal patterns (weekends higher, Mondays lower)",
                       ha='center', fontsize=9, style='italic')

            # Rotate x-axis labels
            plt.xticks(rotation=45)

            # Tight layout
            plt.tight_layout()

            # Save plot
            timestamp = dt_module.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'forecast_{timestamp}.png'
            if item_name:
                # Clean item name for filename (remove special characters)
                clean_item_name = ''.join(c if c.isalnum() else '_' for c in item_name)
                filename = f'forecast_{clean_item_name}_{timestamp}.png'

            plot_path = os.path.join(self.output_dir, filename)
            plt.savefig(plot_path)
            plt.close()

            logger.info(f"Forecast plot saved to {plot_path}")
            return plot_path
        except Exception as e:
            logger.error(f"Error generating forecast plot: {str(e)}")
            raise


    def plot_sales_by_category(self, data, category_column):
        """
        Plot sales by category

        Args:
            data (pandas.DataFrame): Sales data
            category_column (str): Column name for categorization

        Returns:
            str: Path to the saved plot
        """
        try:
            logger.info(f"Generating sales by {category_column} plot")

            # Group by category
            category_sales = data.groupby(category_column)['Sales'].sum().reset_index()

            # Sort by sales
            category_sales = category_sales.sort_values('Sales', ascending=False)

            # Create figure
            plt.figure(figsize=self.config.FIGURE_SIZE)

            # Create bar plot
            sns.barplot(x=category_column, y='Sales', data=category_sales)

            # Add labels and title
            plt.title(f'Sales by {category_column}', fontsize=15)
            plt.xlabel(category_column, fontsize=12)
            plt.ylabel('Total Sales', fontsize=12)
            plt.grid(True, alpha=0.3)

            # Rotate x-axis labels if needed
            if len(category_sales) > 5:
                plt.xticks(rotation=45)

            # Tight layout
            plt.tight_layout()

            # Save plot
            timestamp = dt_module.datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.output_dir, f'sales_by_{category_column}_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close()

            logger.info(f"Sales by {category_column} plot saved to {plot_path}")
            return plot_path
        except Exception as e:
            logger.error(f"Error generating sales by category plot: {str(e)}")
            raise

    def plot_seasonal_patterns(self, data):
        """
        Plot seasonal patterns in sales

        Args:
            data (pandas.DataFrame): Sales data

        Returns:
            str: Path to the saved plot
        """
        try:
            logger.info("Generating seasonal patterns plot")

            # Group by month and day of week
            monthly_sales = data.groupby('Month')['Sales'].sum().reset_index()
            dow_sales = data.groupby('dayofweek')['Sales'].sum().reset_index()

            # Map day of week numbers to names
            dow_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
                        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            dow_sales['Day'] = dow_sales['dayofweek'].map(dow_names)

            # Create figure with subplots
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.FIGURE_SIZE)

            # Plot monthly pattern
            sns.barplot(x='Month', y='Sales', data=monthly_sales, ax=ax1)
            ax1.set_title('Monthly Sales Pattern', fontsize=12)
            ax1.set_xlabel('Month', fontsize=10)
            ax1.set_ylabel('Total Sales', fontsize=10)
            ax1.grid(True, alpha=0.3)

            # Plot day of week pattern
            sns.barplot(x='Day', y='Sales', data=dow_sales, ax=ax2)
            ax2.set_title('Day of Week Sales Pattern', fontsize=12)
            ax2.set_xlabel('Day of Week', fontsize=10)
            ax2.set_ylabel('Total Sales', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

            # Tight layout
            plt.tight_layout()

            # Save plot
            timestamp = dt_module.datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.output_dir, f'seasonal_patterns_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close()

            logger.info(f"Seasonal patterns plot saved to {plot_path}")
            return plot_path
        except Exception as e:
            logger.error(f"Error generating seasonal patterns plot: {str(e)}")
            raise
