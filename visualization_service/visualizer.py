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
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
        Plot historical data and forecast with ARIMA model using exogenous variables

        Args:
            historical_data (pandas.DataFrame): Historical sales data
            forecast_data (pandas.DataFrame): Forecast data
            item_name (str, optional): Item name for specific item forecast

        Returns:
            str: Path to the saved plot
        """
        try:
            logger.info("Generating forecast plot")

            # Prepare historical data
            if item_name:
                title = f'ARIMA Forecast with Feature Engineering for {item_name}'
                historical_data = historical_data[historical_data['Item Name'] == item_name]
            else:
                title = 'ARIMA Forecast with Feature Engineering'

            # Group historical data by date
            historical_daily = historical_data.groupby('Date')['Sales'].sum().reset_index()

            # Sort data by date
            historical_daily = historical_daily.sort_values('Date')
            forecast_data = forecast_data.sort_values('Date')

            # Create figure
            plt.figure(figsize=self.config.FIGURE_SIZE)

            # Get the last 30 days of historical data for better visualization
            if len(historical_daily) > 30:
                historical_plot_data = historical_daily.iloc[-30:]
            else:
                historical_plot_data = historical_daily

            # Plot historical data
            plt.plot(historical_plot_data['Date'], historical_plot_data['Sales'],
                    color='blue', marker='o', linestyle='-', linewidth=2,
                    markersize=6, alpha=0.7, label='Training')

            # Plot forecast - convert Date to datetime if it's a string
            if isinstance(forecast_data['Date'].iloc[0], str):
                forecast_data['Date'] = pd.to_datetime(forecast_data['Date'])

            # Limit forecast to first 30 days for better visualization
            forecast_plot_data = forecast_data.head(30)

            plt.plot(forecast_plot_data['Date'], forecast_plot_data['Forecast'],
                    color='red', marker='o', linestyle='--', linewidth=2,
                    markersize=6, alpha=0.7, label='Forecast')

            # Add confidence interval
            if 'Forecast' in forecast_data.columns:
                # Calculate RMSE-based confidence interval (simulated)
                rmse = historical_daily['Sales'].std() * 0.5  # Simulated RMSE
                plt.fill_between(
                    forecast_plot_data['Date'],
                    forecast_plot_data['Forecast'] - rmse,
                    forecast_plot_data['Forecast'] + rmse,
                    color='red', alpha=0.2, label='95% Confidence Interval'
                )

            # Add labels and title
            plt.title(title, fontsize=15)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Sales', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Rotate x-axis labels
            plt.xticks(rotation=45)

            # Tight layout
            plt.tight_layout()

            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            item_suffix = f'_{item_name}' if item_name else ''
            plot_path = os.path.join(self.output_dir, f'forecast{item_suffix}_{timestamp}.png')
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
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.FIGURE_SIZE)

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
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.output_dir, f'seasonal_patterns_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close()

            logger.info(f"Seasonal patterns plot saved to {plot_path}")
            return plot_path
        except Exception as e:
            logger.error(f"Error generating seasonal patterns plot: {str(e)}")
            raise
