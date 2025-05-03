# Visualization Service

This service is responsible for generating visualizations of sales data and forecasts.

## Components

### Visualizer

The `Visualizer` class provides functionality for:

- Plotting sales trends over time
- Comparing sales for different items
- Visualizing forecasts with historical data
- Analyzing sales by category
- Identifying seasonal patterns

## Usage

```python
from visualization_service.visualizer import Visualizer
from config import config

# Initialize service
visualizer = Visualizer(config)

# Plot sales trend
plot_path = visualizer.plot_sales_trend(data)

# Plot item comparison
plot_path = visualizer.plot_item_comparison(data, ['Batteries', 'Soap'])

# Plot forecast
plot_path = visualizer.plot_forecast(historical_data, forecast_data, item_name='Batteries')

# Plot sales by category
plot_path = visualizer.plot_sales_by_category(data, 'Item Name')

# Plot seasonal patterns
plot_path = visualizer.plot_seasonal_patterns(data)
```

## Output

All visualizations are saved as PNG files in the configured output directory.
