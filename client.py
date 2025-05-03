"""
Client script to interact with the Sales Automation Microservice API
"""

import requests
import json
import argparse
from datetime import datetime, timedelta

def health_check(base_url):
    """Check if the API is running"""
    response = requests.get(f"{base_url}/health")
    return response.json()

def load_data(base_url):
    """Load data from the API"""
    response = requests.get(f"{base_url}/data/load")
    return response.json()

def process_data(base_url):
    """Process data using the API"""
    response = requests.get(f"{base_url}/data/process")
    return response.json()

def train_model(base_url):
    """Train a model using the API"""
    response = requests.post(f"{base_url}/model/train", json={})
    return response.json()

def generate_forecast(base_url, periods=30, item_name=None):
    """Generate a forecast using the API"""
    data = {
        'periods': periods,
        'start_date': datetime.now().isoformat()
    }

    if item_name:
        data['item_name'] = item_name

    response = requests.post(f"{base_url}/model/forecast", json=data)
    return response.json()

def generate_sales_trend(base_url):
    """Generate a sales trend visualization using the API"""
    response = requests.get(f"{base_url}/visualization/sales_trend")
    return response.json()

def generate_item_comparison(base_url, items):
    """Generate an item comparison visualization using the API"""
    response = requests.post(f"{base_url}/visualization/item_comparison", json={'items': items})
    return response.json()

def generate_forecast_plot(base_url, periods=30, item_name=None):
    """Generate a forecast visualization using the API"""
    data = {
        'periods': periods,
        'start_date': datetime.now().isoformat()
    }

    if item_name:
        data['item_name'] = item_name

    response = requests.post(f"{base_url}/visualization/forecast_plot", json=data)
    return response.json()

def main():
    """Main entry point for the client script"""
    parser = argparse.ArgumentParser(description='Client for Sales Automation Microservice API')
    parser.add_argument('--url', default='http://localhost:8080', help='Base URL for the API')
    parser.add_argument('--action', required=True, choices=[
        'health', 'load', 'process', 'train', 'forecast',
        'sales_trend', 'item_comparison', 'forecast_plot'
    ], help='Action to perform')
    parser.add_argument('--periods', type=int, default=30, help='Number of periods to forecast')
    parser.add_argument('--item', help='Item name for item-specific actions')
    parser.add_argument('--items', nargs='+', help='List of items for comparison')

    args = parser.parse_args()

    if args.action == 'health':
        result = health_check(args.url)
    elif args.action == 'load':
        result = load_data(args.url)
    elif args.action == 'process':
        result = process_data(args.url)
    elif args.action == 'train':
        result = train_model(args.url)
    elif args.action == 'forecast':
        result = generate_forecast(args.url, args.periods, args.item)
    elif args.action == 'sales_trend':
        result = generate_sales_trend(args.url)
    elif args.action == 'item_comparison':
        if not args.items:
            parser.error("--items is required for item_comparison action")
        result = generate_item_comparison(args.url, args.items)
    elif args.action == 'forecast_plot':
        result = generate_forecast_plot(args.url, args.periods, args.item)

    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
