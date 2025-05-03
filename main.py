"""
Main entry point for Sales Automation Microservice
"""

import os
import sys
import logging
from api_gateway.app import app
from config import config
from utils.helpers import setup_logging, create_directory_if_not_exists

def main():
    """Main entry point for the application"""
    # Set up logging
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    create_directory_if_not_exists(log_dir)
    log_file = os.path.join(log_dir, 'app.log')
    setup_logging(log_level=logging.INFO, log_file=log_file)
    
    # Create necessary directories
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    create_directory_if_not_exists(data_dir)
    
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    create_directory_if_not_exists(models_dir)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    create_directory_if_not_exists(output_dir)
    
    plots_dir = os.path.join(output_dir, 'plots')
    create_directory_if_not_exists(plots_dir)
    
    # Run the application
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG_MODE
    )

if __name__ == '__main__':
    main()
