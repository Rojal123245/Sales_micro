# Tests

This directory contains unit tests for the Sales Automation Microservice.

## Test Files

- `test_data_loader.py`: Tests for the DataLoader class
- `test_data_processor.py`: Tests for the DataProcessor class
- `test_model_trainer.py`: Tests for the ModelTrainer class
- `test_model_predictor.py`: Tests for the ModelPredictor class
- `test_api_gateway.py`: Tests for the API Gateway
- `test_visualizer.py`: Tests for the Visualizer class

## Running Tests

To run all tests:

```bash
python -m unittest discover -s tests
```

To run a specific test file:

```bash
python -m unittest tests.test_data_loader
```

To run a specific test case:

```bash
python -m unittest tests.test_data_loader.TestDataLoader.test_load_data
```

## Test Coverage

To generate a test coverage report:

```bash
pytest --cov=. tests/
```
