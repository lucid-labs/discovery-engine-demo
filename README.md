# DeFi APY Prediction Modelling

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Example: Supply/Borrow APY Prediction](#example-supplyborrow-apy-prediction)
7. [Customization](#customization)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction

This project implements a flexible and extensible framework for predicting DeFi (Decentralized Finance) metrics, with a focus on supply and borrow APY (Annual Percentage Yield) prediction. The system uses machine learning techniques, including neural architecture search and hyperparameter optimization, to create accurate forecasting models based on historical data.

Key features:
- Flexible data processing for multiple CSV files
- Automated feature engineering
- Neural architecture search to find the best model structure
- Hyperparameter optimization using Optuna
- Comprehensive evaluation metrics

## Project Structure

```
project/
├── data/
│   └── raw/
├── src/
│   ├── data_processing/
│   ├── feature_engineering/
│   ├── models/
│   ├── architecture_search/
│   ├── training/
│   ├── evaluation/
│   └── hyperparameter_optimization/
├── notebooks/
├── configs/
├── main.py
├── requirements.txt
└── README.md
```

- `data/raw/`: Directory for storing input CSV files
- `src/`: Source code for all components of the system
- `notebooks/`: Jupyter notebooks for exploratory data analysis
- `configs/`: Configuration files (YAML)
- `main.py`: Main script to run the entire pipeline

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/lucid-labs/discovery-engine-demo
   cd discovery-engine-demo

   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Configuration

The project uses a YAML configuration file (`configs/config.yaml`) to set various parameters. Here's an example configuration for supply/borrow APY prediction:

```yaml
# Data settings
data_dir: "data/raw/"

# Target columns
target_columns:
  - "lenderRate"  # Supply APY
  - "borrowRate"  # Borrow APY

# Feature columns
feature_columns:
  - "totalValueLockedUSD"
  - "totalBorrowBalanceUSD"
  - "totalDepositBalanceUSD"
  - "hourlyDepositUSD"
  - "hourlyRepayUSD"
  - "hourlyBorrowUSD"
  - "hourlyWithdrawUSD"

# Model settings
model_settings:
  sequence_length: 24
  forecast_horizon: 1

# Training settings
training_settings:
  train_test_split: 0.8
  validation_split: 0.2
  batch_size: 32
  max_epochs: 200
  early_stopping_patience: 20

# Hyperparameter optimization
hyperparameter_optimization:
  n_trials: 50
  learning_rate_range:
    min: 1e-5
    max: 1e-2

# Evaluation metrics
evaluation_metrics:
  - "mse"
  - "rmse"
  - "mae"
  - "r2"

# Logging and output
output_dir: "output/"
log_level: "INFO"
```

Adjust these settings according to your specific requirements and dataset characteristics.

## Usage

To run the APY prediction model:

1. Ensure your CSV files are in the `data/raw/` directory.
2. Adjust the `configs/config.yaml` file as needed.
3. Run the main script:
   ```
   python main.py
   ```

The script will process the data, perform feature engineering, conduct neural architecture search, optimize hyperparameters, train the final model, and output evaluation results.

## Example: Supply/Borrow APY Prediction

Let's walk through a detailed example of predicting supply and borrow APY for a DeFi protocol.

1. Data Preparation:
   Place your CSV files in the `data/raw/` directory. Ensure they have columns for timestamps, supply APY (lenderRate), borrow APY (borrowRate), and other relevant features like TVL, deposit amounts, etc.

2. Configuration:
   Adjust the `configs/config.yaml` file to match your data. For example:

   ```yaml
   target_columns:
     - "lenderRate"
     - "borrowRate"

   feature_columns:
     - "totalValueLockedUSD"
     - "totalBorrowBalanceUSD"
     - "totalDepositBalanceUSD"
     - "hourlyDepositUSD"
     - "hourlyRepayUSD"
     - "hourlyBorrowUSD"
     - "hourlyWithdrawUSD"

   model_settings:
     sequence_length: 24  # Use 24 hours of historical data
     forecast_horizon: 1  # Predict 1 hour ahead
   ```

3. Run the Model:
   Execute the main script:
   ```
   python main.py
   ```

4. Interpret the Results:
   The script will output evaluation metrics for both supply and borrow APY predictions. For example:

   ```
   Final evaluation results:
   Supply APY:
     MSE: 0.00015
     RMSE: 0.01225
     MAE: 0.00987
     R2: 0.8956

   Borrow APY:
     MSE: 0.00022
     RMSE: 0.01483
     MAE: 0.01156
     R2: 0.8732
   ```

   These metrics indicate how well the model is performing. Lower MSE, RMSE, and MAE values, and higher R2 values indicate better performance.

5. Using the Model for Predictions:
   After training, you can use the model to make predictions on new data:

   ```python
   # Assuming 'model' is your trained model and 'new_data' is a numpy array of shape (sequence_length, n_features)
   predictions = model.predict(new_data)
   supply_apy_prediction = predictions[0][0]
   borrow_apy_prediction = predictions[0][1]
   print(f"Predicted Supply APY: {supply_apy_prediction:.4f}")
   print(f"Predicted Borrow APY: {borrow_apy_prediction:.4f}")
   ```

## Customization

You can customize various aspects of the project:

1. Feature Engineering: Modify `src/feature_engineering/feature_transformer.py` to add or change feature transformations.
2. Model Architectures: Add new model architectures in `src/models/` and update `src/architecture_search/nas.py` to include them in the search.
3. Evaluation Metrics: Add new metrics in `src/evaluation/evaluator.py`.
4. Hyperparameter Space: Adjust the hyperparameter search space in `src/hyperparameter_optimization/optimizer.py`.

## Troubleshooting

Common issues and their solutions:

1. "FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/'"
   - Ensure that you have created the `data/raw/` directory and placed your CSV files there.

2. "KeyError: 'lenderRate'"
   - Check that your CSV files contain the columns specified in the `target_columns` and `feature_columns` in the config file.

3. "ValueError: Input contains NaN, infinity or a value too large for dtype('float32')"
   - Your data may contain invalid values. Add additional data cleaning steps in the `DataPreprocessor` class.

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

TODO: Work on the licence