import os
import logging
import yaml
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from src.data_processing.data_preprocessor import DataPreprocessor
from src.feature_engineering.feature_transformer import FeatureTransformer
from src.architecture_search.nas import NeuralArchitectureSearch
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.hyperparameter_optimization.optimizer import HyperparameterOptimizer

logging.basicConfig(level=logging.INFO)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_and_train_model(file_path: str, output_dir: str, config: dict, pbar: tqdm) -> None:
    # Define main steps
    steps = [
        "Load and Preprocess Data",
        "Feature Engineering",
        "Neural Architecture Search",
        "Hyperparameter Optimization",
        "Final Model Training",
        "Model Evaluation",
        "Save Model and Results"
    ]

    with tqdm(total=len(steps), desc=f"Processing {os.path.basename(file_path)}", leave=False) as file_pbar:
        # Load and preprocess data
        df = pd.read_csv(file_path)
        preprocessor = DataPreprocessor(config['target_columns'])
        preprocessed_data = preprocessor.preprocess(df)
        train_data, test_data = preprocessor.split_data(preprocessed_data)
        file_pbar.update(1)
        pbar.set_postfix({"Current Step": steps[0]})

        # Feature engineering
        feature_transformer = FeatureTransformer(config['feature_columns'])
        train_data_transformed = feature_transformer.transform(train_data)
        test_data_transformed = feature_transformer.transform(test_data)
        X_train = train_data_transformed[feature_transformer.get_feature_names()].values
        y_train = train_data_transformed[config['target_columns']].values
        X_test = test_data_transformed[feature_transformer.get_feature_names()].values
        y_test = test_data_transformed[config['target_columns']].values
        file_pbar.update(1)
        pbar.set_postfix({"Current Step": steps[1]})

        # Neural Architecture Search
        nas = NeuralArchitectureSearch(input_shape=(X_train.shape[1], 1), output_shape=len(config['target_columns']))
        best_model = nas.search(X_train, y_train, X_test, y_test)
        file_pbar.update(1)
        pbar.set_postfix({"Current Step": steps[2]})

        # Hyperparameter Optimization
        def objective(trial):
            lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
            best_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
            trainer = ModelTrainer(best_model, epochs=50, batch_size=32)
            history = trainer.train(X_train, y_train, X_test, y_test)
            return history.history['val_loss'][-1]

        optimizer = HyperparameterOptimizer(objective)
        best_params = optimizer.optimize()
        file_pbar.update(1)
        pbar.set_postfix({"Current Step": steps[3]})

        # Final training with best hyperparameters
        best_model.compile(optimizer=tf.keras.optimizers.Adam(best_params['learning_rate']), loss='mse')
        trainer = ModelTrainer(best_model, epochs=100, batch_size=32)
        trainer.train(X_train, y_train, X_test, y_test)
        file_pbar.update(1)
        pbar.set_postfix({"Current Step": steps[4]})

        # Evaluation
        evaluator = ModelEvaluator()
        results = evaluator.evaluate(best_model, X_test, y_test)
        file_pbar.update(1)
        pbar.set_postfix({"Current Step": steps[5]})

        # Save the model
        relative_path = os.path.relpath(file_path, config['data_dir'])
        model_save_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.h5')
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        best_model.save(model_save_path)

        # Save evaluation results
        results_save_path = os.path.splitext(model_save_path)[0] + '_results.yaml'
        with open(results_save_path, 'w') as f:
            yaml.dump(results, f)

        file_pbar.update(1)
        pbar.set_postfix({"Current Step": steps[6]})

    logging.info(f"Model and results saved for {file_path}")

def main():
    # Load configuration
    config = load_config('configs/config.yaml')
    logging.info("Configuration loaded")

    # Create output directory
    output_dir = os.path.join(config['output_dir'], 'models')
    os.makedirs(output_dir, exist_ok=True)

    # Get total number of CSV files
    total_files = sum(1 for root, dirs, files in os.walk(config['data_dir']) 
                      for file in files if file.endswith('.csv'))

    # Process each CSV file
    with tqdm(total=total_files, desc="Overall Progress") as pbar:
        for root, dirs, files in os.walk(config['data_dir']):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    logging.info(f"Processing file: {file_path}")
                    process_and_train_model(file_path, output_dir, config, pbar)
                    pbar.update(1)

    logging.info("All models trained and saved successfully")

if __name__ == "__main__":
    main()