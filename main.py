import yaml
from src.data_processing.data_loader import DataLoader
from src.data_processing.data_preprocessor import DataPreprocessor
from src.feature_engineering.feature_transformer import FeatureTransformer
from src.architecture_search.nas import NeuralArchitectureSearch
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.hyperparameter_optimization.optimizer import HyperparameterOptimizer

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config('configs/config.yaml')

    # Load and preprocess data
    data_loader = DataLoader(config['data_dir'])
    data_dict = data_loader.load_csv_files()
    merged_data = data_loader.merge_dataframes(list(data_dict.values()))

    preprocessor = DataPreprocessor(config['target_columns'])
    preprocessed_data = preprocessor.preprocess(merged_data)
    train_data, test_data = preprocessor.split_data(preprocessed_data)

    # Feature engineering
    feature_transformer = FeatureTransformer(config['feature_columns'])
    train_data_transformed = feature_transformer.transform(train_data)
    test_data_transformed = feature_transformer.transform(test_data)

    # Prepare input data for models
    X_train = train_data_transformed[feature_transformer.get_feature_names()].values
    y_train = train_data_transformed[config['target_columns']].values
    X_test = test_data_transformed[feature_transformer.get_feature_names()].values
    y_test = test_data_transformed[config['target_columns']].values

    # Neural Architecture Search
    nas = NeuralArchitectureSearch(input_shape=(X_train.shape[1], 1), output_shape=len(config['target_columns']))
    best_model = nas.search(X_train, y_train, X_test, y_test)

    # Hyperparameter Optimization
    def objective(trial):
        lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        best_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
        
        trainer = ModelTrainer(best_model, epochs=50, batch_size=32)
        history = trainer.train(X_train, y_train, X_test, y_test)
        
        return history.history['val_loss'][-1]

    optimizer = HyperparameterOptimizer(objective)
    best_params = optimizer.optimize()

    # Final training with best hyperparameters
    best_model.compile(optimizer=tf.keras.optimizers.Adam(best_params['learning_rate']), loss='mse')
    trainer = ModelTrainer(best_model, epochs=100, batch_size=32)
    trainer.train(X_train, y_train, X_test, y_test)

    # Evaluation
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(best_model, X_test, y_test)

    print("Final evaluation results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()