import os

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


# Function to create the GRU model
def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.GRU(96, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.GRU(352, return_sequences=True),
        tf.keras.layers.GRU(96, return_sequences=True),
        tf.keras.layers.GRU(416, return_sequences=True),
        tf.keras.layers.GRU(512),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    return model

# Function to train and save the model
def train_and_save_model(df, feature_columns, target_column, save_path):
    # Remove rows where target_column has NaN values
    df = df.dropna(subset=[target_column])

    # Prepare the data
    features = df[feature_columns]
    target = df[target_column]

    # Split the data: 97% training and 3% testing
    split_index = int(0.97 * len(df))
    X_train, X_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the smaller GRU model with regularization
    model = create_model(input_shape=(1, X_train.shape[1]))

    # Reshape data for GRU input
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Define early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=100, validation_split=0.2, verbose=1, callbacks=[early_stopping])

    # Save the trained model with .keras extension
    model.save(save_path)
    print(f"Model saved to {save_path}")

# Root directories
data_root = './data'
model_root = './model'

# Feature columns for training
feature_columns = [
    'totalValueLockedUSD', 'totalBorrowBalanceUSD',
    'totalDepositBalanceUSD', 'hourlyDepositUSD',
    'hourlyRepayUSD', 'hourlyBorrowUSD',
    'hourlyWithdrawUSD'
]

# Traverse the directory tree
for root, dirs, files in os.walk(data_root):
    for file in files:
        if file.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(root, file)
            
            # Read the CSV into a pandas dataframe
            df = pd.read_csv(file_path)
            
            # Remove rows with NaN values in borrowRate or lenderRate
            df = df.dropna(subset=['borrowRate', 'lenderRate'])
            if df.empty:
                    continue
            # Extract protocol and asset information
            protocol = os.path.basename(root)
            asset = df['inputToken'].iloc[0]
            print(protocol,asset)
            # Create model save paths with .keras extension
            model_folder = os.path.join(model_root, protocol)
            os.makedirs(model_folder, exist_ok=True)
            borrow_rate_model_path = os.path.join(model_folder, f"{asset}_borrow_rate.keras")
            lender_rate_model_path = os.path.join(model_folder, f"{asset}_lender_rate.keras")
            
            # Train and save model for borrowRate
            train_and_save_model(df, feature_columns, 'borrowRate', borrow_rate_model_path)
            
            # Train and save model for lenderRate
            train_and_save_model(df, feature_columns, 'lenderRate', lender_rate_model_path)
