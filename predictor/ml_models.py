import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import os
from tensorflow.keras.regularizers import l2
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# Define models and scaler as global variables
rf_model_class, rf_model_reg, deep_model, lstm_model, scaler = None, None, None, None, StandardScaler()
X_test_class, y_test_class, X_test_reg, y_test_reg = None, None, None, None  # Global test sets

def load_models():
    global rf_model_class, rf_model_reg, deep_model, lstm_model, scaler
    global X_test_class, y_test_class, X_test_reg, y_test_reg

    if os.path.exists('scaler.joblib'):
        scaler = joblib.load('scaler.joblib')  # Load the saved scaler
        rf_model_class = joblib.load('rf_model_class.joblib')
        rf_model_reg = joblib.load('rf_model_reg.joblib')
        deep_model = load_model('deep_model.keras')
        lstm_model = load_model('lstm_model.keras')
    else:
        # Call train_models to fit and save models if not exist
        #train_models(X_train_class, y_train_class, X_train_reg, y_train_reg)
        load_and_train_models()

def load_and_train_models():
    global rf_model_class, rf_model_reg, deep_model, lstm_model, scaler
    global X_test_class, y_test_class, X_test_reg, y_test_reg  # Test data as global

    # Load data
    df = pd.read_csv('data.csv', delimiter=',')
    df.columns = df.columns.str.replace('"', '')

    # Data Preprocessing
    df['updated_at'] = pd.to_datetime(df['updated_at'])
    df['time_diff'] = df['updated_at'].diff().dt.total_seconds().fillna(0)
    battery_threshold_volt = 11.9
    df['target'] = df['volt'].apply(lambda x: 1 if x > battery_threshold_volt else 0)
    df['battery_lifetime'] = np.random.uniform(0, 5, size=len(df))  # synthetic data for demonstration

    # Features and Target
    features = ['volt', 'arus', 'baterai', 'suhu', 'kelembaban', 'time_diff']
    X = df[features]
    y_classification = df['target']
    y_regression = df['battery_lifetime']

    # Train-Test Split
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.3, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.3, random_state=42)

    # Scale Features
    X_train_class = scaler.fit_transform(X_train_class)
    X_test_class = scaler.transform(X_test_class)

    # Check if models exist, if not train them
    if os.path.exists('rf_model_class.joblib'):
        rf_model_class = joblib.load('rf_model_class.joblib')
        rf_model_reg = joblib.load('rf_model_reg.joblib')
        deep_model = load_model('deep_model.keras')  # Directly load the pre-trained model
        lstm_model = load_model('lstm_model.keras') 
        
    # Train Random Forest Classifier (L2 is inherently handled in RF as a complex regularized ensemble method)
    rf_model_class = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_class.fit(X_train_class, y_train_class)

    # Train Random Forest Regressor
    rf_model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_reg.fit(X_train_reg, y_train_reg)

    # Train Deep Learning Model with L2 Regularization
    deep_model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train_class.shape[1],)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
    ])
    deep_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    deep_history = deep_model.fit(X_train_class, y_train_class, epochs=50, batch_size=16, validation_split=0.2)

    # Prepare data for LSTM
    generator = TimeseriesGenerator(X_train_reg.values, y_train_reg.values, length=5, batch_size=1)
    val_generator = TimeseriesGenerator(X_test_reg.values, y_test_reg.values, length=5, batch_size=1)

    # Train LSTM Model with L2 Regularization and Dropout
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', kernel_regularizer=l2(0.01), input_shape=(5, X_train_reg.shape[1])))
    lstm_model.add(Dropout(0.2))  # Make sure to have this line after the import
    lstm_model.add(Dense(1, kernel_regularizer=l2(0.01)))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_history = lstm_model.fit(generator, validation_data=val_generator, epochs=100)

    # Save models
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(rf_model_class, 'rf_model_class.joblib')
    joblib.dump(rf_model_reg, 'rf_model_reg.joblib')
    deep_model.save('deep_model.keras')  # Save Deep Learning model in .keras format
    lstm_model.save('lstm_model.keras')

    # Save plots
    save_path = 'battery_prediction/static/images'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Plot and save Random Forest Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot([0, 1], [rf_model_class.score(X_test_class, y_test_class), rf_model_class.score(X_test_class, y_test_class)], label='Random Forest Accuracy')
    plt.title('Random Forest Model Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{save_path}/rf_accuracy.png')

    # Plot and save Deep Learning Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(deep_history.history['accuracy'], label='Deep Learning Train Accuracy')
    plt.plot(deep_history.history['val_accuracy'], label='Deep Learning Val Accuracy')
    plt.title('Deep Learning Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{save_path}/deep_learning_accuracy.png')

    # Save LSTM Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(lstm_history.history['loss'], label='LSTM Train Loss')
    plt.plot(lstm_history.history['val_loss'], label='LSTM Val Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_path}/lstm_loss.png')

def evaluate_models():
    global X_test_class, y_test_class, X_test_reg, y_test_reg

    # Akurasi Model Random Forest Classifier
    rf_class_predictions = rf_model_class.predict(X_test_class)
    rf_class_accuracy = accuracy_score(y_test_class, rf_class_predictions)

    # MSE (Mean Squared Error) Model Random Forest Regressor
    rf_reg_predictions = rf_model_reg.predict(X_test_reg)
    rf_reg_mse = mean_squared_error(y_test_reg, rf_reg_predictions)

    # Akurasi Model Deep Learning (Menggunakan X_test_class dan y_test_class)
    deep_learning_predictions = deep_model.predict(X_test_class)
    deep_learning_accuracy = accuracy_score(y_test_class, (deep_learning_predictions > 0.5).astype(int))

    # Return nilai akurasi
    return {
        "rf_class_accuracy": rf_class_accuracy,
        "rf_reg_mse": rf_reg_mse,
        "deep_learning_accuracy": deep_learning_accuracy
    }


load_models()  # Call this when initializing the application
