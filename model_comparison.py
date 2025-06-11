import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os


# Загрузка данных из CSV с индексом Date
def load_data(data_paths):
    dfs = {}
    for name, path in data_paths.items():
        df = pd.read_csv(path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        dfs[name] = df
    return dfs


data_paths = {
    "EURUSD": "financial_data_filtered/EURUSD_filtered.csv",
    "GSPC": "financial_data_filtered/GSPC_filtered.csv",
    "AAPL": "financial_data_filtered/AAPL_filtered.csv",
    "MSFT": "financial_data_filtered/MSFT_filtered.csv"
}
data = load_data(data_paths)


# Инжиниринг признаков
def add_features(df):
    df = df.copy()

    # Лог-доходность
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Lag-фичи лог-доходности
    for lag in range(1, 6):
        df[f'log_return_lag_{lag}'] = df['log_return'].shift(lag)

    # Скользящие средние и волатильность
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()

    # Bollinger Bands
    df['BB_upper'] = df['SMA_20'] + 2 * df['STD_20']
    df['BB_lower'] = df['SMA_20'] - 2 * df['STD_20']

    # Стохастик %K
    if {'Low', 'High'}.issubset(df.columns):
        low_min = df['Low'].rolling(14).min()
        high_max = df['High'].rolling(14).max()
        df['%K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)

    # Календарные признаки
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    return df


# Подготовка данных для классических моделей (LR, RF)
def prepare_data_for_ml(df):
    df = add_features(df)

    # Целевая: доходность за 5 дней вперёд (лог-сумма)
    df['target'] = df['log_return'].rolling(5).sum().shift(-5)

    df.dropna(inplace=True)

    X = df.drop(['target', 'log_return'], axis=1)
    y = df['target']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                        y,
                                                        test_size=0.2,
                                                        shuffle=False)

    return X_train, X_test, y_train, y_test, scaler


# Подготовка данных для LSTM
def prepare_data_for_lstm(df, n_steps=30):
    df = add_features(df)

    df['target'] = df['log_return'].rolling(5).sum().shift(-5)

    df.dropna(inplace=True)

    X = df.drop(['target', 'log_return'], axis=1)
    y = df['target']

    x_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X)

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

    X_lstm, y_lstm = [], []
    for i in range(n_steps, len(X_scaled)):
        X_lstm.append(X_scaled[i - n_steps:i, :])
        y_lstm.append(y_scaled[i, 0])

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    split_idx = int(0.8 * len(X_lstm))
    X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
    y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]

    return X_train, X_test, y_train, y_test, x_scaler, y_scaler


# Обучение и оценка линейной регрессии
def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {'model': model, 'mae': mae, 'rmse': rmse, 'r2': r2}


# Обучение и оценка случайного леса
def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {'model': model, 'mae': mae, 'rmse': rmse, 'r2': r2}


# Обучение и оценка LSTM
def train_lstm(X_train,
               X_test,
               y_train,
               y_test,
               y_scaler,
               epochs=50,
               batch_size=32):
    model = Sequential([
        LSTM(64,
             return_sequences=True,
             input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=7,
                                   restore_best_weights=True)

    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping],
                        verbose=1)

    y_pred_scaled = model.predict(X_test).flatten()
    y_test_inv = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = y_scaler.inverse_transform(y_pred_scaled.reshape(
        -1, 1)).flatten()

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)
    directional_accuracy = np.mean(np.sign(y_test_inv) == np.sign(y_pred_inv))

    return {
        'model': model,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'directional_accuracy': directional_accuracy,
        'history': history,
        'y_test': y_test_inv,
        'y_pred': y_pred_inv
    }


# Сравнение моделей
def compare_models(results):
    comparison = pd.DataFrame(
        {
            'Linear Regression':
            [results['lr']['mae'], results['lr']['rmse'], results['lr']['r2']],
            'Random Forest':
            [results['rf']['mae'], results['rf']['rmse'], results['rf']['r2']],
            'LSTM': [
                results['lstm']['mae'], results['lstm']['rmse'],
                results['lstm']['r2']
            ]
        },
        index=['MAE', 'RMSE', 'R2'])
    return comparison


# Основной цикл для каждого актива
for asset_name, df in data.items():
    print(f"\nАнализ актива: {asset_name}")

    # Подготовка данных для классических моделей
    X_train_ml, X_test_ml, y_train_ml, y_test_ml, scaler_ml = prepare_data_for_ml(
        df.copy())

    # Обучение линейной регрессии
    print("Обучение линейной регрессии...")
    lr_results = train_linear_regression(X_train_ml, X_test_ml, y_train_ml,
                                         y_test_ml)

    # Обучение случайного леса
    print("Обучение случайного леса...")
    rf_results = train_random_forest(X_train_ml, X_test_ml, y_train_ml,
                                     y_test_ml)

    # Подготовка данных и обучение LSTM
    print("Обучение LSTM...")
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, x_scaler_lstm, y_scaler_lstm = prepare_data_for_lstm(
        df.copy())
    lstm_results = train_lstm(X_train_lstm,
                              X_test_lstm,
                              y_train_lstm,
                              y_test_lstm,
                              y_scaler_lstm,
                              epochs=30)

    # Сравнение моделей
    results = {'lr': lr_results, 'rf': rf_results, 'lstm': lstm_results}

    comparison = compare_models(results)
    print(f"\nСравнение моделей для {asset_name}:")
    print(comparison)
    print(
        f"Directional Accuracy (LSTM): {lstm_results['directional_accuracy']:.3f}"
    )

    # Сохранение результатов
    if not os.path.exists('results'):
        os.makedirs('results')
    comparison.to_csv(f'results/{asset_name}_model_comparison.csv')

    # Визуализация прогнозов (на примере случайного леса)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_ml.index, y_test_ml, label='Фактические значения')
    plt.plot(y_test_ml.index,
             rf_results['model'].predict(X_test_ml),
             label='Прогноз (Random Forest)')
    plt.title(f'Прогноз доходности для {asset_name} (Random Forest)')
    plt.xlabel('Дата')
    plt.ylabel('Логарифмическая доходность')
    plt.legend()
    plt.savefig(f'results/{asset_name}_rf_prediction.png')
    plt.close()

    # Визуализация истории обучения LSTM
    plt.figure(figsize=(12, 6))
    plt.plot(lstm_results['history'].history['loss'],
             label='Ошибка на обучении')
    plt.plot(lstm_results['history'].history['val_loss'],
             label='Ошибка на валидации')
    plt.title(f'История обучения LSTM для {asset_name}')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(f'results/{asset_name}_lstm_training.png')
    plt.close()

print("\nАнализ завершен. Результаты сохранены в папке 'results'.")
