import pandas as pd
import numpy as np
import pandas_ta as ta

# Загружаем данные
df = pd.read_csv("financial_data/S&P500_GSPC_cleaned.csv",
                 index_col="Date",
                 parse_dates=True)

price_col = 'Close'

# Копируем исходный df, чтобы не изменять оригинал
df_features = df.copy()

# Логарифмическая доходность
df_features['log_return'] = np.log(df_features[price_col] /
                                   df_features[price_col].shift(1))

# Скользящие средние SMA 5 и SMA 20
df_features['SMA_5'] = ta.sma(df_features[price_col], length=5)
df_features['SMA_20'] = ta.sma(df_features[price_col], length=20)

# RSI (14 периодов)
df_features['RSI_14'] = ta.rsi(df_features[price_col], length=14)

# ATR (14 периодов)
df_features['ATR_14'] = ta.atr(df_features['High'],
                               df_features['Low'],
                               df_features['Close'],
                               length=14)

# MACD
macd = ta.macd(df_features[price_col])
df_features['MACD'] = macd['MACD_12_26_9']
df_features['MACD_signal'] = macd['MACDs_12_26_9']

# Удаляем строки с NaN
df_features = df_features.dropna()

# Выбираем только нужные колонки для сохранения
columns_to_save = [
    'Close', 'High', 'Low', 'Open', 'Volume', 'log_return', 'SMA_5', 'SMA_20',
    'RSI_14', 'ATR_14', 'MACD', 'MACD_signal'
]

# Сохраняем в новый файл
df_features[columns_to_save].to_csv("financial_data/GSPC_itog.csv")

print("Файл с признаками сохранён: financial_data/в.csv")
