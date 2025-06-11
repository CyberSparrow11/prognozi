import pandas as pd
import numpy as np
import os

data_paths = {
    "EURUSD": "financial_data/EURUSDX_itog.csv",
    "GSPC": "financial_data/GSPC_itog.csv",
    "AAPL": "financial_data/AAPL_itog.csv",
    "MSFT": "financial_data/MSFT_itog.csv"
}

output_dir = "financial_data_filtered"
os.makedirs(output_dir, exist_ok=True)

# Порог корреляции для отбора технических индикаторов
corr_threshold = 0.05

# Список технических индикаторов для проверки
tech_indicators = [
    'Volume', 'SMA_5', 'SMA_20', 'RSI_14', 'ATR_14', 'MACD', 'MACD_signal'
]

for asset_name, path in data_paths.items():
    df = pd.read_csv(path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)

    # Целевая переменная: сумма лог-доходностей за 5 дней вперед
    target = df['log_return'].rolling(window=5).sum().shift(-5)
    target = target.dropna()

    # Отбираем строки без пропусков в target
    df_valid = df.loc[target.index]

    # Убираем лишние признаки из OHLC, оставляем Close и volume
    # Также сохраняем технические индикаторы
    features = df_valid.drop(columns=['Open', 'High', 'Low', 'log_return'])

    # Корреляцию считаем только для технических индикаторов
    corr = features[tech_indicators].corrwith(target).abs()
    print(f"\n{asset_name} - корреляция признаков с target:")
    print(corr)

    # Отбираем технические индикаторы с корреляцией >= порога
    selected_tech = corr[corr >= corr_threshold].index.tolist()

    # Итоговый список признаков: Close, volume + выбранные технические индикаторы
    final_features = ['Close'] + selected_tech

    print(f"Оставляем признаки: {final_features}")

    df_filtered = features[final_features].copy()
    # Сохраняем очищенный датасет в новый CSV
    output_path = os.path.join(output_dir, f"{asset_name}_filtered.csv")
    df_filtered.to_csv(output_path)
    print(f"Сохранено: {output_path}")
