import yfinance as yf
import pandas as pd
import time
from pathlib import Path

# Создаем папку для сохранения данных (если её нет)
data_folder = Path("financial_data")
data_folder.mkdir(exist_ok=True)

# Список тикеров и их описаний для именования файлов
tickers = {
    "^GSPC": "S&P500",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "EURUSD=X": "EURUSD"
}

# Параметры загрузки
start_date = "2010-01-01"
end_date = "2025-05-31"


def download_and_save_data():
    for ticker, name in tickers.items():
        try:
            print(f"Загружаем данные для {name} ({ticker})...")

            # Загрузка данных
            data = yf.download(ticker,
                               start=start_date,
                               end=end_date,
                               progress=False)

            # Проверка: если данных нет — пропускаем
            if data.empty:
                print(f"Нет данных для {ticker}")
                continue

            # Сброс индекса, чтобы 'Date' стал обычным столбцом
            data.reset_index(inplace=True)

            # Добавляем столбец с тикером
            data['Ticker'] = ticker

            # Сохраняем в CSV
            filename = data_folder / f"{name}_{ticker.replace('^', '').replace('=', '')}.csv"
            data.to_csv(filename, encoding='utf-8', index=False)

            print(f"Данные сохранены в {filename}")
            print(f"Загружено {len(data)} строк\n")

            # Пауза между запросами
            time.sleep(2)

        except Exception as e:
            print(f"Ошибка при загрузке {ticker}: {str(e)}")
            continue


if __name__ == "__main__":
    download_and_save_data()
    print("Загрузка всех данных завершена!")
