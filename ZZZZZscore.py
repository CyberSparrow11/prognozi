import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Загрузка данных
df = pd.read_csv("financial_data/S&P500_GSPC.csv",
                 index_col="Date",
                 parse_dates=True)

# 2. Обработка выбросов с помощью Z-score
# Вычисляем Z-показатели для Volume
volume_z = np.abs(stats.zscore(df["Volume"]))

# Задаем порог для выбросов (3 стандартных отклонения)
threshold = 3
df_clean = df[volume_z < threshold]

# 4. Сравнительная статистика
print("=== Сравнительная статистика ===")
print(f"Исходное количество данных: {len(df)}")
print(f"Оставшееся количество данных: {len(df_clean)}")
print(
    f"Удалено выбросов: {len(df) - len(df_clean)} ({((len(df)-len(df_clean))/len(df))*100:.2f}%)"
)

# 5. Дополнительная визуализация временного ряда
plt.figure(figsize=(14, 6))
plt.plot(df["Volume"], alpha=0.5, label='Исходные данные')
plt.plot(df_clean["Volume"], label='После очистки')
plt.title("Сравнение Volume до и после обработки выбросов")
plt.ylabel("Объем торгов")
plt.xlabel("Дата")
plt.legend()
plt.grid(True)
plt.show()

df_clean.to_csv("financial_data/S&P500_GSPC_cleaned.csv")