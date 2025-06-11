import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Пример: проверка качества данных DataFrame `df`
# Загрузка данных
df = pd.read_csv('financial_data_filtered/GSPC_filtered.csv',
                 parse_dates=['Date'],
                 index_col='Date')

df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
# Быстрый просмотр структуры
print("\nПервые строки датафрейма:")
print(df.head())

# 1. Проверка пропусков и дубликатов
print("Пропущенные значения по колонкам:\n", df.isna().sum())
print("\nДубликаты строк:", df.duplicated().sum())

# 2. Проверка на неадекватные значения
print("\nОтрицательные объемы торгов:", (df['Volume'] < 0).sum())

# 3. Статистика по числовым признакам
print("\nОписание данных:\n", df.describe())

# 4. Визуализация распределения
df.hist(bins=50, figsize=(12, 8))
plt.suptitle("Гистограммы распределения признаков")
plt.tight_layout()
plt.show()

# 5. Проверка выбросов (z-score > 3)
z_scores = stats.zscore(df.select_dtypes(include='number'))
outliers = (abs(z_scores) > 3).sum(axis=0)
print("\nКоличество выбросов по признакам:\n", outliers)

# 6. Матрица корреляции
df_corr = df.copy()
exclude_cols = ['Ticker']
df_corr = df_corr.drop(
    columns=[col for col in exclude_cols if col in df_corr.columns])

plt.figure(figsize=(8, 6))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Матрица корреляции информативных признаков")
plt.show()
