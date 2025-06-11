import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем данные с признаками (предположим, что log_return уже есть)
df = pd.read_csv("financial_data/EURUSDX_itog.csv",
                 index_col="Date",
                 parse_dates=True)

# Гистограмма + плотность распределения
plt.figure(figsize=(10, 6))
sns.histplot(df['log_return'], bins=50, kde=True, color='blue')

plt.title('Гистограмма и плотность распределения логарифмической доходности')
plt.xlabel('log_return')
plt.ylabel('Частота')
plt.grid(True)
plt.show()