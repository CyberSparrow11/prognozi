import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json  # обязательно для загрузки истории

# Создание папок
os.makedirs("results/plots", exist_ok=True)

# Загрузка таблицы сравнения
comparison_df_path = "results/model_comparison.csv"
if os.path.exists(comparison_df_path):
    comparison_df = pd.read_csv(comparison_df_path)
else:
    comparison_df = pd.DataFrame(
        columns=["Model", "Ticker", "MAE", "RMSE", "R2"])

tickers = ['EURUSD', 'GSPC', 'AAPL', 'MSFT']

for ticker in tickers:
    # --- Графики "Истинное vs Предсказанное" ---
    trad_path = f"results/{ticker}_traditional_predictions.csv"
    if os.path.exists(trad_path):
        df_trad = pd.read_csv(trad_path)
        plt.figure(figsize=(10, 5))
        plt.plot(df_trad['Date'], df_trad['Actual'], label='Actual')
        plt.plot(df_trad['Date'],
                 df_trad['LR_Predicted'],
                 label='LinearRegression')
        plt.plot(df_trad['Date'],
                 df_trad['RF_Predicted'],
                 label='RandomForest')
        plt.title(f"{ticker} — Истинное vs Предсказанное (LR/RF)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"results/plots/{ticker}_actual_vs_pred_lr_rf.png")
        plt.close()

    lstm_path = f"results/{ticker}_lstm_predictions.csv"
    if os.path.exists(lstm_path):
        df_lstm = pd.read_csv(lstm_path)
        plt.figure(figsize=(10, 5))
        plt.plot(df_lstm['Actual'], label='Actual')
        plt.plot(df_lstm['LSTM_Predicted'], label='LSTM')
        plt.title(f"{ticker} — Истинное vs Предсказанное (LSTM)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/plots/{ticker}_actual_vs_pred_lstm.png")
        plt.close()

    # --- Матрица ошибок ---
    if os.path.exists(trad_path):
        df = pd.read_csv(trad_path)
        y_true_cls = (df['Actual'] > 0).astype(int)
        y_pred_cls = (df['RF_Predicted'] > 0).astype(int)
        conf_matrix = pd.crosstab(y_true_cls,
                                  y_pred_cls,
                                  rownames=['Actual'],
                                  colnames=['Predicted'])

        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{ticker} — Матрица ошибок (RF направление)")
        plt.savefig(f"results/plots/{ticker}_confusion_matrix.png")
        plt.close()

    # --- График потерь LSTM ---
    history_path = f"results/histories/{ticker}_lstm_history.json"
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history_dict = json.load(f)

        if 'loss' in history_dict and 'val_loss' in history_dict:
            plt.figure(figsize=(8, 5))
            plt.plot(history_dict['loss'], label='Train Loss')
            plt.plot(history_dict['val_loss'], label='Val Loss')
            plt.title(f"{ticker} — График потерь LSTM")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"results/plots/{ticker}_lstm_loss_curve.png")
            plt.close()

# --- Таблица метрик по моделям ---
agg_df = comparison_df.groupby("Model").mean(numeric_only=True).reset_index()
agg_df.to_csv("results/plots/aggregated_metrics.csv", index=False)

print("Визуализации успешно созданы и сохранены в папке results/plots.")
