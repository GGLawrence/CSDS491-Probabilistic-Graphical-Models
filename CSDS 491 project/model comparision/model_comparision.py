import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# Read the CSV file and drop rows with NaN values
data = pd.read_csv("USD_EUR_different_models.csv").dropna()
#data = pd.read_csv("USD_JPY_different_models.csv").dropna()
#data = pd.read_csv("USD_CNY_different_models.csv").dropna()

# Convert date column to datetime format
data["date"] = pd.to_datetime(data["date"])

# Extract columns
date = data["date"]
actual_close_price = data["actual_close_price"]
HMM = data["HMM"]
GMM_HMM = data["GMM_HMM"]
LSTM = data["LSTM"]
XGBoost = data["XGBoost"]

# Create an interactive line chart
fig = go.Figure()

fig.add_trace(go.Scatter(x=date, y=actual_close_price, mode="lines", name="Actual Close Price", line=dict(color="black", dash="dash", width=2)))
fig.add_trace(go.Scatter(x=date, y=HMM, mode="lines", name="HMM"))
fig.add_trace(go.Scatter(x=date, y=GMM_HMM, mode="lines", name="GMM-HMM"))
fig.add_trace(go.Scatter(x=date, y=LSTM, mode="lines", name="LSTM"))
fig.add_trace(go.Scatter(x=date, y=XGBoost, mode="lines", name="XGBoost"))

# Configure the chart layout
fig.update_layout(title="USD/EUR Close Price for Different Models", xaxis_title="Date", yaxis_title="Close Price")
#fig.update_layout(title="USD/JPY Close Price for Different Models", xaxis_title="Date", yaxis_title="Close Price")
#fig.update_layout(title="USD/CNY Close Price for Different Models", xaxis_title="Date", yaxis_title="Close Price")

# Calculate MAE, RMSE, and R-squared for the last three models
models = {"HMM": HMM,"GMM-HMM": GMM_HMM, "LSTM": LSTM, "XGBoost": XGBoost}

for model_name, model_data in models.items():
    mae = mean_absolute_error(actual_close_price, model_data)
    rmse = np.sqrt(mean_squared_error(actual_close_price, model_data))
    r2 = r2_score(actual_close_price, model_data)
    print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R-squared: {r2:.4f}")

# Show the plot
fig.show()


# Data for the chart
models = ['HMM', 'GMM-HMM', 'LSTM', 'XGBoost']
currency_pairs = ['USD/CNY', 'USD/JPY', 'USD/EUR']
metrics = ['MAE', 'RMSE', 'R²']
performance_data = [
    [
        [0.0257, 0.0354, 0.9448],
        [0.9151, 1.2847, 0.9483],
        [0.0049, 0.0064, 0.9497],
    ],
    [
        [0.0390, 0.0478, 0.8994],
        [1.5990, 1.9573, 0.8800],
        [0.0069, 0.0081, 0.9195],
    ],
    [
        [0.1043, 0.1283, 0.2743],
        [4.0745, 5.7972, -0.0527],
        [0.0185, 0.0203, 0.4958],
    ],
    [
        [0.0320, 0.0442, 0.9138],
        [1.4694, 2.0117, 0.8732],
        [0.0078, 0.0106, 0.8617],
    ],
]

n_models = len(models)
n_currency_pairs = len(currency_pairs)
n_metrics = len(metrics)

# Plotting the chart
bar_width = 0.1
fig, axes = plt.subplots(1, n_currency_pairs, figsize=(18, 6), sharey=True)

for k, ax in enumerate(axes):
    for i, model_performance in enumerate(performance_data):
        metric_values = model_performance[k]
        x_positions = np.arange(n_metrics) + i * bar_width
        ax.bar(x_positions, metric_values, width=bar_width, label=models[i])

    # Add title, labels, and legend
    ax.set_title("Model Performance for " + currency_pairs[k])
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Values")
    ax.set_xticks(np.arange(n_metrics))
    ax.set_xticklabels(metrics)
    ax.legend()

# Save the plot
plt.savefig("model_performance_comparison_separate_charts.png")

# Show the plot
plt.show()
