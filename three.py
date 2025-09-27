import json
import pandas as pd
from prophet import Prophet
from datetime import datetime
import matplotlib.pyplot as plt

INPUT_FILE = "sentiment_output.json"
OUTPUT_FILE = "sentiment_output.json"

# ----------------------------
# Load JSON
# ----------------------------
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

if not data:
    raise ValueError("No data found in sentiment_output.json")

# ----------------------------
# Prepare DataFrame for Prophet
# ----------------------------
df = pd.DataFrame(data)
df["ds"] = pd.to_datetime(df["published"]).dt.tz_localize(None)  # remove timezone
df["y"] = pd.to_numeric(df["sentiment_score"], errors="coerce")

# Drop rows with missing sentiment
df = df.dropna(subset=["y"])
if df.empty:
    raise ValueError("No valid sentiment scores for training.")

# ----------------------------
# Train Prophet
# ----------------------------
model = Prophet(daily_seasonality=True, weekly_seasonality=False)
model.fit(df[["ds", "y"]])

# ----------------------------
# Forecast +24h and +48h
# ----------------------------
future = model.make_future_dataframe(periods=2, freq="D")
forecast = model.predict(future)

# Get last 2 predictions
pred_24h = forecast.iloc[-2]["yhat"]
pred_48h = forecast.iloc[-1]["yhat"]

# ----------------------------
# Update JSON
# ----------------------------
for entry in data:
    entry["forecast_24h"] = round(pred_24h, 4)
    entry["forecast_48h"] = round(pred_48h, 4)

with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=4)

print(f"Updated forecasts saved to {OUTPUT_FILE}")
print(f"Forecast +24h: {pred_24h:.4f}, +48h: {pred_48h:.4f}")

# ----------------------------
# Visualization
# ----------------------------

# Plot 1: Actual vs Forecast
plt.figure(figsize=(12, 6))
plt.plot(df["ds"], df["y"], label="Actual Sentiment", marker="o")
plt.plot(forecast["ds"], forecast["yhat"], label="Forecast", linestyle="--")
plt.fill_between(
    forecast["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    color="lightblue",
    alpha=0.4,
    label="Uncertainty Interval",
)
plt.title("Sentiment Forecast (+24h, +48h)")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.legend()
plt.tight_layout()
plt.show()

