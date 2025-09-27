import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from prophet import Prophet
import matplotlib.pyplot as plt

# === Step 1: Load Sentiment JSON ===
def load_sentiment_json(file_path="sentiment_output.json"):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

# === Step 2: Extract Date Range ===
def get_date_range(data):
    dates = [
        datetime.fromisoformat(article["published"].replace("Z", "+00:00")).date()
        for article in data
    ]
    return min(dates), max(dates)

# === Step 3: Fetch YFinance Data (only Close) ===
def fetch_yfinance_data(ticker, start_date, end_date):
    stock_data = yf.download(
        ticker,
        start=start_date,
        end=end_date + timedelta(days=1),  # inclusive
        auto_adjust=False,
        progress=False,
    )

    # Flatten MultiIndex if present
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = ["_".join([str(c) for c in col if c]).strip() for col in stock_data.columns.values]

    # Keep only Close
    close_cols = [c for c in stock_data.columns if "Close" in c]
    stock_data = stock_data[close_cols].reset_index()

    # Rename consistently
    stock_data.rename(columns={"Date": "date", close_cols[0]: "close"}, inplace=True)
    stock_data["date"] = pd.to_datetime(stock_data["date"]).dt.date
    return stock_data

# === Step 4: Aggregate Sentiment by Date ===
def aggregate_sentiment(data):
    df = pd.DataFrame(
        [
            {
                "date": datetime.fromisoformat(article["published"].replace("Z", "+00:00")).date(),
                "sentiment": article["sentiment_score"],
            }
            for article in data
        ]
    )
    return df.groupby("date")["sentiment"].mean().reset_index()

# === Step 5: Merge Sentiment + Stock Data ===
def merge_data(sentiment_df, stock_df):
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date
    stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.date
    merged = pd.merge(sentiment_df, stock_df, on="date", how="inner")
    return merged

# === Step 6: Train Prophet with Sentiment Regressor ===
def train_and_forecast(merged_df, periods=2):
    df = merged_df.rename(columns={"date": "ds", "close": "y"})
    df["ds"] = pd.to_datetime(df["ds"])  # Prophet requires datetime
    df = df[["ds", "y", "sentiment"]]

    model = Prophet(daily_seasonality=True)
    model.add_regressor("sentiment")
    model.fit(df)

    # Make future DF
    future = model.make_future_dataframe(periods=periods, freq="D")
    future = future.merge(df[["ds", "sentiment"]], on="ds", how="left")

    # Fill forward sentiment
    last_sentiment = df["sentiment"].iloc[-1]
    future["sentiment"] = future["sentiment"].fillna(last_sentiment)

    forecast = model.predict(future)
    return forecast, df, model

# === Step 7: Extract Predictions ===
def extract_predictions(forecast, df):
    last_close = df["y"].iloc[-1]
    preds = {}
    future_rows = forecast.tail(2)  # +24h, +48h
    horizons = ["+24h", "+48h"]

    for h, row in zip(horizons, future_rows.itertuples()):
        abs_price = float(row.yhat)
        pct_change = float(((abs_price - last_close) / last_close) * 100)
        preds[h] = {"predicted_close": abs_price, "pct_change": pct_change}
    return preds

# === Step 8: Update JSON ===
def update_json(data, preds, output_file="sentiment_output.json"):
    if "market_forecast" not in data[0]:
        for entry in data:
            entry["market_forecast"] = {}
    data[0]["market_forecast"]["predictions"] = preds
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

# === Step 9: Visualization ===
def plot_results(df, forecast):
    plt.figure(figsize=(12, 8))

    # Subplot 1: Close price (actual vs forecast)
    plt.subplot(2, 1, 1)
    plt.plot(df["ds"], df["y"], label="Actual Close", marker="o")
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast", linestyle="--")
    plt.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        color="lightblue",
        alpha=0.3,
        label="Uncertainty Interval",
    )
    plt.title("Stock Close Price Forecast with Sentiment Influence")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()

    # Subplot 2: % Change
    df["pct_change"] = df["y"].pct_change() * 100
    forecast["pct_change"] = forecast["yhat"].pct_change() * 100

    plt.subplot(2, 1, 2)
    plt.plot(df["ds"], df["pct_change"], label="Actual % Change", marker="o")
    plt.plot(forecast["ds"], forecast["pct_change"], label="Forecast % Change", linestyle="--")
    plt.title("Predicted % Change in Stock Price")
    plt.xlabel("Date")
    plt.ylabel("% Change")
    plt.legend()

    plt.tight_layout()
    plt.show()

# === Main ===
if __name__ == "__main__":
    company = input("Enter company ticker (e.g., AAPL, TSLA): ").strip().upper()
    sentiment_data = load_sentiment_json()

    # Get date range
    start_date, end_date = get_date_range(sentiment_data)

    # Pull YFinance
    stock_df = fetch_yfinance_data(company, start_date, end_date)

    # Aggregate sentiment
    sentiment_df = aggregate_sentiment(sentiment_data)

    # Merge
    merged_df = merge_data(sentiment_df, stock_df)

    if merged_df.empty:
        print("No overlapping data between sentiment and stock market. Try different dates/ticker.")
    else:
        forecast, df, model = train_and_forecast(merged_df, periods=2)
        preds = extract_predictions(forecast, df)
        update_json(sentiment_data, preds)
        print("Predictions added to sentiment_output.json")

        # Plot results
        plot_results(df, forecast)
