import os
import requests
from datetime import datetime, timedelta
from transformers import pipeline
import sqlite3
import json

# ----------------------------
# CONFIG
# ----------------------------
os.environ["NEWSAPI_API_KEY"] = "6849fe83d3424efcab4117431bb2c0b1"
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY", "YOUR_NEWSAPI_KEY")
DB_NAME = "sentiment.db"

# Initialize Roberta sentiment pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
)

# ----------------------------
# FETCH NEWS (per day, 3–4 articles)
# ----------------------------
def fetch_company_news(company: str, days_back: int = 7, per_day: int = 3):
    all_articles = []
    today = datetime.now()

    for i in range(days_back):
        day_start = (today - timedelta(days=i+1)).strftime('%Y-%m-%d')
        day_end = (today - timedelta(days=i)).strftime('%Y-%m-%d')

        url = "https://newsapi.org/v2/everything"
        params = {
            "qInTitle": company,
            "from": day_start,
            "to": day_end,
            "language": "en",
            "sortBy": "publishedAt",   # latest articles first
            "pageSize": per_day,       # 3–4 per day
            "apiKey": NEWSAPI_API_KEY
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            day_articles = response.json().get("articles", [])
            all_articles.extend(day_articles)
        except Exception as e:
            print(f"[ERROR] Fetching news for {day_start} failed: {e}")

    return all_articles

# ----------------------------
# SENTIMENT ANALYSIS
# ----------------------------
def analyze_articles(articles):
    records = []
    for a in articles:
        title = a.get("title", "No title")
        url = a.get("url", "No URL")
        published = a.get("publishedAt", datetime.now().isoformat())
        content = f"{a.get('title','')} {a.get('description','')}"

        try:
            result = sentiment_analyzer(content[:512])[0]
            label, score = result["label"], result["score"]
        except Exception as e:
            label, score = "ERROR", 0.0

        records.append({
            "title": title,
            "url": url,
            "published": published,
            "sentiment_label": label,
            "sentiment_score": score
        })
    return records

# ----------------------------
# DATABASE
# ----------------------------
def init_db(db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT,
            published TEXT,
            sentiment_label TEXT,
            sentiment_score REAL
        )
    """)
    conn.commit()
    conn.close()

def save_records(records, db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    for r in records:
        cur.execute("""
            INSERT INTO news_sentiment
            (title, url, published, sentiment_label, sentiment_score)
            VALUES (?, ?, ?, ?, ?)
        """, (r["title"], r["url"], r["published"], r["sentiment_label"], r["sentiment_score"]))
    conn.commit()
    conn.close()

def export_json(records, file_name="sentiment_output.json"):
    with open(file_name, "w") as f:
        json.dump(records, f, indent=4)

# ----------------------------
# MAIN PIPELINE
# ----------------------------
if __name__ == "__main__":
    init_db()
    company = input("Enter company name: ")
    days_back = int(input("Enter number of past days: "))
    per_day = int(input("Enter articles per day (3–4 recommended): "))

    articles = fetch_company_news(company, days_back=days_back, per_day=per_day)
    if not articles:
        print("No articles found.")
    else:
        records = analyze_articles(articles)
        save_records(records)
        export_json(records)
        print(f"Fetched {len(records)} articles. Saved to {DB_NAME} and sentiment_output.json")
