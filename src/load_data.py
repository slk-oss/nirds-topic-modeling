import os
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
    )


def load_csv(filepath, nrows=10000):
    df = pd.read_csv(filepath, nrows=nrows, usecols=["title", "text", "topic", "date"])
    df = df.dropna(subset=["text"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["word_count"] = df["text"].str.split().str.len()
    return df


def insert_topics(cur, df):
    topics = df["topic"].dropna().unique().tolist()
    cur.executemany(
        "INSERT INTO topics (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
        [(t,) for t in topics],
    )
    cur.execute("SELECT id, name FROM topics")
    return {name: tid for tid, name in cur.fetchall()}


def insert_documents(cur, df, topic_map):
    rows = [
        (
            topic_map.get(r.topic),
            r.title,
            None if pd.isnull(r.date) else r.date,
            r.text,
            r.word_count,
        )
        for r in df.itertuples(index=False)
    ]
    execute_values(
        cur,
        "INSERT INTO raw_documents (topic_id, title, published_at, body, word_count) VALUES %s",
        rows,
    )


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/lenta-ru-news.csv"
    df = load_csv(filepath)
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                topic_map = insert_topics(cur, df)
                insert_documents(cur, df, topic_map)
        print(f"Loaded {len(df)} documents, {len(topic_map)} topics")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
