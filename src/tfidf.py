import os
import csv

import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

from config.stopwords import MODEL_EXTRA

load_dotenv()

matplotlib.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 120

EXCLUDED_TOPICS = {"Бизнес"}
MIN_DOC_TOKENS = 10


def load_corpus():
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"), port=os.getenv("DB_PORT", "5432"),
    )
    df = pd.read_sql("""
        SELECT t.name AS topic, p.lemmas
        FROM processed_documents p
        JOIN raw_documents r ON p.doc_id = r.id
        JOIN topics t ON r.topic_id = t.id
        ORDER BY t.name
    """, conn)
    conn.close()
    total_raw = len(df)
    df = df[~df["topic"].isin(EXCLUDED_TOPICS)]
    after_excl = len(df)
    df = df[df["lemmas"].apply(len) >= MIN_DOC_TOKENS].reset_index(drop=True)
    after_len = len(df)
    print(f"Загружено: {total_raw} → исключено рубрик {total_raw - after_excl} "
          f"→ короткие документы удалены {after_excl - after_len} "
          f"→ итого {after_len} документов, {df['topic'].nunique()} рубрик")
    df["text"] = df["lemmas"].apply(" ".join)
    return df


def build_tfidf(df):
    vectorizer = TfidfVectorizer(
        min_df=5,
        max_df=0.85,
        max_features=5000,
        sublinear_tf=True,
        stop_words=list(MODEL_EXTRA),
    )
    X = vectorizer.fit_transform(df["text"])
    vocab_size = len(vectorizer.get_feature_names_out())
    print(f"Словарь TF-IDF: {vocab_size} лемм (min_df=5, max_df=0.85, max_features=5000)")
    return X, vectorizer


def topic_keywords(X, df, vectorizer, top_n=15):
    feature_names = vectorizer.get_feature_names_out()
    topics = df["topic"].unique()
    results = {}
    for topic in sorted(topics):
        mask = (df["topic"] == topic).values
        avg = np.asarray(X[mask].mean(axis=0)).flatten()
        idx = avg.argsort()[::-1][:top_n]
        results[topic] = [(feature_names[i], float(avg[i])) for i in idx]
    return results


def save_csv(keywords, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "rank", "lemma", "tfidf_score"])
        for topic, words in keywords.items():
            for rank, (lemma, score) in enumerate(words, 1):
                writer.writerow([topic, rank, lemma, f"{score:.6f}"])
    print(f"Сохранено: {path}")


def plot_bar_charts(keywords, path):
    topics = list(keywords.keys())
    cols = 3
    rows = (len(topics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.8))
    axes = axes.flatten()

    palette = sns.color_palette("muted", 15)
    for i, (topic, words) in enumerate(keywords.items()):
        lemmas = [w[0] for w in reversed(words)]
        scores = [w[1] for w in reversed(words)]
        axes[i].barh(lemmas, scores, color=palette[i])
        axes[i].set_title(topic, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("TF-IDF", fontsize=8)
        axes[i].tick_params(axis="y", labelsize=8)
        axes[i].tick_params(axis="x", labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("TF-IDF: топ-15 ключевых слов по рубрикам", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {path}")


def plot_heatmap(keywords, path):
    # берём топ-5 слов каждой рубрики → уникальные → сортируем по макс. весу
    candidate_set = {}
    for words in keywords.values():
        for lemma, score in words[:5]:
            candidate_set[lemma] = max(candidate_set.get(lemma, 0), score)
    top_lemmas = [l for l, _ in sorted(candidate_set.items(), key=lambda x: -x[1])][:35]

    kw_dict = {topic: dict(words) for topic, words in keywords.items()}
    topics = list(keywords.keys())

    matrix = pd.DataFrame(
        [[kw_dict[t].get(l, 0.0) for l in top_lemmas] for t in topics],
        index=topics,
        columns=top_lemmas,
    )

    fig, ax = plt.subplots(figsize=(22, 7))
    sns.heatmap(
        matrix, ax=ax, cmap="YlOrRd", linewidths=0.4,
        xticklabels=True, yticklabels=True,
        cbar_kws={"label": "avg TF-IDF"},
    )
    ax.set_title("TF-IDF heatmap: рубрики × ключевые леммы", fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {path}")


def print_keywords(keywords):
    print()
    for topic, words in keywords.items():
        terms = ", ".join(w[0] for w in words)
        print(f"  {topic:<22} {terms}")


def main():
    df = load_corpus()
    X, vectorizer = build_tfidf(df)
    keywords = topic_keywords(X, df, vectorizer, top_n=15)

    os.makedirs("reports", exist_ok=True)
    save_csv(keywords, "reports/tfidf_keywords.csv")
    plot_bar_charts(keywords, "reports/04_tfidf_top_topics.png")
    plot_heatmap(keywords, "reports/05_tfidf_heatmap.png")
    print_keywords(keywords)


if __name__ == "__main__":
    main()
