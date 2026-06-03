import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import psycopg2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
from dotenv import load_dotenv

from config.stopwords import MODEL_EXTRA

load_dotenv()

matplotlib.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 120

EXCLUDED_TOPICS = {"Бизнес"}
MIN_DOC_TOKENS = 10
K_MIN, K_MAX = 5, 20


# ── загрузка ──────────────────────────────────────────────────────────────────

def load_lemmas():
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"), port=os.getenv("DB_PORT", "5432"),
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT t.name, p.lemmas
        FROM processed_documents p
        JOIN raw_documents r ON p.doc_id = r.id
        JOIN topics t ON r.topic_id = t.id
    """)
    rows = cur.fetchall()
    conn.close()

    total_raw = len(rows)
    n_excluded = sum(1 for topic, _ in rows if topic in EXCLUDED_TOPICS)
    n_short = sum(
        1 for topic, lemmas in rows
        if topic not in EXCLUDED_TOPICS and len(lemmas) < MIN_DOC_TOKENS
    )

    topics, texts = [], []
    for topic, lemmas in rows:
        if topic in EXCLUDED_TOPICS:
            continue
        if len(lemmas) < MIN_DOC_TOKENS:
            continue
        filtered = [l for l in lemmas if l not in MODEL_EXTRA]
        if filtered:
            topics.append(topic)
            texts.append(filtered)

    print(f"Загружено из БД:         {total_raw}")
    print(f"Исключено (рубрики):     {n_excluded}  {EXCLUDED_TOPICS}")
    print(f"Удалено (< {MIN_DOC_TOKENS} токенов):  {n_short}")
    print(f"Итого документов:        {len(texts)}, {len(set(topics))} рубрик")
    return topics, texts


# ── словарь и корпус ──────────────────────────────────────────────────────────

def build_corpus(texts):
    dictionary = corpora.Dictionary(texts)
    before = len(dictionary)
    dictionary.filter_extremes(no_below=5, no_above=0.85)
    after = len(dictionary)
    print(f"Словарь: {before} → {after} токенов (filter_extremes no_below=5, no_above=0.85)")
    bow_corpus = [dictionary.doc2bow(t) for t in texts]
    return dictionary, bow_corpus


# ── обучение и когерентность ─────────────────────────────────────────────────

def train_lda(bow_corpus, dictionary, k, passes, iterations=200, random_state=42):
    return LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=k,
        passes=passes,
        iterations=iterations,
        alpha="auto",
        eta="auto",
        random_state=random_state,
        minimum_probability=0.0,
    )


def coherence_cv(model, texts, dictionary):
    cm = CoherenceModel(
        model=model, texts=texts, dictionary=dictionary, coherence="c_v"
    )
    return cm.get_coherence()


# ── поиск K ──────────────────────────────────────────────────────────────────

def search_k(bow_corpus, texts, dictionary):
    results = []
    k_range = range(K_MIN, K_MAX + 1)
    for k in k_range:
        model = train_lda(bow_corpus, dictionary, k, passes=5, iterations=100)
        score = coherence_cv(model, texts, dictionary)
        results.append({"k": k, "coherence_cv": round(score, 4)})
        print(f"  K={k:>2}  c_v={score:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("reports/lda_coherence_scores.csv", index=False)
    return df


# ── визуализации ──────────────────────────────────────────────────────────────

def plot_coherence(df, best_k, path="reports/06_lda_coherence.png"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["k"], df["coherence_cv"], marker="o", color="steelblue", linewidth=2)
    ax.axvline(best_k, color="tomato", linestyle="--",
               label=f"Оптимум K={best_k}")
    ax.set_xlabel("Число тем (K)")
    ax.set_ylabel("Когерентность c_v")
    ax.set_title("Подбор оптимального K для LDA")
    ax.set_xticks(df["k"])
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {path}")


def plot_topics(model, best_k, path="reports/07_lda_topics.png"):
    cols = 3
    rows = (best_k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.6))
    axes = axes.flatten()
    palette = sns.color_palette("muted", best_k)

    for i in range(best_k):
        top_words = model.show_topic(i, topn=10)
        words = [w for w, _ in reversed(top_words)]
        weights = [s for _, s in reversed(top_words)]
        axes[i].barh(words, weights, color=palette[i])
        axes[i].set_title(f"Тема {i + 1}", fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Вес", fontsize=8)
        axes[i].tick_params(axis="y", labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"LDA K={best_k}: топ-10 слов по темам", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {path}")


def plot_topic_rubric(model, topics, bow_corpus, best_k,
                      path="reports/08_lda_topic_rubric.png"):
    topic_labels = [f"Тема {i + 1}" for i in range(best_k)]
    rubrics = sorted(set(topics))

    # средняя вероятность темы для каждой рубрики
    matrix = np.zeros((len(rubrics), best_k))
    counts = {r: 0 for r in rubrics}
    for rubric, bow in zip(topics, bow_corpus):
        dist = dict(model.get_document_topics(bow, minimum_probability=0.0))
        row = rubrics.index(rubric)
        for t, p in dist.items():
            matrix[row, t] += p
        counts[rubric] += 1
    for i, r in enumerate(rubrics):
        if counts[r]:
            matrix[i] /= counts[r]

    df_heat = pd.DataFrame(matrix, index=rubrics, columns=topic_labels)
    fig, ax = plt.subplots(figsize=(max(14, best_k * 0.9), 7))
    sns.heatmap(df_heat, ax=ax, cmap="YlOrRd", fmt=".2f",
                linewidths=0.4, annot=True, annot_kws={"size": 7},
                cbar_kws={"label": "P(тема|рубрика)"})
    ax.set_title(f"LDA K={best_k}: распределение тем по рубрикам",
                 fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {path}")


def save_pyldavis(model, bow_corpus, dictionary,
                  path="reports/09_lda_pyldavis.html"):
    vis = pyLDAvis.gensim_models.prepare(
        model, bow_corpus, dictionary, sort_topics=False
    )
    pyLDAvis.save_html(vis, path)
    print(f"Сохранено: {path}")


def save_keywords_csv(model, best_k, path="reports/lda_keywords.csv"):
    rows = []
    for i in range(best_k):
        for rank, (word, weight) in enumerate(model.show_topic(i, topn=15), 1):
            rows.append({"topic_id": i + 1, "rank": rank,
                         "word": word, "weight": round(weight, 6)})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Сохранено: {path}")


# ── точка входа ───────────────────────────────────────────────────────────────

def main():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    topics, texts = load_lemmas()
    dictionary, bow_corpus = build_corpus(texts)

    print(f"\nПоиск K от {K_MIN} до {K_MAX} (passes=5)...")
    df_scores = search_k(bow_corpus, texts, dictionary)

    best_k = int(df_scores.loc[df_scores["coherence_cv"].idxmax(), "k"])
    best_cv = df_scores["coherence_cv"].max()
    print(f"\nОптимальный K = {best_k}  (c_v = {best_cv:.4f})")

    print(f"\nОбучение финальной модели K={best_k} (passes=15)...")
    final_model = train_lda(bow_corpus, dictionary, best_k,
                            passes=15, iterations=400)
    final_cv = coherence_cv(final_model, texts, dictionary)
    print(f"Финальная c_v = {final_cv:.4f}")

    model_path = f"models/lda_k{best_k}.gensim"
    final_model.save(model_path)
    print(f"Модель сохранена: {model_path}")

    plot_coherence(df_scores, best_k)
    plot_topics(final_model, best_k)
    plot_topic_rubric(final_model, topics, bow_corpus, best_k)
    save_pyldavis(final_model, bow_corpus, dictionary)
    save_keywords_csv(final_model, best_k)

    print("\n── Топ-10 слов по темам ──")
    for i in range(best_k):
        words = ", ".join(w for w, _ in final_model.show_topic(i, topn=10))
        print(f"  Тема {i + 1:>2}: {words}")

    print("\n── Таблица когерентности ──")
    print(df_scores.to_string(index=False))


if __name__ == "__main__":
    main()
