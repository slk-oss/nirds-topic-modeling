import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psycopg2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim import corpora
from gensim.models import CoherenceModel
from dotenv import load_dotenv

from config.stopwords import MODEL_EXTRA

load_dotenv()

matplotlib.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 120

EXCLUDED_TOPICS = {"Бизнес"}
MIN_DOC_TOKENS  = 10
K_MIN, K_MAX    = 5, 30
K_FINAL         = 19          # финальная модель для отчёта (независимо от аргмакса)
LDA_BEST_K      = 19
LDA_BEST_CV     = 0.5755


# ── загрузка ──────────────────────────────────────────────────────────────────

def load_corpus():
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),  user=os.getenv("DB_USER"),
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

    records = []
    n_excl = n_short = 0
    for topic, lemmas in rows:
        if topic in EXCLUDED_TOPICS:
            n_excl += 1
            continue
        if len(lemmas) < MIN_DOC_TOKENS:
            n_short += 1
            continue
        filtered = [l for l in lemmas if l not in MODEL_EXTRA]
        if filtered:
            records.append({"topic": topic, "lemmas": filtered})

    df = pd.DataFrame(records)
    df["text"] = df["lemmas"].apply(" ".join)
    topics = df["topic"].tolist()
    texts  = df["lemmas"].tolist()

    print(f"Загружено: {len(rows)} → исключено рубрик {n_excl}, "
          f"коротких {n_short} → итого {len(df)} документов, "
          f"{df['topic'].nunique()} рубрик")
    return topics, texts, df


# ── TF-IDF матрица ────────────────────────────────────────────────────────────

def build_tfidf(df):
    vectorizer = TfidfVectorizer(
        min_df=5,
        max_df=0.85,
        max_features=5000,
        sublinear_tf=True,
        stop_words=list(MODEL_EXTRA),
    )
    X = vectorizer.fit_transform(df["text"])
    print(f"TF-IDF матрица: {X.shape[0]} × {X.shape[1]}")
    return X, vectorizer


# ── gensim словарь для когерентности ─────────────────────────────────────────

def build_gensim_dict(texts):
    dictionary = corpora.Dictionary(texts)
    before = len(dictionary)
    dictionary.filter_extremes(no_below=5, no_above=0.85)
    print(f"Gensim словарь: {before} → {len(dictionary)} токенов")
    return dictionary


# ── NMF ───────────────────────────────────────────────────────────────────────

def train_nmf(X, k):
    model = NMF(
        n_components=k,
        init="nndsvda",
        random_state=42,
        max_iter=400,
        alpha_W=0.0,
        alpha_H=0.0,
    )
    model.fit(X)
    return model


def get_topic_words(model, vectorizer, topn=10):
    feature_names = vectorizer.get_feature_names_out()
    result = []
    for comp in model.components_:
        indices = comp.argsort()[::-1][:topn]
        result.append([feature_names[i] for i in indices])
    return result


# ── когерентность ─────────────────────────────────────────────────────────────

def coherence_cv(topic_words, texts, dictionary):
    cm = CoherenceModel(
        topics=topic_words, texts=texts,
        dictionary=dictionary, coherence="c_v", processes=1,
    )
    return cm.get_coherence()


def per_topic_coherence(topic_words, texts, dictionary):
    scores = []
    for words in topic_words:
        cm = CoherenceModel(
            topics=[words], texts=texts,
            dictionary=dictionary, coherence="c_v", processes=1,
        )
        scores.append(round(cm.get_coherence(), 4))
    return scores


# ── поиск K ───────────────────────────────────────────────────────────────────

def search_k(X, texts, vectorizer, dictionary):
    results = []
    for k in range(K_MIN, K_MAX + 1):
        model   = train_nmf(X, k)
        t_words = get_topic_words(model, vectorizer, topn=10)
        cv      = coherence_cv(t_words, texts, dictionary)
        results.append({"k": k, "coherence_cv": round(cv, 4)})
        print(f"  K={k:>2}  c_v={cv:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("reports/nmf_coherence_scores.csv", index=False)
    print("Сохранено: reports/nmf_coherence_scores.csv")
    return df


# ── визуализации ──────────────────────────────────────────────────────────────

def plot_coherence(df_scores, best_k, path="reports/10_nmf_coherence.png"):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df_scores["k"], df_scores["coherence_cv"],
            marker="o", color="tomato", linewidth=2)
    ax.axvline(best_k, color="steelblue", linestyle="--",
               label=f"Оптимум K={best_k}")
    ax.set_xlabel("Число тем (K)")
    ax.set_ylabel("Когерентность c_v")
    ax.set_title("Подбор оптимального K для NMF")
    ax.set_xticks(df_scores["k"])
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {path}")


def plot_topics(model, vectorizer, best_k, path="reports/11_nmf_topics.png"):
    feature_names = vectorizer.get_feature_names_out()
    cols = 3
    rows = (best_k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.6))
    axes = axes.flatten()
    palette = sns.color_palette("muted", best_k)

    for i, comp in enumerate(model.components_):
        indices   = comp.argsort()[::-1][:10]
        words     = [feature_names[j] for j in reversed(indices)]
        weights   = [comp[j]          for j in reversed(indices)]
        axes[i].barh(words, weights, color=palette[i])
        axes[i].set_title(f"Тема {i + 1}", fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Вес", fontsize=8)
        axes[i].tick_params(axis="y", labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"NMF K={best_k}: топ-10 слов по темам",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {path}")


def plot_topic_rubric(model, X, topics, best_k,
                      path="reports/12_nmf_topic_rubric.png"):
    H = model.transform(X)
    row_sums = H.sum(axis=1, keepdims=True)
    H_norm = H / np.where(row_sums == 0, 1, row_sums)

    rubrics      = sorted(set(topics))
    topic_labels = [f"Тема {i + 1}" for i in range(best_k)]

    matrix = np.zeros((len(rubrics), best_k))
    counts = {r: 0 for r in rubrics}
    for rubric, h_row in zip(topics, H_norm):
        idx = rubrics.index(rubric)
        matrix[idx] += h_row
        counts[rubric] += 1
    for i, r in enumerate(rubrics):
        if counts[r]:
            matrix[i] /= counts[r]

    df_heat = pd.DataFrame(matrix, index=rubrics, columns=topic_labels)
    fig, ax  = plt.subplots(figsize=(max(14, best_k * 0.9), 7))
    sns.heatmap(df_heat, ax=ax, cmap="YlOrRd", fmt=".2f",
                linewidths=0.4, annot=True, annot_kws={"size": 7},
                cbar_kws={"label": "P(тема|рубрика)"})
    ax.set_title(f"NMF K={best_k}: распределение тем по рубрикам",
                 fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {path}")


def plot_comparison(nmf_df, best_k_nmf, nmf_final_cv,
                    path="reports/13_comparison_lda_nmf.png"):
    lda_path = "reports/lda_coherence_scores.csv"
    lda_df   = pd.read_csv(lda_path) if os.path.exists(lda_path) else None

    fig, ax = plt.subplots(figsize=(12, 5))

    if lda_df is not None:
        ax.plot(lda_df["k"], lda_df["coherence_cv"],
                marker="o", color="steelblue", linewidth=2,
                label="LDA (passes=5, K=5–20)")
    ax.plot(nmf_df["k"], nmf_df["coherence_cv"],
            marker="s", color="tomato", linewidth=2,
            label="NMF (K=5–30)")

    # финальные значения
    ax.scatter([LDA_BEST_K], [LDA_BEST_CV], color="steelblue",
               s=120, zorder=5, marker="*",
               label=f"LDA финал K={LDA_BEST_K} c_v={LDA_BEST_CV}")
    ax.scatter([best_k_nmf], [nmf_final_cv], color="tomato",
               s=120, zorder=5, marker="*",
               label=f"NMF финал K={best_k_nmf} c_v={nmf_final_cv:.4f}")

    ax.set_xlabel("Число тем (K)")
    ax.set_ylabel("Когерентность c_v")
    ax.set_title("LDA vs NMF: кривые когерентности и финальные модели")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {path}")


# ── сохранение данных ─────────────────────────────────────────────────────────

def save_keywords_csv(model, vectorizer, best_k,
                      path="reports/nmf_keywords.csv"):
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for i, comp in enumerate(model.components_):
        indices = comp.argsort()[::-1][:15]
        for rank, idx in enumerate(indices, 1):
            rows.append({
                "topic_id": i + 1,
                "rank":     rank,
                "word":     feature_names[idx],
                "weight":   round(float(comp[idx]), 6),
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Сохранено: {path}")


def save_topic_coherence_csv(topic_words, texts, dictionary,
                              path="reports/nmf_topic_coherence.csv"):
    scores = per_topic_coherence(topic_words, texts, dictionary)
    rows   = [{"topic_id": i + 1, "coherence_cv": cv}
              for i, cv in enumerate(scores)]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Сохранено: {path}")
    return scores


def save_comparison_csv(best_k_nmf, cv_nmf,
                        path="reports/comparison_summary.csv"):
    rows = [
        {"model": "LDA", "best_k": LDA_BEST_K, "coherence_cv": LDA_BEST_CV},
        {"model": "NMF", "best_k": best_k_nmf, "coherence_cv": round(cv_nmf, 4)},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Сохранено: {path}")


# ── точка входа ───────────────────────────────────────────────────────────────

def main():
    os.makedirs("reports", exist_ok=True)

    # 1. данные
    topics, texts, df = load_corpus()

    # 2. матрица и словарь
    X, vectorizer = build_tfidf(df)
    dictionary    = build_gensim_dict(texts)

    # 3. поиск K
    print(f"\nПоиск K от {K_MIN} до {K_MAX}...")
    df_scores = search_k(X, texts, vectorizer, dictionary)

    argmax_k = int(df_scores.loc[df_scores["coherence_cv"].idxmax(), "k"])
    argmax_cv = df_scores["coherence_cv"].max()
    print(f"\nАргмакс c_v: K={argmax_k} ({argmax_cv:.4f})")
    print(f"Финальная модель: K={K_FINAL} (зафиксировано для отчёта)")

    # 4. финальная модель
    print(f"\nОбучение финальной модели K={K_FINAL}...")
    final_model = train_nmf(X, K_FINAL)
    t_words     = get_topic_words(final_model, vectorizer, topn=10)
    final_cv    = coherence_cv(t_words, texts, dictionary)
    print(f"Финальная c_v = {final_cv:.4f}")

    # 5. отчёты
    save_keywords_csv(final_model, vectorizer, K_FINAL)
    per_topic_scores = save_topic_coherence_csv(t_words, texts, dictionary)
    plot_coherence(df_scores, K_FINAL)
    plot_topics(final_model, vectorizer, K_FINAL)
    plot_topic_rubric(final_model, X, topics, K_FINAL)
    plot_comparison(df_scores, K_FINAL, final_cv)
    save_comparison_csv(K_FINAL, final_cv)

    # 6. итоговый отчёт
    import statistics
    print("\n" + "=" * 62)
    print(f"  NMF финал: K={K_FINAL}  c_v={final_cv:.4f}")
    print(f"  LDA финал: K={LDA_BEST_K}  c_v={LDA_BEST_CV}")
    print(f"  Δ c_v NMF−LDA = {final_cv - LDA_BEST_CV:+.4f}")
    print("=" * 62)

    print("\n── Топ-10 слов по темам ──")
    for i, words in enumerate(t_words):
        cv_str = f"  c_v={per_topic_scores[i]:.4f}"
        print(f"  Тема {i + 1:>2}{cv_str}  {', '.join(words)}")

    print("\n── Per-topic coherence ──")
    print(f"  Средняя:  {statistics.mean(per_topic_scores):.4f}")
    print(f"  Медиана:  {statistics.median(per_topic_scores):.4f}")
    print(f"  Минимум:  {min(per_topic_scores):.4f}")
    print(f"  Максимум: {max(per_topic_scores):.4f}")

    worst5 = sorted(enumerate(per_topic_scores), key=lambda x: x[1])[:5]
    print("\n── Топ-5 худших тем ──")
    for i, cv in worst5:
        print(f"  Тема {i + 1:>2}  c_v={cv:.4f}  {', '.join(t_words[i])}")


if __name__ == "__main__":
    main()
