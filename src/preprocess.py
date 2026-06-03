import os
import re
import time
from collections import Counter

import psycopg2
from psycopg2.extras import execute_values
import pymorphy2
import nltk
from dotenv import load_dotenv

from config.stopwords import PREPROCESS_EXTRA

load_dotenv()

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

STOPWORDS = set(nltk_stopwords.words("russian")) | PREPROCESS_EXTRA

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NONALPHA_RE = re.compile(r"[^а-яёa-z\s]")
_SPACES_RE = re.compile(r"\s+")
_CYR_RE = re.compile(r"^[а-яё]+$")

MIN_LEN = 3


def clean_text(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _NONALPHA_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", text).strip()


def tokenize(text: str) -> list:
    return [t for t in text.split() if len(t) >= MIN_LEN and _CYR_RE.match(t)]


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS]


def lemmatize(tokens: list, morph) -> list:
    lemmas = [morph.parse(t)[0].normal_form for t in tokens]
    return [l for l in lemmas if l not in STOPWORDS and len(l) >= MIN_LEN]


def process_document(text: str, morph) -> tuple:
    cleaned = clean_text(text)
    tokens = remove_stopwords(tokenize(cleaned))
    lemmas = lemmatize(tokens, morph)
    return lemmas, tokens


def insert_batch(cur, rows: list) -> None:
    execute_values(
        cur,
        """
        INSERT INTO processed_documents (doc_id, lemmas, tokens_raw, lang)
        VALUES %s
        ON CONFLICT (doc_id) DO NOTHING
        """,
        rows,
    )


def main(batch_size: int = 500) -> None:
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
    )

    morph = pymorphy2.MorphAnalyzer()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, body FROM raw_documents
            WHERE id NOT IN (SELECT doc_id FROM processed_documents)
            ORDER BY id
        """)
        docs = cur.fetchall()

    total = len(docs)
    print(f"Документов для обработки: {total}")

    all_token_counts = []
    all_lemma_counts = []
    all_lemmas = []

    t0 = time.time()
    with conn:
        with conn.cursor() as cur:
            for start in range(0, total, batch_size):
                batch = docs[start: start + batch_size]
                rows = []
                for doc_id, body in batch:
                    lemmas, tokens_raw = process_document(body, morph)
                    rows.append((doc_id, lemmas, tokens_raw, "ru"))
                    all_token_counts.append(len(tokens_raw))
                    all_lemma_counts.append(len(lemmas))
                    all_lemmas.extend(lemmas)
                insert_batch(cur, rows)

                done = min(start + batch_size, total)
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  {done}/{total}  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nГотово за {elapsed:.1f} сек\n")

    avg_tokens = sum(all_token_counts) / len(all_token_counts)
    avg_lemmas = sum(all_lemma_counts) / len(all_lemma_counts)
    top20 = Counter(all_lemmas).most_common(20)

    print(f"Среднее токенов до обработки:  {avg_tokens:.1f}")
    print(f"Среднее лемм после обработки:  {avg_lemmas:.1f}")
    print(f"Сжатие:                        {avg_lemmas/avg_tokens*100:.1f}% от исходного\n")
    print("Топ-20 частотных лемм:")
    for rank, (lemma, count) in enumerate(top20, 1):
        print(f"  {rank:>2}. {lemma:<20} {count}")

    conn.close()


if __name__ == "__main__":
    main()
