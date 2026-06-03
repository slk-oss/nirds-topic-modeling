import os, warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv; load_dotenv()
import psycopg2
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from config.stopwords import MODEL_EXTRA

EXCLUDED = {"Бизнес"}
MIN_DOC_TOKENS = 10

conn = psycopg2.connect(dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"))
cur = conn.cursor()
cur.execute("""SELECT t.name, p.lemmas FROM processed_documents p
               JOIN raw_documents r ON p.doc_id = r.id
               JOIN topics t ON r.topic_id = t.id""")
rows = cur.fetchall(); conn.close()

topics, texts = [], []
for topic, lemmas in rows:
    if topic in EXCLUDED: continue
    if len(lemmas) < MIN_DOC_TOKENS: continue
    filtered = [l for l in lemmas if l not in MODEL_EXTRA]
    if filtered:
        topics.append(topic); texts.append(filtered)

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=5, no_above=0.85)
bow_corpus = [dictionary.doc2bow(t) for t in texts]

print("Обучение K=12 (passes=15)...")
m12 = LdaModel(corpus=bow_corpus, id2word=dictionary, num_topics=12,
               passes=15, iterations=400, alpha="auto", eta="auto",
               random_state=42, minimum_probability=0.0)
cv12 = CoherenceModel(model=m12, texts=texts, dictionary=dictionary,
                      coherence="c_v").get_coherence()
print(f"K=12  c_v = {cv12:.4f}\n")
print("=== K=12 топ-10 слов ===")
for i in range(12):
    words = ", ".join(w for w, _ in m12.show_topic(i, topn=10))
    print(f"  Тема {i+1:>2}: {words}")

# per-topic coherence
print("\n=== Когерентность по темам (K=12) ===")
for i in range(12):
    top_words = [w for w, _ in m12.show_topic(i, topn=10)]
    cm = CoherenceModel(topics=[top_words], texts=texts,
                        dictionary=dictionary, coherence="c_v")
    print(f"  Тема {i+1:>2}: {cm.get_coherence():.4f}")
