CREATE TABLE topics (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE raw_documents (
    id           SERIAL PRIMARY KEY,
    topic_id     INT REFERENCES topics(id) ON DELETE RESTRICT,
    title        TEXT,
    body         TEXT,
    published_at DATE,
    word_count   INT,
    char_count   INT GENERATED ALWAYS AS (LENGTH(COALESCE(body, ''))) STORED,
    loaded_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE processed_documents (
    id           SERIAL PRIMARY KEY,
    doc_id       INT UNIQUE REFERENCES raw_documents(id) ON DELETE CASCADE,
    lemmas       TEXT[],
    tokens_raw   TEXT[],
    lang         VARCHAR(10) DEFAULT 'ru',
    processed_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_raw_docs_topic  ON raw_documents(topic_id);
CREATE INDEX idx_raw_docs_date   ON raw_documents(published_at);
CREATE INDEX idx_proc_docs_doc   ON processed_documents(doc_id);
