import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from load_data import load_csv

SAMPLE = (
    "title,text,topic,date\n"
    "Заголовок 1,Текст статьи о политике,Политика,2020-01-01\n"
    "Заголовок 2,Текст статьи о спорте,Спорт,2020-01-02\n"
    "Заголовок 3,Текст статьи о науке,Наука,2020-01-03\n"
)


@pytest.fixture
def csv_file(tmp_path):
    f = tmp_path / "sample.csv"
    f.write_text(SAMPLE, encoding="utf-8")
    return str(f)


def test_csv_loads_without_error(csv_file):
    df = load_csv(csv_file, nrows=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_no_empty_documents(csv_file):
    df = load_csv(csv_file, nrows=10)
    assert df["text"].notna().all()
    assert (df["text"].str.strip() != "").all()


def test_topics_not_empty(csv_file):
    df = load_csv(csv_file, nrows=10)
    assert df["topic"].notna().all()
    assert df["topic"].nunique() > 0
