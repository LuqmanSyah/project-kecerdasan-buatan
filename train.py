"""Melatih ulang model final dari notebook dan menyimpannya ke folder models/.

Langkah dan parameter sama dengan notebook (split 80:20, random_state=42,
tanpa stemming, unigram, alpha=0.05 untuk NB dan C=10 untuk SVM), sehingga
hasil evaluasinya identik dengan bagian "Evaluasi Model Setelah Tuning".

Jalankan:  python train.py
"""

import glob
import json
import os

import joblib
import kagglehub
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocessing import siapkan_teks

ALPHA_TERBAIK = 0.05
C_TERBAIK = 10
FOLDER_MODEL = "models"


def buat_tfidf():
    return TfidfVectorizer(ngram_range=(1, 1), min_df=2, max_df=0.95, sublinear_tf=True)


def muat_data():
    path = kagglehub.dataset_download("gevabriel/indonesian-email-spam")
    file_csv = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)[0]

    df = pd.read_csv(file_csv)
    df = df.dropna(subset=["Pesan", "Kategori"]).drop_duplicates(subset=["Pesan"])
    df["label"] = df["Kategori"].str.lower().str.strip().map({"ham": 0, "spam": 1})
    df["teks"] = df["Pesan"].apply(siapkan_teks)
    df = df[df["teks"].str.len() > 0].reset_index(drop=True)

    return df


def hitung_metrik(y_asli, y_prediksi):
    return {
        "accuracy": accuracy_score(y_asli, y_prediksi),
        "precision": precision_score(y_asli, y_prediksi),
        "recall": recall_score(y_asli, y_prediksi),
        "f1": f1_score(y_asli, y_prediksi),
    }


def main():
    df = muat_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["teks"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    model = {
        "naive_bayes": Pipeline([
            ("tfidf", buat_tfidf()),
            ("model", MultinomialNB(alpha=ALPHA_TERBAIK)),
        ]),
        "svm": Pipeline([
            ("tfidf", buat_tfidf()),
            ("model", SVC(kernel="linear", C=C_TERBAIK, random_state=42)),
        ]),
    }

    os.makedirs(FOLDER_MODEL, exist_ok=True)
    ringkasan = {"jumlah_train": len(X_train), "jumlah_test": len(X_test)}

    for nama, pipe in model.items():
        pipe.fit(X_train, y_train)
        metrik = hitung_metrik(y_test, pipe.predict(X_test))
        ringkasan[nama] = metrik

        joblib.dump(pipe, os.path.join(FOLDER_MODEL, f"{nama}.pkl"))
        print(f"{nama:12s}", {k: round(v, 4) for k, v in metrik.items()})

    with open(os.path.join(FOLDER_MODEL, "metrik.json"), "w") as f:
        json.dump(ringkasan, f, indent=2)

    print(f"Model tersimpan di folder {FOLDER_MODEL}/")


if __name__ == "__main__":
    main()
