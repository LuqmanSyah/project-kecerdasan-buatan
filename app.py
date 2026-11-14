"""Demo klasifikasi email spam bahasa Indonesia.

Jalankan:  streamlit run app.py
"""

import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from preprocessing import siapkan_teks

FOLDER_MODEL = Path(__file__).parent / "models"

CONTOH_EMAIL = {
    "Contoh spam": (
        "Selamat! Anda terpilih sebagai pemenang undian berhadiah uang tunai Rp50.000.000. "
        "Klik tautan berikut dan masukkan data rekening Anda untuk klaim hadiah gratis sekarang juga."
    ),
    "Contoh ham": (
        "Halo semua, terlampir jadwal rapat proyek minggu depan. Mohon dicek kembali dan "
        "kabari saya jika ada perubahan. Terima kasih atas kerja samanya."
    ),
}

st.set_page_config(page_title="Deteksi Email Spam", page_icon="📧", layout="centered")


@st.cache_resource
def muat_model():
    model = {
        "Multinomial Naive Bayes": joblib.load(FOLDER_MODEL / "naive_bayes.pkl"),
        "SVM (Linear)": joblib.load(FOLDER_MODEL / "svm.pkl"),
    }
    metrik = json.loads((FOLDER_MODEL / "metrik.json").read_text())
    return model, metrik


def bobot_kata(pipeline):
    """Bobot tiap kata: positif mengarah ke spam, negatif mengarah ke ham."""
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_log_prob_"):
        return model.feature_log_prob_[1] - model.feature_log_prob_[0]

    koef = model.coef_
    return np.asarray(koef.toarray() if hasattr(koef, "toarray") else koef)[0]


def kontribusi_kata(pipeline, teks_siap, jumlah=8):
    """Kata di dalam email yang paling mendorong prediksi (TF-IDF x bobot kata)."""
    tfidf = pipeline.named_steps["tfidf"]
    vektor = tfidf.transform([teks_siap])
    indeks = vektor.indices

    if len(indeks) == 0:
        return pd.DataFrame(columns=["Kata", "Kontribusi"])

    nilai = vektor.data * bobot_kata(pipeline)[indeks]
    kata = tfidf.get_feature_names_out()[indeks]

    df = pd.DataFrame({"Kata": kata, "Kontribusi": nilai})
    df = df.reindex(df["Kontribusi"].abs().sort_values(ascending=False).index)
    return df.head(jumlah).reset_index(drop=True)


def tampilkan_hasil(nama, pipeline, teks_siap):
    prediksi = int(pipeline.predict([teks_siap])[0])
    model = pipeline.named_steps["model"]

    st.markdown(f"**{nama}**")

    if prediksi == 1:
        st.error("🚫 SPAM")
    else:
        st.success("✅ HAM (bukan spam)")

    if hasattr(model, "predict_proba"):
        prob = pipeline.predict_proba([teks_siap])[0][1]
        st.metric("Probabilitas spam", f"{prob:.1%}")
    else:
        skor = pipeline.decision_function([teks_siap])[0]
        st.metric(
            "Skor keputusan",
            f"{skor:+.2f}",
            help="Jarak ke garis pemisah SVM. Positif berarti spam, negatif berarti ham. "
            "Makin jauh dari 0, makin yakin modelnya.",
        )

    kontribusi = kontribusi_kata(pipeline, teks_siap)
    if not kontribusi.empty:
        st.caption("Kata paling berpengaruh pada email ini")
        kontribusi["Arah"] = np.where(kontribusi["Kontribusi"] > 0, "spam", "ham")
        grafik = (
            alt.Chart(kontribusi)
            .mark_bar()
            .encode(
                x=alt.X("Kontribusi:Q", title="kontribusi (positif → spam, negatif → ham)"),
                y=alt.Y("Kata:N", sort="-x", title=None),
                color=alt.Color(
                    "Arah:N",
                    scale=alt.Scale(domain=["spam", "ham"], range=["#d6453d", "#2b7bba"]),
                    legend=None,
                ),
                tooltip=["Kata", alt.Tooltip("Kontribusi:Q", format="+.3f")],
            )
            .properties(height=260)
        )
        st.altair_chart(grafik, width="stretch")


def isi_contoh(teks):
    st.session_state["email"] = teks


model, metrik = muat_model()

with st.sidebar:
    st.header("Tentang")
    st.write(
        "Demo project akhir mata kuliah **Kecerdasan Buatan**: perbandingan "
        "Multinomial Naive Bayes dan SVM untuk klasifikasi email spam bahasa Indonesia "
        "dengan fitur TF-IDF."
    )

    st.subheader("Kinerja pada data testing")
    st.dataframe(
        pd.DataFrame(
            {
                "Naive Bayes": metrik["naive_bayes"],
                "SVM": metrik["svm"],
            }
        )
        .rename(index={"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1-score"})
        .style.format("{:.4f}"),
        width="stretch",
    )
    st.caption(f"{metrik['jumlah_test']} email testing, {metrik['jumlah_train']} email training.")

    st.subheader("Keterbatasan")
    st.caption(
        "Dataset adalah email bahasa Inggris (korpus Enron) yang diterjemahkan ke bahasa "
        "Indonesia, sehingga model bisa kurang akurat untuk email atau SMS asli berbahasa "
        "Indonesia maupun pesan yang sangat pendek."
    )

st.title("📧 Deteksi Email Spam")
st.write("Tempel isi email berbahasa Indonesia, lalu lihat prediksi kedua model.")

kolom = st.columns(len(CONTOH_EMAIL))
for kol, (label, teks) in zip(kolom, CONTOH_EMAIL.items()):
    kol.button(label, on_click=isi_contoh, args=(teks,), width="stretch")

email = st.text_area("Isi email", key="email", height=180, placeholder="Tulis atau tempel isi email di sini...")

if st.button("Periksa", type="primary", width="stretch"):
    teks_siap = siapkan_teks(email)

    if not teks_siap:
        st.warning("Teks kosong setelah dibersihkan. Masukkan email yang berisi kata-kata.")
        st.stop()

    jumlah_dikenal = model["Multinomial Naive Bayes"].named_steps["tfidf"].transform([teks_siap]).nnz
    if jumlah_dikenal == 0:
        st.warning("Tidak ada kata di email ini yang dikenal model, sehingga hasil prediksi tidak bisa diandalkan.")

    kiri, kanan = st.columns(2)
    for kol, (nama, pipeline) in zip((kiri, kanan), model.items()):
        with kol:
            tampilkan_hasil(nama, pipeline, teks_siap)

    with st.expander("Lihat teks setelah prapemrosesan"):
        st.code(teks_siap, language=None)
