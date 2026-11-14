# Klasifikasi Email Spam Bahasa Indonesia

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LuqmanSyah/project-kecerdasan-buatan/blob/main/Klasifikasi_Email_Spam_NB_SVM.ipynb)

Project akhir mata kuliah **Kecerdasan Buatan**: perbandingan **Multinomial Naive Bayes** dan **Support Vector Machine (SVM, kernel linear)** untuk mengklasifikasikan email berbahasa Indonesia menjadi **spam** atau **ham** (bukan spam), dengan fitur **TF-IDF**.

## Dataset

[Indonesian Email Spam](https://www.kaggle.com/datasets/gevabriel/indonesian-email-spam) (Kaggle, Gevabriel), dengan dua kolom: `Kategori` (spam/ham) dan `Pesan`.

| | Jumlah |
|---|---|
| Data awal | 2.636 |
| Setelah buang duplikat | 2.620 |
| Spam (label 1) | 1.368 |
| Ham (label 0) | 1.252 |

Data dibagi menjadi 80% training (2.096) dan 20% testing (524) dengan stratifikasi.

## Tahapan

1. **EDA**: distribusi kelas, nilai kosong, duplikat, dan panjang teks.
2. **Pembersihan data**: duplikat dibuang *sebelum* split supaya tidak terjadi kebocoran data.
3. **Prapemrosesan teks**: case folding, hapus URL/email/angka/tanda baca, stopword removal, dan stemming (Sastrawi).
4. **Ekstraksi fitur**: TF-IDF (`min_df=2`, `max_df=0.95`, `sublinear_tf=True`), di-fit hanya pada data training.
5. **Training dan evaluasi**: accuracy, precision, recall, F1-score, dan confusion matrix.
6. **5-fold stratified cross-validation**: TF-IDF dimasukkan ke dalam `Pipeline` sehingga dihitung ulang di setiap fold.
7. **Hyperparameter tuning**: `alpha` untuk Naive Bayes dan `C` untuk SVM, dipilih berdasarkan F1-score CV pada data training.
8. **Eksperimen tambahan**: pengaruh stemming dan bigram.
9. **Analisis**: kesalahan prediksi dan kata yang paling berpengaruh.
10. **Prediksi email baru dan penyimpanan model** (`.pkl`).

## Hasil Utama

Data testing (524 email), kelas spam sebagai kelas positif:

| Model | Parameter | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| Multinomial NB | alpha = 1.0 | 0,9885 | 0,9891 | 0,9891 | 0,9891 |
| Multinomial NB (tuning) | alpha = 0.05 | **0,9924** | **0,9927** | 0,9927 | **0,9927** |
| SVM Linear | C = 1 | 0,9866 | 0,9785 | **0,9964** | 0,9873 |
| SVM Linear (tuning) | C = 10 | 0,9828 | 0,9715 | **0,9964** | 0,9838 |

5-fold cross-validation pada data training (rata-rata ± std):

| Model | Accuracy | F1-score |
|---|---|---|
| Multinomial NB | 0,9857 ± 0,0050 | 0,9864 ± 0,0047 |
| SVM Linear | 0,9866 ± 0,0056 | 0,9873 ± 0,0052 |

Kedua model sama-sama mencapai akurasi sekitar 98–99%. Selisihnya kecil (beberapa email dari 524) dan masih di dalam rentang standar deviasi CV. Detail lengkap, termasuk pengaruh stemming/bigram dan analisis kesalahan, ada di notebook.

## Cara Menjalankan

**Google Colab (disarankan):** klik badge *Open in Colab* di atas, lalu jalankan semua sel (*Runtime → Run all*). Dataset diunduh otomatis lewat `kagglehub`.

**Lokal:**

```bash
pip install kagglehub PySastrawi scikit-learn pandas numpy matplotlib joblib jupyter
jupyter notebook Klasifikasi_Email_Spam_NB_SVM.ipynb
```

Proses stemming memakan waktu beberapa menit.

## Memakai Model yang Disimpan

Notebook menyimpan `model_naive_bayes_spam.pkl` dan `model_svm_spam.pkl`. Setiap file berisi pipeline TF-IDF dan model. Teks baru harus melewati prapemrosesan yang sama (`bersihkan_teks` lalu `hapus_stopword`, tanpa stemming) sebelum diprediksi:

```python
import joblib

model = joblib.load("model_naive_bayes_spam.pkl")
teks_siap = hapus_stopword(bersihkan_teks("Selamat! Anda memenangkan hadiah, klik tautan ini."))
model.predict([teks_siap])  # 1 = spam, 0 = ham
```

## Struktur Repository

```
.
├── Klasifikasi_Email_Spam_NB_SVM.ipynb   # notebook utama
└── README.md
```

## Teknologi

Python, scikit-learn, PySastrawi, pandas, NumPy, Matplotlib, kagglehub.
