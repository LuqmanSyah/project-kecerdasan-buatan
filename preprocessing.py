"""Prapemrosesan teks, sama persis dengan yang dipakai di notebook.

Dipakai bersama oleh train.py dan app.py supaya teks yang diprediksi
diproses dengan cara yang sama seperti data training.
"""

import re

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

DAFTAR_STOPWORD = set(StopWordRemoverFactory().get_stop_words())


def bersihkan_teks(teks):
    teks = str(teks).lower()

    # hapus URL
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)

    # hapus alamat email
    teks = re.sub(r"\S+@\S+", " ", teks)

    # hanya sisakan huruf (angka dan tanda baca dibuang)
    teks = re.sub(r"[^a-z\s]", " ", teks)

    # rapikan spasi
    teks = re.sub(r"\s+", " ", teks).strip()

    return teks


def hapus_stopword(teks):
    return " ".join(kata for kata in teks.split() if kata not in DAFTAR_STOPWORD)


def siapkan_teks(teks):
    """Pipeline prapemrosesan model final (tanpa stemming)."""
    return hapus_stopword(bersihkan_teks(teks))
