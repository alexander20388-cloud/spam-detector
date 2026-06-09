# 🛡️ SpamShield — Sistem Deteksi Spam Berbasis Machine Learning

Aplikasi web deteksi spam menggunakan algoritma **Naive Bayes** dengan dataset bilingual (Indonesia & Inggris).

---

## 📋 Fitur

- 🔍 **Deteksi spam real-time** — analisis teks Bahasa Indonesia dan Inggris
- 📊 **Confidence Score** — persentase keyakinan model (spam vs normal)
- 📈 **Visualisasi lengkap** — Confusion Matrix, Metrics Chart, Top Words, Pie Chart
- 🕑 **Riwayat prediksi** — semua hasil tersimpan selama sesi berjalan
- ⬇️ **Export CSV** — unduh riwayat prediksi sebagai file CSV
- ℹ️ **Penjelasan model** — edukasi tentang cara kerja Naive Bayes

---

## ⚙️ Instalasi & Menjalankan

### 1. Pastikan Python sudah terinstall (versi 3.8+)
```bash
python --version
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan aplikasi
```bash
python app.py
```

### 4. Buka browser
Akses: **http://localhost:5000**

---

## 🗂️ Struktur Proyek

```
spam_detector/
├── app.py              ← Aplikasi utama (Flask + ML model)
├── requirements.txt    ← Daftar library yang dibutuhkan
├── README.md           ← Dokumentasi ini
└── templates/
    └── index.html      ← Tampilan web (Frontend)
```

---

## 🤖 Teknologi yang Digunakan

| Komponen        | Teknologi                    |
|-----------------|------------------------------|
| Backend         | Python, Flask                |
| Machine Learning| scikit-learn (Naive Bayes)   |
| Ekstraksi Fitur | TF-IDF Vectorizer (bigram)   |
| Visualisasi     | matplotlib, seaborn          |
| Frontend        | HTML, CSS, JavaScript        |

---

## 📊 Dataset

- **Total**: 80 pesan (40 spam + 40 ham)
- **Bahasa**: Indonesia (40) + Inggris (40)
- **Split**: 80% training / 20% testing

---

## 📐 Cara Kerja Naive Bayes

1. **Preprocessing** — teks dinormalisasi (lowercase, hapus URL & angka, bersihkan tanda baca)
2. **Vektorisasi TF-IDF** — teks dikonversi menjadi representasi numerik berbasis frekuensi kata
3. **Training** — model Naive Bayes belajar dari pola kata di data training
4. **Prediksi** — pesan baru diklasifikasi berdasarkan probabilitas P(spam|teks) vs P(ham|teks)
5. **Confidence Score** — probabilitas dari kedua kelas ditampilkan sebagai persentase

---

## 💡 Tips Penggunaan

- Gunakan **Ctrl + Enter** di area teks untuk langsung menganalisis
- Klik chip contoh teks untuk mencoba sampel spam/ham
- Tab **Visualisasi** menampilkan semua grafik evaluasi model
- Tab **Riwayat** menyimpan semua prediksi yang sudah dilakukan
- Gunakan tombol **Export CSV** untuk menyimpan riwayat sebagai file

---

*Dibuat sebagai Tugas Besar AI — Sistem Deteksi Spam Berbasis Machine Learning*
