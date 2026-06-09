from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import json
import csv
import io
import base64
import re
import os
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter

app = Flask(__name__)

# ── Dataset fallback (digunakan jika file eksternal tidak ditemukan) ──────────
FALLBACK_DATASET = [
    # SPAM ID
    ("Selamat! Anda memenangkan hadiah Rp 50.000.000! Hubungi kami sekarang!", "spam"),
    ("Klik link ini untuk klaim hadiah gratis Anda hari ini!", "spam"),
    ("Pinjaman cepat cair tanpa jaminan! Hubungi segera!", "spam"),
    ("Anda terpilih sebagai pemenang undian berhadiah! Konfirmasi segera!", "spam"),
    ("Investasi untung 50% per bulan! Daftar sekarang sebelum terlambat!", "spam"),
    ("GRATIS pulsa Rp 100.000 untuk 100 pendaftar pertama! Klik sekarang!", "spam"),
    ("Lowongan kerja gaji 30 juta per bulan dari rumah! WA sekarang!", "spam"),
    ("Dana bantuan pemerintah Rp 25 juta menunggu Anda! Klaim segera!", "spam"),
    ("Rekening Anda akan diblokir! Verifikasi data segera di link berikut!", "spam"),
    ("Nomor Anda menang lotere! Kirim data KTP untuk proses pencairan!", "spam"),
    # HAM ID
    ("Hai, kamu sudah makan siang belum? Nanti kita ketemu di kantin ya.", "ham"),
    ("Besok ada rapat jam 9 pagi di ruang conference. Jangan lupa hadir.", "ham"),
    ("Ibu, aku pulang agak terlambat karena ada kegiatan ekstrakulikuler.", "ham"),
    ("Laporan sudah aku revisi, tolong dicek lagi ya pak sebelum dikumpulkan.", "ham"),
    ("Jadwal kuliah hari ini berubah, cek aplikasi akademik ya.", "ham"),
    ("Nilai ujian sudah keluar, lumayan bagus hasilnya alhamdulillah.", "ham"),
    ("Skripsi bab 3 sudah selesai, tinggal bab 4 dan 5 lagi.", "ham"),
    ("Ada yang bisa bantu aku jelaskan materi algoritma greedy ini?", "ham"),
    ("Hari Minggu kita piknik ke pantai yuk, ajak keluarga juga.", "ham"),
    ("Meeting project kita dimajukan ke hari Rabu, bisa hadir kan?", "ham"),
    # SPAM EN
    ("Congratulations! You've won a $1,000,000 prize! Claim now!", "spam"),
    ("FREE iPhone! Click here to claim your prize immediately!", "spam"),
    ("URGENT: Your account has been compromised. Click to verify now!", "spam"),
    ("Make $5000 per week working from home! No experience needed!", "spam"),
    ("WINNER! You have been chosen for our weekly draw! Reply to claim!", "spam"),
    ("Your PayPal account is suspended! Verify immediately or lose access!", "spam"),
    ("FREE gift card! Complete this survey and get $500 Amazon card!", "spam"),
    ("Crypto investment opportunity! 300% returns guaranteed!", "spam"),
    ("Lose 30 pounds in 30 days with this miracle weight loss pill!", "spam"),
    ("Nigerian prince needs your help transferring $10 million!", "spam"),
    # HAM EN
    ("Hey, are you coming to the study group tonight at the library?", "ham"),
    ("The meeting has been moved to Thursday at 2 PM. Please confirm.", "ham"),
    ("I finished the project report. Can you review it before submission?", "ham"),
    ("Thanks for your help yesterday. Really appreciate it!", "ham"),
    ("The professor extended the deadline to next Friday. Great news!", "ham"),
    ("Can you send me the notes from today's lecture? I was absent.", "ham"),
    ("Your package has been shipped and will arrive in 3-5 business days.", "ham"),
    ("Great job on the presentation today! The client loved it.", "ham"),
    ("The power will be out from 8 AM to 2 PM for maintenance.", "ham"),
    ("Looking forward to seeing you at the conference next week!", "ham"),
]

# ── Nama file dataset yang dicari di folder yang sama dengan app.py ───────────
EN_DATASET_FILE = "SMSSpamCollection"        # TSV dari UCI
ID_DATASET_FILE = "spam_indonesia.csv"       # CSV dari Kaggle

dataset_info = {
    "en_loaded": False,
    "id_loaded": False,
    "en_count": 0,
    "id_count": 0,
    "using_fallback": False,
}

def load_datasets():
    """
    Coba muat kedua dataset eksternal.
    Jika tidak ditemukan, gunakan fallback bawaan.
    """
    frames = []

    # ── Dataset Bahasa Inggris (TSV) ──────────────────────────────────────────
    en_path = os.path.join(os.path.dirname(__file__), EN_DATASET_FILE)
    if os.path.exists(en_path):
        try:
            df_en = pd.read_csv(en_path, sep='\t', header=None,
                                names=['label', 'text'], encoding='latin-1')
            df_en = df_en[['text', 'label']].dropna()
            df_en['label'] = df_en['label'].str.strip().str.lower()
            df_en = df_en[df_en['label'].isin(['spam', 'ham'])]
            frames.append(df_en)
            dataset_info["en_loaded"] = True
            dataset_info["en_count"] = len(df_en)
            print(f"[OK] Dataset Inggris dimuat: {len(df_en)} baris")
        except Exception as e:
            print(f"[WARN] Gagal baca {EN_DATASET_FILE}: {e}")
    else:
        print(f"[INFO] {EN_DATASET_FILE} tidak ditemukan. Taruh di folder yang sama dengan app.py.")

    # ── Dataset Bahasa Indonesia (CSV) ────────────────────────────────────────
    id_path = os.path.join(os.path.dirname(__file__), ID_DATASET_FILE)
    if os.path.exists(id_path):
        try:
            df_id = pd.read_csv(id_path, encoding='utf-8')
            # Normalisasi nama kolom (beberapa versi punya nama berbeda)
            df_id.columns = df_id.columns.str.strip().str.lower()
            if 'label' not in df_id.columns:
                # Cari kolom yang mungkin berisi label
                for col in df_id.columns:
                    if df_id[col].str.lower().isin(['spam','ham']).mean() > 0.5:
                        df_id = df_id.rename(columns={col: 'label'})
                        break
            if 'text' not in df_id.columns:
                text_col = [c for c in df_id.columns if c != 'label'][0]
                df_id = df_id.rename(columns={text_col: 'text'})
            df_id = df_id[['text', 'label']].dropna()
            df_id['label'] = df_id['label'].str.strip().str.lower()
            df_id = df_id[df_id['label'].isin(['spam', 'ham'])]
            frames.append(df_id)
            dataset_info["id_loaded"] = True
            dataset_info["id_count"] = len(df_id)
            print(f"[OK] Dataset Indonesia dimuat: {len(df_id)} baris")
        except Exception as e:
            print(f"[WARN] Gagal baca {ID_DATASET_FILE}: {e}")
    else:
        print(f"[INFO] {ID_DATASET_FILE} tidak ditemukan. Taruh di folder yang sama dengan app.py.")

    # ── Gabung atau fallback ──────────────────────────────────────────────────
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.dropna().drop_duplicates(subset=['text'])
        print(f"[OK] Total dataset gabungan: {len(df)} baris")
        return df
    else:
        print("[WARN] Tidak ada dataset eksternal. Menggunakan data bawaan (80 pesan).")
        dataset_info["using_fallback"] = True
        return pd.DataFrame(FALLBACK_DATASET, columns=["text", "label"])

# ── History untuk export CSV ──────────────────────────────────────────────────
prediction_history = []

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\d+', ' angka ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Train model ───────────────────────────────────────────────────────────────
def train_model():
    df = load_datasets()
    df["clean"] = df["text"].apply(preprocess)

    X = df["clean"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, pos_label="spam") * 100, 2),
        "recall": round(recall_score(y_test, y_pred, pos_label="spam") * 100, 2),
        "f1": round(f1_score(y_test, y_pred, pos_label="spam") * 100, 2),
        "cm": confusion_matrix(y_test, y_pred, labels=["spam", "ham"]).tolist(),
        "total_data": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "spam_count": int((df["label"] == "spam").sum()),
        "ham_count": int((df["label"] == "ham").sum()),
        "en_loaded": dataset_info["en_loaded"],
        "id_loaded": dataset_info["id_loaded"],
        "en_count": dataset_info["en_count"],
        "id_count": dataset_info["id_count"],
        "using_fallback": dataset_info["using_fallback"],
    }
    return model, vectorizer, df, metrics

model, vectorizer, df_data, metrics = train_model()

# ── Chart helpers ─────────────────────────────────────────────────────────────
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64

def make_confusion_matrix():
    cm = np.array(metrics["cm"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Spam", "Ham"],
                yticklabels=["Spam", "Ham"],
                linewidths=0.5, ax=ax, cbar=False,
                annot_kws={"size": 16, "weight": "bold"})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig_to_base64(fig)

def make_metrics_chart():
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]
    colors = ["#4F75FF", "#00C9A7", "#FF6B6B", "#FFC107"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor='white')
    ax.set_xlim(0, 115)
    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Score (%)", fontsize=11)
    ax.set_title("Model Performance Metrics", fontsize=13, fontweight="bold")
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    return fig_to_base64(fig)

def make_wordcloud_chart(label):
    texts = df_data[df_data["label"] == label]["clean"].tolist()
    all_words = " ".join(texts).split()
    # Filter kata pendek
    all_words = [w for w in all_words if len(w) > 3]
    counter = Counter(all_words)
    top_words = counter.most_common(20)

    if not top_words:
        return None

    words, counts = zip(*top_words)
    color = "#FF6B6B" if label == "spam" else "#00C9A7"

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = range(len(words))
    ax.barh(list(y_pos), list(counts), color=color, alpha=0.85, edgecolor='white')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(list(words), fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Frekuensi", fontsize=11)
    ax.set_title(f"Top 20 Kata — {'SPAM' if label == 'spam' else 'HAM (Normal)'}", 
                 fontsize=13, fontweight="bold", color=color)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig_to_base64(fig)

def make_dataset_pie():
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sizes = [metrics["spam_count"], metrics["ham_count"]]
    colors = ["#FF6B6B", "#00C9A7"]
    explode = (0.05, 0.05)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=["Spam", "Ham (Normal)"], colors=colors,
        autopct="%1.1f%%", explode=explode, startangle=90,
        textprops={"fontsize": 12}
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Distribusi Dataset", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig_to_base64(fig)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Teks tidak boleh kosong"}), 400

    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    classes = model.classes_.tolist()
    spam_idx = classes.index("spam")
    ham_idx = classes.index("ham")

    confidence_spam = round(float(proba[spam_idx]) * 100, 2)
    confidence_ham  = round(float(proba[ham_idx])  * 100, 2)

    # ── Bayes step-by-step ────────────────────────────────────────────
    steps = build_bayes_steps(clean, classes, spam_idx, ham_idx, proba)

    result = {
        "prediction": prediction,
        "confidence_spam": confidence_spam,
        "confidence_ham": confidence_ham,
        "text": text,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bayes_steps": steps,
    }

    prediction_history.append({k: v for k, v in result.items() if k != "bayes_steps"})
    return jsonify(result)


def build_bayes_steps(clean_text, classes, spam_idx, ham_idx, proba):
    """Return a structured breakdown of the Bayes calculation for display."""
    total   = len(df_data)
    n_spam  = int((df_data["label"] == "spam").sum())
    n_ham   = int((df_data["label"] == "ham").sum())
    p_spam  = round(n_spam / total, 4)
    p_ham   = round(n_ham  / total, 4)

    # Feature names & log-probs from the trained model
    feat_names  = vectorizer.get_feature_names_out()
    log_prob_spam = model.feature_log_prob_[spam_idx]   # shape (n_features,)
    log_prob_ham  = model.feature_log_prob_[ham_idx]

    # Which features are present in this text?
    vec_arr = vectorizer.transform([clean_text]).toarray()[0]
    present_indices = np.where(vec_arr > 0)[0]

    word_details = []
    for i in present_indices[:10]:   # cap at 10 words for readability
        word = feat_names[i]
        lp_s = round(float(log_prob_spam[i]), 4)
        lp_h = round(float(log_prob_ham[i]),  4)
        p_w_spam = round(float(np.exp(lp_s)), 6)
        p_w_ham  = round(float(np.exp(lp_h)), 6)
        word_details.append({
            "word":      word,
            "p_w_spam":  p_w_spam,
            "p_w_ham":   p_w_ham,
            "log_spam":  lp_s,
            "log_ham":   lp_h,
            "favors":    "spam" if lp_s > lp_h else "ham",
        })

    # Sort: most discriminating words first
    word_details.sort(key=lambda x: abs(x["log_spam"] - x["log_ham"]), reverse=True)

    return {
        "prior": {
            "total": total, "n_spam": n_spam, "n_ham": n_ham,
            "p_spam": p_spam, "p_ham": p_ham,
        },
        "words": word_details,
        "posterior": {
            "p_spam": round(float(proba[spam_idx]), 6),
            "p_ham":  round(float(proba[ham_idx]),  6),
        },
        "clean_text": clean_text,
    }

@app.route("/charts")
def charts():
    return jsonify({
        "confusion_matrix": make_confusion_matrix(),
        "metrics_chart": make_metrics_chart(),
        "wordcloud_spam": make_wordcloud_chart("spam"),
        "wordcloud_ham": make_wordcloud_chart("ham"),
        "dataset_pie": make_dataset_pie(),
    })

@app.route("/export")
def export():
    if not prediction_history:
        return jsonify({"error": "Belum ada data prediksi"}), 400

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "timestamp", "text", "prediction", "confidence_spam", "confidence_ham"
    ])
    writer.writeheader()
    for row in prediction_history:
        writer.writerow(row)

    output.seek(0)
    buf = io.BytesIO(output.getvalue().encode("utf-8"))
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True,
                     download_name=f"spam_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

@app.route("/history")
def history():
    return jsonify(prediction_history[-20:])

if __name__ == "__main__":
    app.run(debug=True, port=5000)
