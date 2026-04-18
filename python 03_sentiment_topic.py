
"""
FILE: 03_sentiment_topic.py
TUJUAN: Analisis sentimen (IndoBERT / rule-based fallback) + topic modeling (LDA).
        Menambahkan kolom 'sentiment' dan 'topic' ke dataset.

JALANKAN: python 03_sentiment_topic.py
INPUT   : data/clean_tweets.csv
OUTPUT  : data/analyzed_tweets.csv

CATATAN MODEL:
  - Mode A: IndoBERT via HuggingFace (akurat, butuh ~2GB RAM, download ~500MB)
  - Mode B: Rule-Based + TF-IDF (cepat, tidak butuh internet, akurasi lebih rendah)
  Script otomatis pilih Mode A jika transformers terinstall, B jika tidak.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report

os.makedirs("data", exist_ok=True)

# ─── Cek ketersediaan transformers (IndoBERT) ──────────────
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_OK = True
    print("✅ transformers & torch tersedia → Mode IndoBERT aktif")
except ImportError:
    TRANSFORMERS_OK = False
    print("⚠️  transformers/torch tidak tersedia → Mode Rule-Based aktif")
    print("   Install IndoBERT: pip install transformers torch sentencepiece")


# ═══════════════════════════════════════════════════════════════
# NAMA TOPIK — Sesuaikan dengan hasil topik yang muncul di datamu
# ═══════════════════════════════════════════════════════════════
TOPIC_NAMES = {
    0: "Ketakutan Melapor & Intimidasi",
    1: "Transparansi Sanksi & Akuntabilitas",
    2: "Victim Blaming & Budaya Toxic",
    3: "Dukungan Solidaritas & Gerakan",
    4: "Reformasi Kebijakan & Regulasi",
}


# ═══════════════════════════════════════════════════════════════
# MODE A: ANALISIS SENTIMEN DENGAN INDOBERT
# ═══════════════════════════════════════════════════════════════

def load_indobert_pipeline():
    """
    Load model IndoBERT sentiment analysis dari HuggingFace.
    Model: w11wo/indonesian-roberta-base-sentiment-classifier
    Label: positif, negatif, netral
    
    Download pertama kali ~500MB, setelah itu cached otomatis.
    """
    print("   📥 Loading IndoBERT... (download ~500MB jika pertama kali, sabar ya)")
    MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"
    try:
        classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=-1,          # -1 = CPU (aman untuk semua laptop)
            truncation=True,
            max_length=128,
        )
        print("   ✅ IndoBERT berhasil dimuat!")
        return classifier
    except Exception as e:
        print(f"   ⚠️  Gagal load IndoBERT: {e}")
        print("   Fallback ke Rule-Based...")
        return None


def predict_indobert(texts: list, classifier, batch_size: int = 32) -> list:
    """
    Prediksi sentimen menggunakan IndoBERT.
    Memproses dalam batch agar efisien di memory.
    """
    results = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        # Potong teks panjang (IndoBERT max 512 token)
        batch_truncated = [t[:400] if len(t) > 400 else t for t in batch]
        preds = classifier(batch_truncated)
        for pred in preds:
            label = pred["label"].upper()
            # Normalisasi label ke format standar
            if "POS" in label:
                results.append("POSITIF")
            elif "NEG" in label:
                results.append("NEGATIF")
            else:
                results.append("NETRAL")

        # Progress setiap 100 tweet
        done = min(i + batch_size, total)
        if done % 100 == 0 or done == total:
            print(f"   Progress: {done}/{total} tweet dianalisis...")

    return results


# ═══════════════════════════════════════════════════════════════
# MODE B: ANALISIS SENTIMEN RULE-BASED (FALLBACK)
# ═══════════════════════════════════════════════════════════════

# Leksikon kata positif/negatif untuk Bahasa Indonesia
POSITIVE_WORDS = {
    "bagus","baik","berani","berhasil","bantuan","aman","dukungan",
    "mendukung","progress","solusi","responsif","aktif","melindungi",
    "transparansi","keadilan","harapan","semangat","solidaritas",
    "speak up","berubah","maju","berhasil","positif","sukses",
    "terima kasih","salut","membantu","berdampak","perubahan",
}

NEGATIVE_WORDS = {
    "takut","bungkam","ancaman","diam","tidak ada","gagal","lambat",
    "buruk","parah","kecewa","marah","frustasi","malu","trauma",
    "tidak transparan","tutup mata","impunitas","lolos","victim blaming",
    "disalahkan","dikucilkan","diancam","tidak diproses","formalitas",
    "lip service","nol","tidak berfungsi","toxic","predator","bungkam",
    "intimidasi","gaslight","normalize","zalim","tidak adil","diskriminasi",
}

INTENSIFIERS = {"sangat","banget","amat","sekali","benar","sungguh","bgt"}
NEGATIONS    = {"tidak","bukan","belum","tanpa","tak","non","anti","gak","ga"}


def predict_rule_based(text: str) -> str:
    """
    Prediksi sentimen berbasis leksikon + aturan.
    Lebih sederhana dari ML, tapi tidak butuh training data.
    """
    if not isinstance(text, str) or not text.strip():
        return "NETRAL"

    tokens = text.lower().split()
    token_set = set(tokens)

    # Hitung skor dasar
    pos_score = len(token_set & POSITIVE_WORDS)
    neg_score = len(token_set & NEGATIVE_WORDS)

    # Bonus untuk intensifier
    intensifier_count = len(token_set & INTENSIFIERS)
    dominant_sentiment = "POSITIF" if pos_score > neg_score else "NEGATIF"
    if dominant_sentiment == "POSITIF":
        pos_score += intensifier_count * 0.5
    else:
        neg_score += intensifier_count * 0.5

    # Penanganan negasi ("tidak bagus" → negatif)
    for i, token in enumerate(tokens[:-1]):
        if token in NEGATIONS:
            next_token = tokens[i + 1]
            if next_token in POSITIVE_WORDS:
                pos_score -= 1
                neg_score += 1

    # Marker sarkasme: balik sentimen jika ada
    if "[SARKASME]" in text:
        pos_score, neg_score = neg_score, pos_score

    # Keputusan final
    if pos_score > neg_score:
        return "POSITIF"
    elif neg_score > pos_score:
        return "NEGATIF"
    else:
        return "NETRAL"


# ═══════════════════════════════════════════════════════════════
# TOPIC MODELING — LDA
# ═══════════════════════════════════════════════════════════════

def run_lda(texts: list, n_topics: int = 5) -> tuple:
    """
    Jalankan Latent Dirichlet Allocation untuk topic modeling.
    
    Returns:
        topic_labels: list label topik per tweet
        topic_names: dict mapping topic_id → nama topik
        top_words: dict mapping topic_id → list kata kunci
        vectorizer: TfidfVectorizer yang sudah fit
        lda_model: model LDA yang sudah fit
    """
    print("   Membuat TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=800,
        min_df=3,           # Kata harus muncul di minimal 3 dokumen
        max_df=0.85,        # Abaikan kata yang muncul di >85% dokumen
        ngram_range=(1, 2), # Unigram + bigram
    )
    X = vectorizer.fit_transform(texts)
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_):,} kata")

    print(f"   Training LDA ({n_topics} topik)...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=30,
        learning_method="online",
        random_state=42,
        n_jobs=-1,          # Gunakan semua CPU core
    )
    lda.fit(X)

    # Assign topik ke setiap tweet
    topic_dist = lda.transform(X)
    topic_labels = topic_dist.argmax(axis=1).tolist()

    # Ambil kata kunci per topik
    feature_names = vectorizer.get_feature_names_out()
    top_words = {}
    for i, component in enumerate(lda.components_):
        top_idx = component.argsort()[-10:][::-1]
        top_words[i] = [feature_names[j] for j in top_idx]

    # Print ringkasan topik
    print("\n   ── Topik yang Ditemukan ──")
    for tid, words in top_words.items():
        name = TOPIC_NAMES.get(tid, f"Topik {tid}")
        print(f"   Topik {tid} | {name}")
        print(f"           Kata kunci: {', '.join(words[:7])}")

    return topic_labels, TOPIC_NAMES, top_words, vectorizer, lda


# ═══════════════════════════════════════════════════════════════
# EVALUASI (hanya jika ada ground truth label)
# ═══════════════════════════════════════════════════════════════

CATEGORY_TO_SENTIMENT = {
    "negative_fear":    "NEGATIF",
    "negative_sanction":"NEGATIF",
    "negative_blaming": "NEGATIF",
    "positive_support": "POSITIF",
    "neutral_policy":   "NETRAL",
}


def evaluate_sentiment(df: pd.DataFrame):
    """Hitung akurasi model jika kolom 'category' (ground truth) tersedia."""
    if "category" not in df.columns:
        return

    y_true = df["category"].map(CATEGORY_TO_SENTIMENT).dropna()
    y_pred = df.loc[y_true.index, "sentiment"]

    print("\n── Evaluasi Model Sentimen ──")
    print(classification_report(y_true, y_pred, zero_division=0))


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("   FASE 3 — ANALISIS SENTIMEN & TOPIC MODELING")
    print("=" * 55)

    # ── Load data ──────────────────────────────────────────────
    input_path = "data/clean_tweets.csv"
    if not os.path.exists(input_path):
        print(f"❌ File tidak ditemukan: {input_path}")
        print("   Jalankan dulu: python 02_preprocessing.py")
        sys.exit(1)

    print(f"📂 Memuat data dari {input_path}...")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    print(f"   {len(df):,} tweet siap dianalisis")

    # ── Analisis Sentimen ─────────────────────────────────────
    print("\n🧠 [1/2] Analisis Sentimen...")

    if TRANSFORMERS_OK:
        classifier = load_indobert_pipeline()
    else:
        classifier = None

    if classifier is not None:
        print("   Mode: IndoBERT")
        texts = df["clean_text"].fillna("").tolist()
        df["sentiment"] = predict_indobert(texts, classifier)
        df["sentiment_method"] = "indobert"
    else:
        print("   Mode: Rule-Based Lexicon")
        df["sentiment"] = df["clean_text"].apply(predict_rule_based)
        df["sentiment_method"] = "rule_based"

    # Distribusi sentimen
    sent_dist = df["sentiment"].value_counts()
    print(f"\n   Distribusi Sentimen:")
    for sent, count in sent_dist.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 3)
        print(f"   {sent:10s}: {count:4d} ({pct:.1f}%) {bar}")

    # ── Topic Modeling ────────────────────────────────────────
    print("\n🗂️  [2/2] Topic Modeling (LDA)...")
    texts = df["clean_text"].fillna("kosong").tolist()

    topic_labels, topic_names, top_words, vectorizer, lda = run_lda(
        texts, n_topics=5
    )

    df["topic_id"]   = topic_labels
    df["topic_name"] = df["topic_id"].map(topic_names)

    # ── Evaluasi ──────────────────────────────────────────────
    evaluate_sentiment(df)

    # ── Simpan hasil ──────────────────────────────────────────
    output_path = "data/analyzed_tweets.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Simpan juga ringkasan topik untuk visualisasi
    topic_summary = []
    for tid, words in top_words.items():
        topic_summary.append({
            "topic_id":    tid,
            "topic_name":  topic_names.get(tid, f"Topik {tid}"),
            "top_words":   ", ".join(words),
        })
    pd.DataFrame(topic_summary).to_csv(
        "data/topic_summary.csv", index=False, encoding="utf-8-sig"
    )

    print(f"\n✅ Hasil tersimpan → {output_path}")
    print(f"   Kolom baru: 'sentiment', 'topic_id', 'topic_name'")
    print("\n▶ Langkah berikutnya: python 04_visualisasi.py")


if __name__ == "__main__":
    main()