
"""
FILE: 02_preprocessing.py
TUJUAN: Membersihkan data mentah Twitter — normalisasi slang, hapus noise,
        stemming Bahasa Indonesia, dan deteksi sarkasme.

JALANKAN: python 02_preprocessing.py
INPUT   : data/raw_tweets.csv
OUTPUT  : data/clean_tweets.csv
"""

import os
import re
import sys
import pandas as pd

os.makedirs("data", exist_ok=True)

# ─── Cek Sastrawi (stemmer Indo) ───────────────────────────
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    STEMMER = factory.create_stemmer()
    SASTRAWI_OK = True
except ImportError:
    print("⚠️  PySastrawi belum terinstall. Stemming dilewati.")
    print("   Jalankan: pip install PySastrawi")
    STEMMER = None
    SASTRAWI_OK = False

# ─── Cek emoji library ─────────────────────────────────────
try:
    import emoji as emoji_lib
    EMOJI_OK = True
except ImportError:
    print("⚠️  emoji belum terinstall. Konversi emoji dilewati.")
    EMOJI_OK = False


# ═══════════════════════════════════════════════════════════════
# KAMUS NORMALISASI — Edit sesuai temuan di datamu
# ═══════════════════════════════════════════════════════════════

# Normalisasi kata tersensor (penting untuk tweet sensitif)
CENSORED_PATTERNS = [
    (r'k[*_\-]?k[e3]r[a4]s[a4]n', 'kekerasan'),
    (r'p[e3]l[e3]c[e3]h[a4]n',    'pelecehan'),
    (r'p[e3]rk[o0]s[a4]',          'perkosa'),
    (r's[e3]k[s$][uv][a4]l',       'seksual'),
    (r'p[*_]l[*_]c[*_]h[*_]n',    'pelecehan'),
]

# Kamus slang & singkatan Twitter Indonesia
SLANG_DICT = {
    # Sensor & eufemisme umum
    "kkerasan":  "kekerasan",
    "pelecehn":  "pelecehan",
    "seksul":    "seksual",
    "kkersan":   "kekerasan",
    "vict1m":    "victim",
    "korbn":     "korban",
    # Slang percakapan
    "gak": "tidak",  "ga": "tidak",   "gk":  "tidak",
    "gw":  "saya",   "gue": "saya",   "lo":  "kamu",
    "lu":  "kamu",   "yg":  "yang",   "dgn": "dengan",
    "utk": "untuk",  "krn": "karena", "bgt": "banget",
    "sdh": "sudah",  "udh": "sudah",  "udah":"sudah",
    "kgk": "tidak",  "drpd":"daripada","pd":  "pada",
    "bs":  "bisa",   "blm": "belum",  "jg":  "juga",
    "aja": "saja",   "nih": "",       "sih": "",
    "dong":"",       "deh": "",       "nah": "",
    "yah": "",       "loh": "",       "kok": "",
    # Institusi
    "ptn":    "perguruan tinggi negeri",
    "pts":    "perguruan tinggi swasta",
    "bem":    "badan eksekutif mahasiswa",
    "bem km": "badan eksekutif mahasiswa",
    "ppks":   "ppks",   # Pertahankan akronim penting
    # Sarkasme markers — JANGAN dihapus, penting untuk sentimen
    "wkwk":   "[SARKASME]",
    "wkwkwk": "[SARKASME]",
    "hahaha": "[SARKASME]",
    "haha":   "[SARKASME]",
    "lol":    "[SARKASME]",
    "anjir":  "[EKSPRESI_KUAT]",
    "gila":   "[EKSPRESI_KUAT]",
    "parah":  "[EKSPRESI_KUAT]",
    "wtf":    "[EKSPRESI_KUAT]",
}

# Stopwords Bahasa Indonesia (dikurangi agar tidak hapus kata bermakna)
STOPWORDS = {
    "yang","di","ke","dari","dan","atau","ini","itu","ada","juga",
    "dengan","untuk","pada","adalah","jadi","bisa","akan","sudah",
    "tidak","aja","kok","nih","sih","ya","yah","tapi","kalau","kalo",
    "karena","lebih","banyak","masih","saja","satu","dua","tiga",
    "kami","kita","mereka","dia","nya","mu","ku","si","sang","para",
    "jika","maka","agar","supaya","bahwa","namun","tetapi","walau",
    "meski","malah","justru","bahkan","pun","lagi","lalu","kemudian",
}


# ═══════════════════════════════════════════════════════════════
# FUNGSI PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def remove_pii(text: str) -> str:
    """
    Hapus Personally Identifiable Information (PII).
    Sesuai UU PDP No. 27/2022.
    """
    text = re.sub(r'@\w+', '[USER]', text)               # Mention
    text = re.sub(r'\b(\+62|0)[0-9]{8,13}\b', '[PHONE]', text)  # No telp
    text = re.sub(r'\S+@\S+\.\w+', '[EMAIL]', text)      # Email
    text = re.sub(r'\b[0-9]{16}\b', '[NIK]', text)        # NIK
    return text


def decode_emoji(text: str) -> str:
    """Konversi emoji ke kata deskriptif (informatif untuk sentimen)."""
    if EMOJI_OK:
        return emoji_lib.demojize(text, language='id')
    return text


def normalize_censored(text: str) -> str:
    """Pulihkan kata-kata yang disensor/dimodifikasi pengguna."""
    for pattern, replacement in CENSORED_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_repeated_chars(text: str) -> str:
    """'kampuuuus' → 'kampuss' (max 2 karakter berulang)."""
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def normalize_slang(text: str) -> str:
    """Normalisasi slang dan singkatan Twitter Indonesia."""
    tokens = text.split()
    normalized = []
    for token in tokens:
        replacement = SLANG_DICT.get(token, token)
        if replacement:   # Hapus token jika nilai di dict kosong ""
            normalized.append(replacement)
    return ' '.join(normalized)


def remove_stopwords(text: str) -> str:
    """Hapus stopwords dari teks."""
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return ' '.join(tokens)


def stem_text(text: str) -> str:
    """Stemming Bahasa Indonesia menggunakan Sastrawi."""
    if SASTRAWI_OK and STEMMER:
        return STEMMER.stem(text)
    return text   # Skip jika Sastrawi tidak ada


def detect_sarcasm(text: str) -> int:
    """
    Deteksi sarkasme sederhana berbasis rule.
    Return 1 jika ada indikasi sarkasme, 0 jika tidak.
    """
    sarcasm_indicators = [
        '[SARKASME]',
        'dong bagus banget',
        'mantap banget ini',
        'oh tentu saja',
        'jelas sekali',
        'pasti dong',
    ]
    text_lower = text.lower()
    return int(any(ind in text_lower for ind in sarcasm_indicators))


def full_pipeline(text: str) -> str:
    """
    Pipeline preprocessing lengkap.
    Urutan pemrosesan sangat penting — jangan diubah.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Hapus PII
    text = remove_pii(text)
    # 2. Lowercase
    text = text.lower()
    # 3. Decode emoji → teks
    text = decode_emoji(text)
    # 4. Hapus URL
    text = re.sub(r'http\S+|www\S+', '', text)
    # 5. Hapus hashtag (simpan kata, hapus #)
    text = re.sub(r'#(\w+)', r'\1', text)
    # 6. Pulihkan kata tersensor
    text = normalize_censored(text)
    # 7. Normalisasi karakter berulang
    text = normalize_repeated_chars(text)
    # 8. Normalisasi slang
    text = normalize_slang(text)
    # 9. Hapus karakter non-alfanumerik (kecuali marker [])
    text = re.sub(r'[^a-z0-9\s\[\]_]', ' ', text)
    # 10. Hapus angka standalone (biasanya tidak informatif)
    text = re.sub(r'\b\d+\b', '', text)
    # 11. Hapus stopwords
    text = remove_stopwords(text)
    # 12. Stemming
    text = stem_text(text)
    # 13. Bersihkan whitespace berlebih
    text = ' '.join(text.split())

    return text


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("   FASE 2 — PREPROCESSING")
    print("=" * 55)

    # ── Load data ──────────────────────────────────────────────
    input_path = "data/raw_tweets.csv"
    if not os.path.exists(input_path):
        print(f"❌ File tidak ditemukan: {input_path}")
        print("   Jalankan dulu: python 01_scraping.py")
        sys.exit(1)

    print(f"📂 Memuat data dari {input_path}...")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"   Jumlah tweet mentah: {len(df):,}")

    # ── Hapus duplikat ─────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"   Duplikat dihapus   : {before - len(df)}")

    # ── Preprocessing ──────────────────────────────────────────
    print("\n🔧 Memproses teks...")
    df["clean_text"] = df["text"].apply(full_pipeline)
    df["sarcasm"]    = df["text"].apply(detect_sarcasm)

    # Hapus tweet yang jadi kosong setelah preprocessing
    before = len(df)
    df = df[df["clean_text"].str.len() > 5].reset_index(drop=True)
    print(f"   Tweet terlalu pendek dihapus: {before - len(df)}")

    # ── Info tambahan ──────────────────────────────────────────
    df["text_length"]       = df["clean_text"].apply(len)
    df["word_count"]        = df["clean_text"].apply(lambda x: len(x.split()))
    df["has_sarcasm"]       = df["sarcasm"].astype(bool)
    df["date"]              = pd.to_datetime(df["date"])
    df["month"]             = df["date"].dt.to_period("M").astype(str)
    df["engagement_score"]  = df["likes"] + df["retweets"] * 2 + df["replies"]

    # ── Simpan ─────────────────────────────────────────────────
    output_path = "data/clean_tweets.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ Data bersih tersimpan → {output_path}")
    print(f"   Tweet siap diproses: {len(df):,}")
    print(f"   Avg panjang teks   : {df['text_length'].mean():.0f} karakter")
    print(f"   Avg jumlah kata    : {df['word_count'].mean():.1f} kata")
    print(f"   Tweet sarkasme     : {df['has_sarcasm'].sum()} ({df['has_sarcasm'].mean()*100:.1f}%)")

    # ── Contoh hasil ───────────────────────────────────────────
    print("\n── Contoh Sebelum vs Sesudah Preprocessing ──")
    samples = df.sample(3, random_state=1)
    for _, row in samples.iterrows():
        print(f"  SEBELUM: {row['text'][:80]}...")
        print(f"  SESUDAH: {row['clean_text'][:80]}")
        print()

    print("▶ Langkah berikutnya: python 03_sentiment_topic.py")


if __name__ == "__main__":
    main()