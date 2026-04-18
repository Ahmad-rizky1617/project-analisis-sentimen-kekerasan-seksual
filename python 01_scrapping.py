
"""
FILE: 01_scraping.py
TUJUAN: Mengambil tweet dari Twitter/X API v2 tentang PPKS & kekerasan seksual kampus.
        Jika tidak ada API key, otomatis generate dataset synthetic yang realistis.

JALANKAN: python 01_scraping.py
OUTPUT  : data/raw_tweets.csv
"""

import os
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Buat folder data jika belum ada
os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# KONFIGURASI — ISI BEARER TOKEN KAMU DI SINI
# ═══════════════════════════════════════════════════════════════
BEARER_TOKEN = ""   # ← Paste token kamu. Kosongkan = pakai synthetic data

# Parameter scraping
SEARCH_CONFIG = {
    # Query Boolean: sesuaikan jika mau tambah/kurangi keyword
    "query": """
        (kekerasan seksual OR pelecehan seksual OR PPKS OR "kampus aman" 
         OR predator dosen OR satgas ppks OR kkerasan seksual OR victim blaming kampus)
        (kampus OR universitas OR mahasiswa OR PTN OR PTS OR dosen)
        -is:retweet
        lang:id
    """,
    "start_time": "2021-11-01T00:00:00Z",   # Sejak Permendikbud PPKS disahkan
    "end_time":   "2024-06-30T23:59:59Z",
    "max_results_per_page": 100,             # Max 100 per request (limit API)
    "total_target": 500,                     # Total tweet yang mau dikumpulkan
    "tweet_fields": "created_at,public_metrics,geo,lang",
    "user_fields":  "name,username,created_at,public_metrics,protected",
    "expansions":   "author_id",
}


# ═══════════════════════════════════════════════════════════════
# BAGIAN 1: SCRAPING DARI TWITTER API (jika ada token)
# ═══════════════════════════════════════════════════════════════

def check_tweepy_available():
    """Cek apakah tweepy sudah terinstall."""
    try:
        import tweepy
        return True
    except ImportError:
        print("⚠️  tweepy belum terinstall. Jalankan: pip install tweepy")
        return False


def is_likely_bot(user: dict) -> bool:
    """
    Heuristik sederhana untuk filter akun bot.
    Return True jika akun mencurigakan (bot).
    """
    signals = 0

    # Rasio following/follower terlalu tinggi
    followers = user.get("public_metrics", {}).get("followers_count", 1)
    following = user.get("public_metrics", {}).get("following_count", 0)
    if followers > 0 and (following / max(followers, 1)) > 10:
        signals += 1

    # Akun sangat baru (< 30 hari)
    created_str = user.get("created_at", "")
    if created_str:
        try:
            created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            age_days = (datetime.now(created.tzinfo) - created).days
            if age_days < 30:
                signals += 1
        except Exception:
            pass

    # Tidak ada deskripsi (bio kosong)
    if not user.get("description", "").strip():
        signals += 1

    return signals >= 2   # Bot jika ≥ 2 sinyal


def anonymize_user(user_id: str) -> str:
    """
    Anonimisasi user ID menggunakan hash SHA-256.
    Sesuai UU PDP No. 27/2022.
    """
    import hashlib
    return hashlib.sha256(f"{user_id}_ppks_salt_2024".encode()).hexdigest()[:12]


def scrape_from_api() -> pd.DataFrame:
    """Scraping tweet dari Twitter API v2 via tweepy."""
    import tweepy

    print("🔗 Menghubungkan ke Twitter API v2...")
    client = tweepy.Client(
        bearer_token=BEARER_TOKEN,
        wait_on_rate_limit=True   # Auto tunggu jika kena rate limit
    )

    # Build users lookup dict
    users_dict = {}
    all_tweets = []
    next_token = None
    collected = 0

    print(f"🔍 Mulai scraping (target: {SEARCH_CONFIG['total_target']} tweet)...")

    while collected < SEARCH_CONFIG["total_target"]:
        try:
            response = client.search_all_tweets(
                query=SEARCH_CONFIG["query"].replace("\n", " ").strip(),
                start_time=SEARCH_CONFIG["start_time"],
                end_time=SEARCH_CONFIG["end_time"],
                max_results=SEARCH_CONFIG["max_results_per_page"],
                tweet_fields=SEARCH_CONFIG["tweet_fields"],
                user_fields=SEARCH_CONFIG["user_fields"],
                expansions=SEARCH_CONFIG["expansions"],
                next_token=next_token,
            )

            if not response.data:
                print("   Tidak ada data lagi.")
                break

            # Build user lookup dari includes
            if response.includes and "users" in response.includes:
                for user in response.includes["users"]:
                    users_dict[user.id] = user.data

            # Proses setiap tweet
            for tweet in response.data:
                user_data = users_dict.get(tweet.author_id, {})

                # Skip bot
                if is_likely_bot(user_data):
                    continue

                # Filter interaksi minimal (hindari noise)
                metrics = tweet.public_metrics or {}
                if (metrics.get("like_count", 0) < 3 and
                        metrics.get("retweet_count", 0) < 1):
                    continue

                all_tweets.append({
                    "id":         anonymize_user(str(tweet.id)),
                    "text":       tweet.text,
                    "date":       str(tweet.created_at)[:10],
                    "likes":      metrics.get("like_count", 0),
                    "retweets":   metrics.get("retweet_count", 0),
                    "replies":    metrics.get("reply_count", 0),
                    "city":       "Indonesia",  # Geo jarang tersedia
                    "source":     "twitter_api",
                })
                collected += 1

            print(f"   ✓ Terkumpul: {collected} tweet")

            # Cek apakah ada halaman selanjutnya
            if response.meta and "next_token" in response.meta:
                next_token = response.meta["next_token"]
                time.sleep(1)   # Jeda sopan antara request
            else:
                break

        except Exception as e:
            print(f"   ⚠️  Error API: {e}")
            print("   Menunggu 60 detik sebelum retry...")
            time.sleep(60)
            break

    return pd.DataFrame(all_tweets)


# ═══════════════════════════════════════════════════════════════
# BAGIAN 2: SYNTHETIC DATASET (fallback jika tidak ada API)
# ═══════════════════════════════════════════════════════════════

def generate_synthetic_dataset(n: int = 1000) -> pd.DataFrame:
    """
    Generate dataset synthetic yang realistis merepresentasikan
    tweet tentang PPKS. Digunakan untuk development & demo.
    """
    print("📦 Menggunakan dataset synthetic (representasi data nyata)...")

    random.seed(42)
    np.random.seed(42)

    # Template tweet per kategori
    TEMPLATES = {
        "negative_fear": [
            "korban takut lapor ke satgas ppks karena ancaman dari pelaku yg dosen",
            "mahasiswi takut melapor kkerasan seksual di kampus karena takut dikeluarkan",
            "siapa yg berani lapor pelecehan seksual kalo pelakunya dosen pembimbing skripsi",
            "sistem perlindungan korban di kampus cuma lip service doang gak ada actionnya",
            "korban bungkam krn kampus lebih lindungi nama baik institusi drpd mahasiswa",
            "satgas ppks kampus gw cuma formalitas, ga ada yg beneran bantu korban",
            "takut lapor kkerasan seksual krn pasti di victim blaming sama pihak kampus",
            "bukti kgk diproses, pelapor malah dikucilkan teman2nya, ini beneran terjadi",
            "mahasiswi yg lapor malah disuruh damai aja sama rektorat, parah banget sistemnya",
            "korban pelecehan seksual kampus dipaksa tanda tangan perjanjian bungkam wtf",
            "gimana mau lapor kalau dosen yg lakokin adalah orang paling berkuasa di jurusan",
            "trauma lapor kkerasan seksual ke kampus tapi malah dipermasalahkan balik",
            "tidak ada safe space sama sekali buat korban pelecehan seksual di kampus ini",
            "korban pelecehan seksual kampus sering kali dipaksa diam demi kebaikan bersama",
            "rasa takut itu nyata, lapor justru bikin situasi korban makin susah",
        ],
        "negative_sanction": [
            "sudah lapor ke satgas ppks tapi pelaku dosen masih ngajar biasa aja ga ada sanksi",
            "sanksi PPKS tidak transparan sama sekali, tidak ada kejelasan hukuman untuk pelaku",
            "dosen predator kampus ptn masih aktif mengajar padahal sdh banyak korban lapor",
            "permendikbud ppks bagus di atas kertas tapi implementasinya nol besar",
            "sudah 6 bulan lapor, kasusnya masih diproses katanya, pelaku masih bebas ngajar",
            "kampus tutup mata soal pelecehan seksual dosen, reputasi lebih penting dari korban",
            "kemendikbud harus audit semua satgas ppks kampus yg tidak berfungsi dengan baik",
            "regulasi ppks tidak punya gigi hukum yang kuat, pelaku gampang lolos dari sanksi",
            "rektorat kampus lebih takut heboh di medsos drpd ngasih sanksi tegas ke pelaku",
            "tidak ada transparansi sama sekali soal proses penanganan kasus pelecehan seksual",
            "pelaku pelecehan seksual di kampus lolos terus karena tidak ada mekanisme sanksi",
            "PPKS hanya jadi regulasi di atas kertas saja, eksekusinya tidak ada sama sekali",
            "sudah laporkan ke satgas tapi tidak ada kabar sampai sekarang, kemana hasilnya",
            "dosen yg dilaporkan malah dipindah jurusan bukan dipecat, itu bukan sanksi",
            "transparansi dalam proses hukum PPKS sangat buruk, korban tidak tahu perkembangan",
        ],
        "negative_blaming": [
            "korban pelecehan seksual kampus malah disalahkan krn pakaiannya, victim blaming parah",
            "teman2 di kampus justru membela pelaku pelecehan seksual, korban dikucilkan",
            "media sosial kampus penuh victim blaming saat ada kasus kekerasan seksual muncul",
            "dosen senior bilang korban pelecehan seksual harusnya jaga diri sendiri, gila ga sih",
            "atmosfer kampus gw super toxic, korban yg lapor malah dianggap tukang bikin masalah",
            "lingkungan kampus normalize pelecehan seksual, dianggap bercandaan biasa aja",
            "kalo korban cerita di medsos langsung dikeroyok netizen kampus, gaslight abis",
            "budaya patriarki di kampus bikin korban pelecehan seksual selalu disalahkan",
            "senior cowok di jurusan gw justify pelecehan seksual dengan bercanda doang kok",
            "victim blaming paling parah itu justru dari sesama mahasiswi, internalized misogyny",
            "komunitas kampus masih menganggap korban pelecehan sebagai pembuat onar",
            "pelaku mendapat simpati sementara korban dikucilkan, ini realita kampus kita",
            "korban disalahkan atas pakaian dan perilakunya, bukan pelaku yg bertanggung jawab",
            "normalisasi pelecehan seksual di lingkungan kampus adalah masalah budaya serius",
            "tidak ada empati untuk korban, justru penghakiman yang mereka terima",
        ],
        "positive_support": [
            "akhirnya ada teman yg berani share pengalaman kkerasan seksual di kampus, salut",
            "satgas ppks kampus gw aktif dan responsif, korban dibantu dengan baik puji syukur",
            "gerakan kampus aman mulai berdampak, banyak mahasiswi berani speak up sekarang",
            "solidaritas sesama perempuan di kampus makin kuat utk lawan pelecehan seksual",
            "ppks seharusnya jadi standar minimum setiap kampus, dukungan penuh untuk regulasi",
            "teman2 di bem kampus gw gencar sosialisasi ppks dan konseling gratis utk korban",
            "kampus gw akhirnya pecat dosen predator setelah mahasiswi berani lapor bareng",
            "semangat buat semua korban yg berani speak up, kalian tidak sendirian",
            "komunitas kampus aman sangat membantu korban yg tidak tau harus kemana melapor",
            "progres bagus saat kampus mau transparan soal penanganan kasus pelecehan seksual",
            "berkat PPKS akhirnya ada mekanisme formal yg bisa melindungi mahasiswi",
            "gerakan solidaritas untuk korban kekerasan seksual kampus terus berkembang",
            "satgas ppks yang kompeten sangat dibutuhkan dan beberapa kampus sudah menunjukkan",
            "dukungan psikolog kampus sangat membantu korban dalam proses pemulihan",
            "kesadaran tentang consent dan PPKS semakin meningkat di kalangan mahasiswa baru",
        ],
        "neutral_policy": [
            "permendikbud ppks no 30 2021 perlu direvisi agar lebih berpihak ke korban",
            "kemendikbud harus evaluasi implementasi ppks di seluruh perguruan tinggi indonesia",
            "satgas ppks perlu diperkuat dengan anggaran dan sdm yang memadai dari pusat",
            "perlu regulasi yang lebih spesifik soal mekanisme sanksi dalam kasus ppks",
            "riset menunjukkan 1 dari 3 mahasiswi pernah alami pelecehan seksual di kampus",
            "diskusi panel tentang reformasi ppks di perguruan tinggi indonesia perlu dilanjutkan",
            "data kemenangan hukum korban pelecehan seksual kampus masih sangat rendah",
            "perlu ada hotline nasional khusus korban kekerasan seksual di perguruan tinggi",
            "implementasi ppks harus jadi syarat akreditasi perguruan tinggi oleh ban pt",
            "anggaran satgas ppks di banyak kampus masih sangat minim perlu ditingkatkan",
        ],
    }

    CITIES = ["Jakarta","Bandung","Yogyakarta","Surabaya","Malang",
              "Semarang","Medan","Makassar","Denpasar","Bogor"]
    CITY_W  = [0.25,0.18,0.15,0.12,0.08,0.07,0.06,0.04,0.03,0.02]

    # Distribusi berdasarkan realitas (banyak keluhan)
    CAT_SIZES = {
        "negative_fear":    280,
        "negative_sanction":260,
        "negative_blaming": 200,
        "positive_support": 150,
        "neutral_policy":   110,
    }

    HASHTAG_VARIANTS = [
        "#PPKS #KampusAman",
        "#MahasiswaAman #PPKS",
        "#StopKekerasanSeksual",
        "#Permendikbud #PPKS",
        "",  # Tanpa hashtag (realistis)
    ]

    dates = pd.date_range("2021-11-01", "2024-06-30", freq="D")
    rows = []

    for cat, total_cat in CAT_SIZES.items():
        templates = TEMPLATES[cat]
        for i in range(total_cat):
            base = templates[i % len(templates)]
            # Variasikan sedikit agar tidak duplikat
            hashtag = random.choice(HASHTAG_VARIANTS)
            text = f"{base} {hashtag}".strip()
            tweet_date = random.choice(dates)

            rows.append({
                "id":       f"anon_{cat[:3]}_{i:04d}",
                "text":     text,
                "date":     str(tweet_date.date()),
                "likes":    int(np.random.lognormal(2.0, 1.2)),   # Distribusi realistis
                "retweets": int(np.random.lognormal(1.0, 1.0)),
                "replies":  int(np.random.lognormal(0.8, 0.8)),
                "city":     np.random.choice(CITIES, p=CITY_W),
                "category": cat,    # Label ground truth (untuk evaluasi)
                "source":   "synthetic",
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   ✓ Dataset synthetic: {len(df)} tweet dari {df['city'].nunique()} kota")
    return df


# ═══════════════════════════════════════════════════════════════
# MAIN — PILIH SCRAPING ATAU SYNTHETIC OTOMATIS
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("   FASE 1 — DATA COLLECTION")
    print("=" * 55)

    if BEARER_TOKEN and check_tweepy_available():
        print("🌐 Mode: LIVE SCRAPING dari Twitter API")
        df = scrape_from_api()
        if df.empty:
            print("⚠️  Scraping gagal/kosong. Fallback ke synthetic.")
            df = generate_synthetic_dataset()
    else:
        print("💾 Mode: SYNTHETIC DATASET (API key tidak ada/tweepy belum install)")
        df = generate_synthetic_dataset()

    # Simpan hasil
    output_path = "data/raw_tweets.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Data tersimpan → {output_path}")
    print(f"   Jumlah tweet : {len(df):,}")
    print(f"   Kolom        : {list(df.columns)}")
    print(f"   Rentang waktu: {df['date'].min()} s/d {df['date'].max()}")
    print("\n▶ Langkah berikutnya: python 02_preprocessing.py")


if __name__ == "__main__":
    main()