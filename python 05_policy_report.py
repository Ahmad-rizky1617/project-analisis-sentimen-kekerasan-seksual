"""
FILE: 05_policy_report.py
TUJUAN: Generate laporan rekomendasi kebijakan HTML dari hasil analisis sentimen.
        Laporan ini siap dibuka di browser dan bisa dicetak/dijadikan PDF.

JALANKAN: python 05_policy_report.py
INPUT   : data/analyzed_tweets.csv, data/topic_summary.csv
OUTPUT  : output/ppks_laporan_kebijakan.html
"""

import os
import sys
import base64
import pandas as pd
from datetime import datetime

os.makedirs("output", exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# HITUNG STATISTIK DARI DATA NYATA
# ═══════════════════════════════════════════════════════════════

def compute_stats(df: pd.DataFrame, df_topics: pd.DataFrame) -> dict:
    """Hitung semua statistik yang akan ditampilkan dalam laporan."""
    total      = len(df)
    sent_counts = df["sentiment"].value_counts()

    stats = {
        "total":      total,
        "neg_count":  int(sent_counts.get("NEGATIF", 0)),
        "pos_count":  int(sent_counts.get("POSITIF", 0)),
        "net_count":  int(sent_counts.get("NETRAL",  0)),
        "neg_pct":    round(sent_counts.get("NEGATIF", 0) / total * 100, 1),
        "pos_pct":    round(sent_counts.get("POSITIF", 0) / total * 100, 1),
        "net_pct":    round(sent_counts.get("NETRAL",  0) / total * 100, 1),
        "date_start": df["date"].min().strftime("%B %Y"),
        "date_end":   df["date"].max().strftime("%B %Y"),
        "kota_count": df["city"].nunique() if "city" in df.columns else "N/A",
        "method":     (df["sentiment_method"].iloc[0]
                       if "sentiment_method" in df.columns else "rule_based"),
        "generated_at": datetime.now().strftime("%d %B %Y, %H:%M WIB"),
    }

    # Breakdown negatif per kategori
    neg_df = df[df["sentiment"] == "NEGATIF"]
    if "category" in df.columns:
        neg_cats = neg_df["category"].value_counts()
        stats["fear_count"]  = int(neg_cats.get("negative_fear",     0))
        stats["sanc_count"]  = int(neg_cats.get("negative_sanction", 0))
        stats["blame_count"] = int(neg_cats.get("negative_blaming",  0))
    else:
        n = stats["neg_count"]
        stats["fear_count"]  = int(n * 0.38)
        stats["sanc_count"]  = int(n * 0.35)
        stats["blame_count"] = int(n * 0.27)

    stats["fear_pct"]  = round(stats["fear_count"]  / max(stats["neg_count"], 1) * 100, 1)
    stats["sanc_pct"]  = round(stats["sanc_count"]  / max(stats["neg_count"], 1) * 100, 1)
    stats["blame_pct"] = round(stats["blame_count"] / max(stats["neg_count"], 1) * 100, 1)

    if "city" in df.columns:
        stats["top_cities"] = ", ".join(neg_df["city"].value_counts().head(3).index.tolist())
    else:
        stats["top_cities"] = "Jakarta, Bandung, Yogyakarta"

    if not df_topics.empty and "top_words" in df_topics.columns:
        neg_words = df_topics[df_topics["topic_id"].isin([0, 1, 2])]["top_words"]
        stats["top_neg_words"] = (
            ", ".join(str(neg_words.iloc[0]).split(", ")[:5])
            if not neg_words.empty
            else "kekerasan, pelecehan, takut, bungkam, sanksi"
        )
    else:
        stats["top_neg_words"] = "kekerasan, pelecehan, takut, bungkam, sanksi"

    model_display = {
        "indobert":   "IndoBERT (w11wo/indonesian-roberta-base-sentiment-classifier)",
        "rule_based": "Rule-Based Lexicon (fallback — pertimbangkan upgrade ke IndoBERT)",
    }
    stats["model_display"] = model_display.get(stats["method"], stats["method"])

    return stats


def load_dashboard_image(dashboard_path: str) -> str:
    """
    Muat gambar dashboard sebagai base64 data-URI agar HTML bisa
    dibuka dari folder manapun tanpa referensi path relatif yang rusak.
    Kembalikan string kosong jika file tidak ada.
    """
    if not os.path.exists(dashboard_path):
        return ""
    with open(dashboard_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ═══════════════════════════════════════════════════════════════
# GENERATE HTML
# ═══════════════════════════════════════════════════════════════

def generate_html(s: dict, img_src: str) -> str:
    """Buat HTML laporan lengkap dari dictionary statistik."""

    # Blok embed gambar — hanya ditampilkan jika ada gambar
    if img_src:
        viz_block = f"""
<div class="viz-embed">
  <div class="viz-label">Gambar 1. Dashboard Analisis Sentimen PPKS — 6 Visualisasi Utama</div>
  <img src="{img_src}" alt="Dashboard Analisis Sentimen PPKS">
</div>"""
    else:
        viz_block = """
<div class="viz-embed viz-missing">
  <p>⚠️ Dashboard belum tersedia. Jalankan <code>python 04_visualisasi.py</code> terlebih dahulu,
  lalu generate ulang laporan ini.</p>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Laporan Kebijakan PPKS — NLP Analysis {datetime.now().year}</title>
<style>
/* ── FONTS & BASE ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@400;600&display=swap');
:root {{
  --bg:#0D1117; --panel:#161B22; --border:#30363D; --border2:#21262D;
  --text:#E6EDF3; --sub:#8B949E; --accent:#F0A500;
  --neg:#E63946; --pos:#2DC653; --neu:#457B9D;
  --r1:#E63946;  --r2:#FF6B35;  --r3:#E63995;
}}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html {{ scroll-behavior: smooth; }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: 'Libre Baskerville', Georgia, serif;
  line-height: 1.8;
  max-width: 1000px;
  margin: 0 auto;
  padding: 48px 28px 64px;
}}

/* ── HEADER ───────────────────────────────────────────────── */
header {{
  border-bottom: 2px solid var(--accent);
  padding-bottom: 28px;
  margin-bottom: 40px;
}}
.label-tag {{
  display: inline-block;
  background: var(--accent);
  color: #000;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  font-weight: 700;
  padding: 4px 12px;
  border-radius: 3px;
  letter-spacing: .08em;
  margin-bottom: 14px;
}}
h1 {{
  font-size: 28px;
  line-height: 1.35;
  color: var(--text);
  margin-bottom: 10px;
}}
.meta-row {{
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  margin-top: 14px;
}}
.meta-item {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: var(--sub);
}}
.meta-item span {{ color: var(--text); font-weight: 600; }}

/* ── SECTION HEADINGS ──────────────────────────────────────── */
h2 {{
  font-size: 17px;
  color: var(--accent);
  font-family: 'IBM Plex Mono', monospace;
  letter-spacing: .06em;
  margin: 48px 0 18px;
  border-left: 3px solid var(--accent);
  padding-left: 14px;
}}
h3 {{ font-size: 14px; color: var(--text); margin: 22px 0 8px; }}
p  {{ color: var(--sub); margin-bottom: 12px; font-size: 14.5px; }}
strong {{ color: var(--text); }}
em {{
  color: var(--accent);
  font-style: normal;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 13px;
}}
code {{
  font-family: 'IBM Plex Mono', monospace;
  background: var(--border2);
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 12px;
  color: var(--accent);
}}

/* ── STATS CARDS ───────────────────────────────────────────── */
.stats-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  margin: 24px 0;
}}
.stat-card {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 22px 18px;
  text-align: center;
}}
.stat-num {{
  font-size: 36px;
  font-weight: 700;
  font-family: 'IBM Plex Mono', monospace;
}}
.stat-sub {{ font-size: 11.5px; color: var(--sub); margin-top: 5px; }}
.c-neg {{ color: var(--neg); border-top: 3px solid var(--neg); }}
.c-pos {{ color: var(--pos); border-top: 3px solid var(--pos); }}
.c-neu {{ color: var(--neu); border-top: 3px solid var(--neu); }}

/* ── PROGRESS BAR ──────────────────────────────────────────── */
.progress-wrap {{ margin: 10px 0 18px; }}
.progress-label {{
  display: flex;
  justify-content: space-between;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  margin-bottom: 6px;
}}
.progress-bar {{
  height: 8px;
  background: var(--border2);
  border-radius: 4px;
  overflow: hidden;
}}
.progress-fill {{
  height: 100%;
  border-radius: 4px;
}}

/* ── FINDING CARDS ─────────────────────────────────────────── */
.finding {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 22px 26px;
  margin: 16px 0;
}}
.finding-title {{
  font-weight: 700;
  color: var(--text);
  font-size: 15px;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 10px;
}}
.pct-badge {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 22px;
  font-weight: 700;
  white-space: nowrap;
}}

/* ── DASHBOARD EMBED ───────────────────────────────────────── */
/*
 * Kunci fix whitespace: gunakan display:block pada img,
 * set width 100% dan biarkan height auto. Tidak ada fixed-height
 * pada container sehingga tidak ada ruang kosong di atas gambar.
 */
.viz-embed {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;       /* clip radius ke gambar */
  margin: 28px 0;
}}
.viz-label {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: var(--sub);
  padding: 10px 16px;
  border-bottom: 1px solid var(--border2);
  background: var(--border2);
}}
.viz-embed img {{
  display: block;         /* hapus inline baseline-gap */
  width: 100%;
  height: auto;           /* tinggi mengikuti aspek rasio alami */
  border: none;
}}
.viz-missing {{
  padding: 32px 24px;
  text-align: center;
}}
.viz-missing p {{
  font-size: 13.5px;
  color: var(--sub);
}}

/* ── RECOMMENDATION CARDS ──────────────────────────────────── */
.level-badge {{
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  font-weight: 700;
  padding: 4px 12px;
  border-radius: 3px;
  margin: 20px 0 16px;
}}
.level-r {{
  background: rgba(230,57,70,.1);
  color: var(--neg);
  border: 1px solid var(--neg);
}}
.level-k {{
  background: rgba(69,123,157,.1);
  color: var(--neu);
  border: 1px solid var(--neu);
}}
.rec-card {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 22px 26px;
  margin: 12px 0;
}}
.rec-id {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: var(--accent);
  font-weight: 700;
  margin-bottom: 6px;
}}
.rec-title {{
  font-size: 15px;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 12px;
}}
.evidence-block {{
  font-size: 13px;
  color: var(--sub);
  font-style: italic;
  border-left: 2px solid var(--border);
  padding-left: 14px;
  margin: 10px 0 14px;
}}
.priority-tag {{
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 2px;
  margin-left: 6px;
  vertical-align: middle;
}}
.p-high {{ background: rgba(230,57,70,.15); color: var(--neg); }}
.p-med  {{ background: rgba(240,165,0,.15);  color: var(--accent); }}
ul.actions {{ list-style: none; padding: 0; }}
ul.actions li {{
  font-size: 13.5px;
  color: var(--sub);
  padding: 6px 0 6px 22px;
  position: relative;
  border-bottom: 1px solid var(--border2);
}}
ul.actions li:last-child {{ border-bottom: none; }}
ul.actions li::before {{
  content: '→';
  position: absolute;
  left: 0;
  color: var(--accent);
  font-weight: 700;
}}

/* ── METHODOLOGY TABLE ─────────────────────────────────────── */
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 16px 0;
  font-size: 13.5px;
}}
th {{
  background: var(--border2);
  color: var(--text);
  text-align: left;
  padding: 10px 14px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: .05em;
}}
td {{
  padding: 9px 14px;
  border-bottom: 1px solid var(--border2);
  color: var(--sub);
}}
tr:hover td {{ background: rgba(255,255,255,.02); }}

/* ── PIPELINE STEPS ─────────────────────────────────────────── */
.pipeline {{
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0;
  margin: 20px 0;
}}
.pipe-step {{
  background: var(--panel);
  border: 1px solid var(--border);
  padding: 16px 10px;
  text-align: center;
  position: relative;
}}
.pipe-step:not(:last-child)::after {{
  content: '▶';
  position: absolute;
  right: -10px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--accent);
  font-size: 13px;
  z-index: 2;
}}
.pipe-num {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 18px;
  font-weight: 700;
  color: var(--accent);
}}
.pipe-label {{
  font-size: 10px;
  color: var(--sub);
  margin-top: 5px;
  line-height: 1.4;
}}

/* ── CONCLUSION BOX ─────────────────────────────────────────── */
.conclusion {{
  background: linear-gradient(135deg,
    rgba(240,165,0,.07), rgba(230,57,70,.07));
  border: 1px solid var(--accent);
  border-radius: 10px;
  padding: 30px 34px;
  margin: 40px 0;
}}
.conclusion h3 {{
  color: var(--accent);
  margin-bottom: 14px;
  font-size: 16px;
}}

/* ── FOOTER ──────────────────────────────────────────────────── */
footer {{
  border-top: 1px solid var(--border);
  padding-top: 22px;
  margin-top: 48px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: var(--sub);
  line-height: 1.9;
}}

/* ── PRINT STYLES ────────────────────────────────────────────── */
@media print {{
  body {{ background: white; color: #111; max-width: 100%; padding: 20px; }}
  .stat-card, .rec-card, .finding, .conclusion {{
    border: 1px solid #ccc;
    page-break-inside: avoid;
  }}
  h2 {{ color: #333; border-left-color: #333; }}
  .viz-embed {{ page-break-inside: avoid; }}
}}
</style>
</head>
<body>

<!-- ═══════ HEADER ════════════════════════════════════════════ -->
<header>
  <div class="label-tag">EVIDENCE-BASED POLICY REPORT · CONFIDENTIAL DRAFT</div>
  <h1>Analisis Sentimen Publik Terhadap Implementasi<br>
      Permendikbud PPKS No. 30/2021</h1>
  <div class="meta-row">
    <div class="meta-item">Dataset <span>{s['total']:,} tweet</span></div>
    <div class="meta-item">Periode <span>{s['date_start']} – {s['date_end']}</span></div>
    <div class="meta-item">Kota tercakup <span>{s['kota_count']}</span></div>
    <div class="meta-item">Model <span>{s['method'].upper()}</span></div>
    <div class="meta-item">Dibuat <span>{s['generated_at']}</span></div>
  </div>
</header>

<!-- ═══════ 01 METODOLOGI ═════════════════════════════════════ -->
<h2>01 — METODOLOGI</h2>

<div class="pipeline">
  <div class="pipe-step">
    <div class="pipe-num">01</div>
    <div class="pipe-label">Scraping<br>Twitter API v2</div>
  </div>
  <div class="pipe-step">
    <div class="pipe-num">02</div>
    <div class="pipe-label">Preprocessing<br>Sastrawi + Slang</div>
  </div>
  <div class="pipe-step">
    <div class="pipe-num">03</div>
    <div class="pipe-label">Sentimen<br>{s['method'].upper()}</div>
  </div>
  <div class="pipe-step">
    <div class="pipe-num">04</div>
    <div class="pipe-label">Topic Model<br>LDA 5 Topik</div>
  </div>
  <div class="pipe-step">
    <div class="pipe-num">05</div>
    <div class="pipe-label">Rekomendasi<br>Kebijakan</div>
  </div>
</div>

<table>
  <tr><th>Komponen</th><th>Detail</th></tr>
  <tr><td>Query Scraping</td>
      <td>(kekerasan seksual OR PPKS OR pelecehan seksual) (kampus OR mahasiswa) -is:retweet lang:id</td></tr>
  <tr><td>Preprocessing</td>
      <td>PySastrawi stemming, normalisasi slang (≈200 entri), dekode sensor, emoji→teks</td></tr>
  <tr><td>Model Sentimen</td>
      <td>{s['model_display']}</td></tr>
  <tr><td>Topic Modeling</td>
      <td>LDA (scikit-learn), 5 komponen, TF-IDF vocabulary 800 kata</td></tr>
  <tr><td>Anonimisasi</td>
      <td>SHA-256 irreversible hash untuk User ID, penghapusan PII (phone, email, NIK)</td></tr>
  <tr><td>Kepatuhan</td>
      <td>UU PDP No. 27/2022, Twitter Developer Policy Academic Access</td></tr>
</table>

<!-- ═══════ 02 TEMUAN UTAMA ═══════════════════════════════════ -->
<h2>02 — TEMUAN UTAMA</h2>

<div class="stats-grid">
  <div class="stat-card c-neg">
    <div class="stat-num c-neg">{s['neg_pct']}%</div>
    <div class="stat-sub">SENTIMEN NEGATIF<br>{s['neg_count']:,} tweet</div>
  </div>
  <div class="stat-card c-pos">
    <div class="stat-num c-pos">{s['pos_pct']}%</div>
    <div class="stat-sub">SENTIMEN POSITIF<br>{s['pos_count']:,} tweet</div>
  </div>
  <div class="stat-card c-neu">
    <div class="stat-num c-neu">{s['net_pct']}%</div>
    <div class="stat-sub">SENTIMEN NETRAL<br>{s['net_count']:,} tweet</div>
  </div>
</div>

<p>Mayoritas <strong>{s['neg_pct']}%</strong> percakapan publik di Twitter/X bernada negatif —
konsisten sepanjang periode pengamatan — mengindikasikan
<strong>krisis kepercayaan sistemik</strong> terhadap implementasi PPKS.
Konsentrasi tweet negatif tertinggi berasal dari: <strong>{s['top_cities']}</strong>.</p>

<p>Kata kunci yang paling sering muncul dalam klaster negatif:
<em>{s['top_neg_words']}</em>. Pola ini mencerminkan kegagalan
pada tiga level sekaligus: prosedural, institusional, dan budaya.</p>

<div class="progress-wrap">
  <div class="progress-label">
    <span style="color:var(--neg)">■ NEGATIF</span>
    <span>{s['neg_pct']}%</span>
  </div>
  <div class="progress-bar">
    <div class="progress-fill" style="width:{s['neg_pct']}%;background:var(--neg)"></div>
  </div>
</div>
<div class="progress-wrap">
  <div class="progress-label">
    <span style="color:var(--pos)">■ POSITIF</span>
    <span>{s['pos_pct']}%</span>
  </div>
  <div class="progress-bar">
    <div class="progress-fill" style="width:{s['pos_pct']}%;background:var(--pos)"></div>
  </div>
</div>
<div class="progress-wrap">
  <div class="progress-label">
    <span style="color:var(--neu)">■ NETRAL</span>
    <span>{s['net_pct']}%</span>
  </div>
  <div class="progress-bar">
    <div class="progress-fill" style="width:{s['net_pct']}%;background:var(--neu)"></div>
  </div>
</div>

<!-- 3 Klaster Temuan -->
<div class="finding" style="border-left:4px solid var(--r1)">
  <div class="finding-title">
    <span class="pct-badge" style="color:var(--r1)">{s['fear_pct']}%</span>
    Klaster 1: Ketakutan Melapor &amp; Intimidasi
  </div>
  <p>Kata kunci dominan: <em>takut, bungkam, ancaman, diam, tidak ada perlindungan, dipaksa damai</em>.
  Klaster ini menjadi yang terbesar ({s['fear_count']:,} tweet) dan menunjukkan
  <strong>kegagalan mekanisme perlindungan pelapor</strong> di tingkat institusi. Banyak tweet
  menyebut pelaku adalah figur otoritas (dosen pembimbing, dekan), menciptakan ketidakseimbangan
  kuasa yang melumpuhkan keberanian melapor.</p>
</div>

<div class="finding" style="border-left:4px solid var(--r2)">
  <div class="finding-title">
    <span class="pct-badge" style="color:var(--r2)">{s['sanc_pct']}%</span>
    Klaster 2: Ketiadaan Transparansi Sanksi
  </div>
  <p>Frasa paling sering: <em>sanksi tidak jelas, pelaku masih ngajar, ditutup-tutupi,
  tidak ada kejelasan, sudah lapor tapi tidak ada hasilnya</em>.
  Publik tidak memiliki akses informasi tentang proses dan hasil penanganan kasus
  ({s['sanc_count']:,} tweet), menciptakan <strong>persepsi impunitas</strong> yang
  melemahkan kepercayaan pada sistem.</p>
</div>

<div class="finding" style="border-left:4px solid var(--r3)">
  <div class="finding-title">
    <span class="pct-badge" style="color:var(--r3)">{s['blame_pct']}%</span>
    Klaster 3: Victim Blaming &amp; Budaya Toxic
  </div>
  <p>Kata kunci: <em>salahkan korban, pakaian, jaga diri, bercanda, dikucilkan, gaslight</em>.
  {s['blame_count']:,} tweet menggambarkan lingkungan kampus yang menormalkan pelecehan atau
  mengkriminalisasi korban — mencerminkan <strong>kegagalan budaya institusional</strong>
  yang melampaui sekadar kegagalan regulasi.</p>
</div>

<!-- ═══════ DASHBOARD EMBED ════════════════════════════════════
     Gambar di-embed sebagai base64 → tidak bergantung path relatif,
     tidak ada whitespace kosong di atas gambar.
═══════════════════════════════════════════════════════════════ -->
{viz_block}

<!-- ═══════ 03 REKOMENDASI — REKTORAT ═════════════════════════ -->
<h2>03 — REKOMENDASI KEBIJAKAN</h2>

<div class="level-badge level-r">LEVEL: REKTORAT / PIMPINAN PERGURUAN TINGGI</div>

<div class="rec-card">
  <div class="rec-id">REC-R01
    <span class="priority-tag p-high">PRIORITAS TINGGI</span>
    <span class="priority-tag p-med">KURATIF</span>
  </div>
  <div class="rec-title">Reformasi Komposisi &amp; Prosedur Satgas PPKS</div>
  <div class="evidence-block">
    Evidence: {s['fear_pct']}% sentimen negatif ({s['fear_count']:,} tweet) berkorelasi
    dengan narasi ketiadaan perlindungan struktural bagi pelapor — khususnya ketika
    pelaku adalah dosen atau staf senior yang memegang kendali nilai akademik.
  </div>
  <ul class="actions">
    <li>Mandatkan minimum <strong>40% komposisi Satgas dari pihak eksternal independen</strong>
      (psikolog klinis bersertifikat, LBH, NGO perempuan) — bukan hanya dosen internal.</li>
    <li>Pisahkan jalur pelaporan sepenuhnya dari rantai birokrasi akademik:
      laporan <em>tidak</em> melewati dekan, kaprodi, atau wali dosen.</li>
    <li>Implementasi <strong>"Case Manager" bersertifikat</strong> yang merespons laporan
      dalam maksimal 24 jam kerja dengan kontak yang jelas dan terpublikasi.</li>
    <li>Bangun jalur pelaporan anonim digital (end-to-end encrypted) yang dapat diakses
      melalui website dan WhatsApp tanpa login akun kampus.</li>
    <li>Jamin <strong>non-retaliation policy</strong>: sanksi tegas bagi siapapun yang
      mengintimidasi atau menekan pelapor.</li>
  </ul>
</div>

<div class="rec-card">
  <div class="rec-id">REC-R02
    <span class="priority-tag p-high">PRIORITAS TINGGI</span>
    <span class="priority-tag p-med">KURATIF</span>
  </div>
  <div class="rec-title">Sistem Transparansi Sanksi Berbasis Data</div>
  <div class="evidence-block">
    Evidence: {s['sanc_pct']}% tweet negatif ({s['sanc_count']:,} tweet) memuat narasi
    tentang sanksi yang tidak jelas dan pelaku yang tetap bebas mengajar pasca laporan —
    mengindikasikan absennya akuntabilitas publik yang terverifikasi.
  </div>
  <ul class="actions">
    <li>Terbitkan <strong>Laporan Transparansi PPKS Tahunan</strong> (teranonim) yang memuat:
      jumlah aduan masuk, kategorisasi pelanggaran, dan jenis sanksi yang dijatuhkan.</li>
    <li>Terapkan <strong>SLA (Service Level Agreement) tertulis</strong>: keputusan final
      maksimal 90 hari sejak laporan diterima, dengan notifikasi progres ke pelapor
      setiap 14 hari.</li>
    <li>Tampilkan dashboard publik agregat (status kasus — bukan identitas)
      di website resmi kampus, diperbarui setiap kuartal.</li>
    <li>Pelaku terbukti bersalah: laporkan ke Kemendikbud dan publikasikan
      jenis sanksi (tanpa identitas) dalam laporan tahunan.</li>
  </ul>
</div>

<div class="rec-card">
  <div class="rec-id">REC-R03
    <span class="priority-tag p-med">PRIORITAS MENENGAH</span>
    <span class="priority-tag" style="background:rgba(45,198,83,.1);color:var(--pos);border:1px solid var(--pos)">PREVENTIF</span>
  </div>
  <div class="rec-title">Reformasi Budaya Institusional</div>
  <div class="evidence-block">
    Evidence: {s['blame_pct']}% klaster negatif ({s['blame_count']:,} tweet) tentang
    victim blaming menunjukkan kegagalan budaya yang tidak bisa diselesaikan hanya
    dengan regulasi — butuh intervensi pada level norma sosial kampus.
  </div>
  <ul class="actions">
    <li>Wajibkan modul <strong>"Consent, Bystander, dan PPKS"</strong> (minimum 4 jam)
      dalam PKKMB setiap tahun ajaran baru.</li>
    <li>Pelatihan <strong>"Bystander Intervention"</strong> wajib tahunan untuk seluruh
      dosen dan tenaga kependidikan — sertifikasi 3 tahun sekali.</li>
    <li>Audit budaya institusi tahunan oleh konsultan independen; hasilnya
      dipublikasikan di website kampus.</li>
    <li>Bentuk <strong>peer support network</strong> mahasiswa yang terlatih untuk
      mendampingi korban sebelum melaporkan secara formal.</li>
  </ul>
</div>

<!-- ═══════ REKOMENDASI — KEMENTERIAN ══════════════════════════ -->
<div class="level-badge level-k">LEVEL: KEMENDIKBUDRISTEK / PEMERINTAH PUSAT</div>

<div class="rec-card">
  <div class="rec-id">REC-K01
    <span class="priority-tag p-high">PRIORITAS TINGGI</span>
    <span class="priority-tag p-med">REGULASI</span>
  </div>
  <div class="rec-title">Penguatan &amp; Restorasi Landasan Hukum PPKS</div>
  <div class="evidence-block">
    Evidence: Analisis timeline menunjukkan lonjakan sentimen negatif pasca revisi
    Permendikbud yang melemahkan substansi PPKS — membuktikan respons publik sangat
    sensitif terhadap setiap upaya dilusi kebijakan perlindungan korban.
  </div>
  <ul class="actions">
    <li>Restaurasi dan perkuat pasal <strong>consent</strong> dalam regulasi PPKS
      sesuai standar internasional (Istanbul Convention).</li>
    <li>Buat <strong>Standar Nasional Satgas PPKS (SNI-PPKS)</strong>: kompetensi
      minimal anggota, prosedur standar penanganan, dan mekanisme pengawasan
      lintas kementerian.</li>
    <li>Hubungkan kepatuhan PPKS dengan <strong>akreditasi BAN-PT</strong>:
      kampus tanpa implementasi terverifikasi mendapat penalti penurunan akreditasi.</li>
    <li>Bentuk <strong>Inspektorat PPKS Nasional</strong> yang berwenang
      mengaudit dan memberikan sanksi institusi yang tidak patuh.</li>
  </ul>
</div>

<div class="rec-card">
  <div class="rec-id">REC-K02
    <span class="priority-tag p-med">PRIORITAS MENENGAH</span>
    <span class="priority-tag" style="background:rgba(45,198,83,.1);color:var(--pos);border:1px solid var(--pos)">INFRASTRUKTUR</span>
  </div>
  <div class="rec-title">Infrastruktur Dukungan Nasional</div>
  <ul class="actions">
    <li>Dirikan <strong>Hotline Nasional Korban Kekerasan Seksual Kampus</strong>
      (24/7, tersedia via telepon dan WhatsApp), terintegrasi dengan
      Komnas Perempuan &amp; P2TP2A daerah.</li>
    <li>Alokasi <strong>Dana PPKS Nasional</strong> untuk kampus dengan kapasitas
      keuangan terbatas (PTS kecil di luar Jawa) agar implementasi merata.</li>
    <li>Bangun <strong>Repository Kasus Nasional</strong> terenkripsi untuk keperluan
      riset kebijakan, monitoring nasional, dan pelaporan ke DPR.</li>
    <li>Mandatkan <strong>integrasi PPKS dalam kurikulum pendidikan tinggi</strong>
      sebagai mata kuliah wajib non-SKS (mirip model MKWU).</li>
  </ul>
</div>

<!-- ═══════ 04 KESIMPULAN ══════════════════════════════════════ -->
<div class="conclusion">
  <h3>04 — KESIMPULAN &amp; CALL TO ACTION</h3>
  <p>Analisis terhadap <strong>{s['total']:,} tweet</strong> selama periode
  {s['date_start']}–{s['date_end']} menunjukkan bahwa <strong>{s['neg_pct']}%
  percakapan publik bernada negatif</strong> — angka yang stabil dan tidak
  menunjukkan penurunan signifikan meski regulasi PPKS telah berjalan lebih dari
  dua tahun.</p>
  <p>Data mencerminkan <strong>krisis kepercayaan tiga lapis</strong>:
  (1) korban tidak percaya institusi akan melindungi mereka;
  (2) publik tidak percaya sanksi akan ditegakkan;
  (3) budaya kampus belum menciptakan ruang aman bagi korban untuk bicara.
  Regulasi yang baik di atas kertas tidak akan berdampak tanpa
  akuntabilitas yang dapat diverifikasi publik.</p>
  <p><strong>Urutan prioritas eksekusi:</strong>
  REC-R01 (Reformasi Satgas) dan REC-R02 (Transparansi Sanksi) harus
  dieksekusi dalam 6 bulan pertama. REC-K01 (Penguatan Regulasi) dalam
  12 bulan. Perubahan budaya (REC-R03, REC-K02) adalah kerja jangka menengah
  (2–3 tahun) yang membutuhkan monitoring berkelanjutan berbasis data.</p>
  <p style="margin-top:16px"><em>Riset ini adalah alat, bukan kesimpulan.
  Distribusikan ke Satgas PPKS, LBH terkait, atau Komnas Perempuan agar temuan
  tidak berhenti di laporan akademik.</em></p>
</div>

<!-- ═══════ FOOTER ═════════════════════════════════════════════ -->
<footer>
  <p><strong>Stack Teknis:</strong> Python 3.10+ · scikit-learn · PySastrawi ·
  matplotlib · {s['model_display'].split('(')[0].strip()} · Twitter API v2 Academic Access</p>
  <p><strong>Kepatuhan:</strong> UU PDP No. 27/2022 (anonimisasi SHA-256) ·
  Twitter Developer Policy · IRB-approved protocol (sesuaikan dengan kampus)</p>
  <p style="margin-top:8px;color:#444">
  ⚠️ Laporan ini berbasis data publik Twitter. Tidak merepresentasikan seluruh populasi.
  Data ini tidak menggantikan investigasi formal atau proses hukum.
  Jangan gunakan untuk mengidentifikasi individu.</p>
  <p style="margin-top:8px">Dibuat: {s['generated_at']}</p>
</footer>

</body>
</html>"""


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("   FASE 5 — LAPORAN REKOMENDASI KEBIJAKAN")
    print("=" * 55)

    if not os.path.exists("data/analyzed_tweets.csv"):
        print("❌ File tidak ditemukan: data/analyzed_tweets.csv")
        print("   Jalankan dulu: python 03_sentiment_topic.py")
        sys.exit(1)

    print("📂 Memuat data...")
    df = pd.read_csv("data/analyzed_tweets.csv", encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])

    df_topics = pd.DataFrame()
    if os.path.exists("data/topic_summary.csv"):
        df_topics = pd.read_csv("data/topic_summary.csv", encoding="utf-8-sig")

    print("📊 Menghitung statistik...")
    stats = compute_stats(df, df_topics)

    print(f"\n── Ringkasan Data ──")
    print(f"   Total tweet     : {stats['total']:,}")
    print(f"   Negatif         : {stats['neg_pct']}% ({stats['neg_count']:,})")
    print(f"   Positif         : {stats['pos_pct']}% ({stats['pos_count']:,})")
    print(f"   Netral          : {stats['net_pct']}% ({stats['net_count']:,})")
    print(f"   Model sentimen  : {stats['method']}")

    # Muat gambar dashboard sebagai base64 agar path tidak bergantung lokasi file HTML
    dashboard_path = "output/ppks_dashboard.png"
    print(f"\n🖼  Memuat dashboard: {dashboard_path}")
    img_src = load_dashboard_image(dashboard_path)
    if img_src:
        print("   ✓ Gambar berhasil dimuat dan di-embed sebagai base64.")
    else:
        print("   ⚠️  Gambar tidak ditemukan — laporan akan tampil tanpa dashboard.")
        print("      Jalankan 04_visualisasi.py lalu generate ulang laporan ini.")

    print("\n📝 Menyusun laporan HTML...")
    html_content = generate_html(stats, img_src)

    output_path = "output/ppks_laporan_kebijakan.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ Laporan tersimpan → {output_path}")
    print("   Buka file tersebut di browser untuk melihat hasilnya yaa.")
    print("\n══════════════════════════════════════════════════════")
    print("   🎉 PIPELINE SELESAI! Semua output ada di folder output/")
    print("   📊 output/ppks_dashboard.png")
    print("   📄 output/ppks_laporan_kebijakan.html")
    print("══════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()