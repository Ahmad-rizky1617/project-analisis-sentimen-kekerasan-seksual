"""
FILE: 04_visualisasi.py
TUJUAN: Menghasilkan 6 visualisasi analisis sentimen PPKS dalam satu dashboard.

JALANKAN: python 04_visualisasi.py
INPUT   : data/analyzed_tweets.csv, data/topic_summary.csv
OUTPUT  : output/ppks_dashboard.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

os.makedirs("output", exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# TEMA WARNA — Dark analytical theme
# ═══════════════════════════════════════════════════════════════
CLR = {
    "NEGATIF": "#E63946",
    "POSITIF": "#2DC653",
    "NETRAL":  "#457B9D",
    "bg":      "#0D1117",
    "panel":   "#161B22",
    "text":    "#E6EDF3",
    "subtext": "#8B949E",
    "accent":  "#F0A500",
    "grid":    "#21262D",
    "r1":      "#E63946",
    "r2":      "#FF6B35",
    "r3":      "#E63995",
}

POLICY_EVENTS = [
    ("2021-11-01", "Permendikbud\nPPKS disahkan", "#F0A500"),
    ("2022-04-01", "Revisi\nkontroversial",        "#FF6B6B"),
    ("2023-03-01", "Kasus viral\n#DosenPredator",  "#F0A500"),
    ("2023-09-01", "Evaluasi\nKemendikbud",        "#8B949E"),
]


def setup_matplotlib():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "text.color":        CLR["text"],
        "axes.facecolor":    CLR["panel"],
        "figure.facecolor":  CLR["bg"],
        "axes.edgecolor":    CLR["grid"],
        "axes.labelcolor":   CLR["text"],
        "xtick.color":       CLR["subtext"],
        "ytick.color":       CLR["subtext"],
        "grid.color":        CLR["grid"],
        "grid.linewidth":    0.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ═══════════════════════════════════════════════════════════════
# 6 FUNGSI CHART
# ═══════════════════════════════════════════════════════════════

def chart_donut_sentimen(ax, df):
    """VIZ 1: Donut chart distribusi sentimen keseluruhan."""
    sent_counts = df["sentiment"].value_counts()
    colors = [CLR.get(s, "#888") for s in sent_counts.index]

    wedges, texts, autotexts = ax.pie(
        sent_counts.values,
        labels=sent_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=CLR["bg"], linewidth=2.5),
        pctdistance=0.78,
        textprops={"color": CLR["text"], "fontsize": 9, "fontweight": "bold"},
    )
    for at in autotexts:
        at.set_fontsize(8)

    total = sent_counts.sum()
    ax.text(0, 0.10, str(total), ha="center", va="center",
            fontsize=16, fontweight="bold", color=CLR["text"])
    ax.text(0, -0.15, "tweet", ha="center", va="center",
            fontsize=9, color=CLR["subtext"])

    ax.set_title("Distribusi Sentimen Publik", color=CLR["text"],
                 fontsize=11, fontweight="bold", pad=10)


def chart_timeline(ax, df):
    """VIZ 2: Line chart sentimen per bulan dengan anotasi event kebijakan."""
    df["month_dt"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby(["month_dt", "sentiment"]).size().unstack(fill_value=0)

    for sent, color in [("NEGATIF", CLR["NEGATIF"]),
                         ("POSITIF", CLR["POSITIF"]),
                         ("NETRAL",  CLR["NETRAL"])]:
        if sent not in monthly.columns:
            continue
        vals = monthly[sent].rolling(3, min_periods=1).mean()
        ax.plot(monthly.index, vals, color=color, linewidth=2.0,
                label=sent, alpha=0.9, zorder=3)
        ax.fill_between(monthly.index, vals, alpha=0.10, color=color, zorder=2)

    ax.autoscale(axis="y")
    y_max = ax.get_ylim()[1]

    for date_str, label, color in POLICY_EVENTS:
        xdate = pd.Timestamp(date_str)
        if not monthly.empty and monthly.index.min() <= xdate <= monthly.index.max():
            ax.axvline(x=xdate, color=color, linestyle="--",
                       alpha=0.65, linewidth=1.2, zorder=1)
            ax.text(xdate, y_max * 0.90, label,
                    color=color, fontsize=6.5, ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=CLR["panel"],
                              edgecolor=color, alpha=0.85))

    ax.set_title("Timeline Sentimen (3-Bulan Moving Average)", color=CLR["text"],
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Periode", fontsize=8)
    ax.set_ylabel("Jumlah Tweet", fontsize=8)
    ax.legend(loc="upper left", facecolor=CLR["panel"],
              edgecolor=CLR["grid"], labelcolor=CLR["text"], fontsize=8)
    ax.grid(True, alpha=0.3)


def chart_topic_bar(ax, df):
    """VIZ 3: Horizontal bar chart distribusi topik."""
    topic_counts = df["topic_name"].value_counts().sort_values()

    def get_color(name):
        if any(k in name for k in ["Ketakutan", "Victim", "Transparansi"]):
            return CLR["NEGATIF"]
        if "Solidaritas" in name or "Dukungan" in name:
            return CLR["POSITIF"]
        return CLR["NETRAL"]

    bar_colors = [get_color(n) for n in topic_counts.index]
    bars = ax.barh(topic_counts.index, topic_counts.values,
                   color=bar_colors, height=0.50,
                   edgecolor=CLR["bg"], linewidth=1)

    for bar, val in zip(bars, topic_counts.values):
        ax.text(val + 1.0, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", color=CLR["text"],
                fontsize=8, fontweight="bold")

    ax.set_xlim(0, topic_counts.max() * 1.18)
    ax.set_title("Distribusi 5 Topik Utama (LDA)", color=CLR["text"],
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Jumlah Tweet", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)


def chart_topic_keywords(ax, df_topic_summary):
    """VIZ 4: Daftar kata kunci per topik."""
    ax.set_facecolor(CLR["panel"])
    ax.set_title("Kata Kunci Dominan per Topik", color=CLR["text"],
                 fontsize=11, fontweight="bold")
    ax.axis("off")

    palette = [CLR["r1"], CLR["r2"], CLR["r3"], CLR["POSITIF"], CLR["NETRAL"]]
    y = 0.95

    for i, row in df_topic_summary.iterrows():
        if y < 0.05:
            break
        color = palette[i % len(palette)]
        name  = row["topic_name"]
        words = str(row["top_words"]).split(", ")[:6]

        ax.text(0.02, y, f"● {name}", transform=ax.transAxes,
                fontsize=8, fontweight="bold", color=color, va="top")
        ax.text(0.06, y - 0.05, ", ".join(words),
                transform=ax.transAxes, fontsize=7,
                color=CLR["subtext"], va="top")
        y -= 0.20


def chart_kota(ax, df):
    """VIZ 5: Bar chart grouped distribusi sentimen per kota."""
    if "city" not in df.columns:
        ax.text(0.5, 0.5, "Kolom 'city' tidak tersedia",
                ha="center", va="center", transform=ax.transAxes,
                color=CLR["subtext"], fontsize=10)
        ax.set_title("Distribusi Sentimen per Kota", color=CLR["text"],
                     fontsize=11, fontweight="bold")
        return

    city_sent = (df.groupby(["city", "sentiment"])
                   .size().unstack(fill_value=0)
                   .sort_values("NEGATIF", ascending=True)
                   .tail(8))
    x = np.arange(len(city_sent))
    w = 0.26

    sentiments = [s for s in ["NEGATIF", "NETRAL", "POSITIF"]
                  if s in city_sent.columns]

    for i, (sent, color) in enumerate(
            zip(sentiments, [CLR[s] for s in sentiments])):
        ax.bar(x + i * w, city_sent[sent], width=w, label=sent,
               color=color, alpha=0.85, edgecolor=CLR["bg"], linewidth=1.0)

    ax.set_xticks(x + w)
    ax.set_xticklabels(city_sent.index, rotation=30, ha="right", fontsize=8)
    ax.set_title("Distribusi Sentimen per Kota", color=CLR["text"],
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Jumlah Tweet", fontsize=8)
    ax.legend(facecolor=CLR["panel"], edgecolor=CLR["grid"],
              labelcolor=CLR["text"], fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def chart_negatif_breakdown(ax, df):
    """VIZ 6: Pie chart breakdown klaster sentimen negatif."""
    neg_df = df[df["sentiment"] == "NEGATIF"]

    if "category" in df.columns:
        neg_cats = neg_df["category"].value_counts()
        label_map = {
            "negative_fear":     "Ketakutan\nMelapor",
            "negative_sanction": "Ketiadaan\nSanksi",
            "negative_blaming":  "Victim\nBlaming",
        }
        display_labels = [label_map.get(k, k) for k in neg_cats.index[:3]]
        sizes = neg_cats.values[:3]
    else:
        neg_topics = neg_df["topic_name"].value_counts().head(3)
        display_labels = [n.split("&")[0].strip() for n in neg_topics.index]
        sizes = neg_topics.values

    if len(sizes) == 0:
        ax.text(0.5, 0.5, "Tidak ada data negatif",
                ha="center", va="center", transform=ax.transAxes,
                color=CLR["subtext"], fontsize=10)
        ax.set_title("Klaster Sentimen Negatif", color=CLR["text"],
                     fontsize=11, fontweight="bold")
        return

    pie_colors = [CLR["r1"], CLR["r2"], CLR["r3"]]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=display_labels, autopct="%1.0f%%",
        colors=pie_colors[:len(sizes)], startangle=90,
        explode=[0.04] * len(sizes),
        wedgeprops=dict(edgecolor=CLR["bg"], linewidth=2),
        pctdistance=0.72,
        textprops={"color": CLR["text"], "fontsize": 8},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(8)

    ax.set_title("Klaster Sentimen Negatif", color=CLR["text"],
                 fontsize=11, fontweight="bold")


# ═══════════════════════════════════════════════════════════════
# HEADER BANNER — digambar di figure, bukan suptitle
# ═══════════════════════════════════════════════════════════════

def draw_header(fig, df):
    """
    Gambar header banner di bagian atas figure menggunakan figure.text()
    sehingga tidak tumpang tindih dengan subplot manapun.
    """
    total   = len(df)
    neg_pct = (df["sentiment"] == "NEGATIF").mean() * 100
    d_start = df["date"].min().strftime("%b %Y")
    d_end   = df["date"].max().strftime("%b %Y")

    # Judul utama — dua baris, posisi y=0.992 dengan clip_on=False
    fig.text(0.5, 0.992,
             "ANALISIS SENTIMEN PUBLIK — KEBIJAKAN PPKS",
             ha="center", va="top", fontsize=17, fontweight="bold",
             color=CLR["text"], fontfamily="DejaVu Sans")
    fig.text(0.5, 0.974,
             "Kekerasan Seksual di Perguruan Tinggi Indonesia",
             ha="center", va="top", fontsize=11,
             color=CLR["subtext"], fontfamily="DejaVu Sans")

    # Metadata ringkas
    meta = (f"Dataset: {total:,} tweet  ·  Sentimen Negatif: {neg_pct:.1f}%  ·  "
            f"Periode: {d_start} – {d_end}")
    fig.text(0.5, 0.957,
             meta, ha="center", va="top", fontsize=9,
             color=CLR["accent"], fontfamily="DejaVu Sans")

    # Garis pemisah tipis di bawah header
    line = plt.Line2D([0.04, 0.96], [0.950, 0.950],
                      transform=fig.transFigure,
                      color=CLR["grid"], linewidth=1.0)
    fig.add_artist(line)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("   FASE 4 — VISUALISASI")
    print("=" * 55)

    for path in ["data/analyzed_tweets.csv", "data/topic_summary.csv"]:
        if not os.path.exists(path):
            print(f"❌ File tidak ditemukan: {path}")
            print("   Jalankan dulu: python 03_sentiment_topic.py")
            sys.exit(1)

    df       = pd.read_csv("data/analyzed_tweets.csv", encoding="utf-8-sig")
    df_topics = pd.read_csv("data/topic_summary.csv",  encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    print(f"📂 Data dimuat: {len(df):,} tweet")

    setup_matplotlib()

    # ── Figure & GridSpec ──────────────────────────────────────
    # top=0.945 memberi ruang cukup untuk header 3 baris di atas
    fig = plt.figure(figsize=(20, 22), facecolor=CLR["bg"])

    gs = GridSpec(
        3, 3, figure=fig,
        hspace=0.46, wspace=0.36,
        top=0.942,   # ← ruang header
        bottom=0.04,
        left=0.06,
        right=0.97,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[2, :2])
    ax6 = fig.add_subplot(gs[2, 2])

    # ── Render chart ──────────────────────────────────────────
    print("🎨 Merender 6 visualisasi...")
    chart_donut_sentimen(ax1, df);    print("   [1/6] Donut sentimen ✓")
    chart_timeline(ax2, df);          print("   [2/6] Timeline sentimen ✓")
    chart_topic_bar(ax3, df);         print("   [3/6] Topic bar chart ✓")
    chart_topic_keywords(ax4, df_topics); print("   [4/6] Kata kunci topik ✓")
    chart_kota(ax5, df);              print("   [5/6] Distribusi per kota ✓")
    chart_negatif_breakdown(ax6, df); print("   [6/6] Breakdown negatif ✓")

    # ── Header banner — setelah subplot terbentuk ─────────────
    draw_header(fig, df)

    # ── Simpan ─────────────────────────────────────────────────
    output_path = "output/ppks_dashboard.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=CLR["bg"], format="png")
    plt.close()

    print(f"\n✅ Dashboard tersimpan → {output_path}")
    print("   Resolusi: ~3000×3300px (150 dpi)")
    print("\n▶ Langkah berikutnya: python 05_policy_report.py")


if __name__ == "__main__":
    main()