"""
utils/visualize_stats.py
Reads emotion_log.csv and plots a bar chart of emotion frequency.
Run after a session: python utils/visualize_stats.py
"""

import csv
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict


EMOTION_COLORS = {
    "happy":    "#00FF96",
    "sad":      "#5B8CFF",
    "angry":    "#FF4444",
    "fear":     "#CC44CC",
    "surprise": "#FFD700",
    "disgust":  "#44AA44",
    "neutral":  "#AAAAAA",
}


def load_log(log_path: str = "output/emotion_log.csv") -> dict:
    counts = defaultdict(int)
    if not os.path.exists(log_path):
        print(f"[ERROR] Log not found: {log_path}")
        return {}
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row["dominant_emotion"]] += 1
    return dict(counts)


def plot_stats(counts: dict, save_path: str = "output/emotion_stats.png"):
    if not counts:
        print("[INFO] No data to plot.")
        return

    emotions = list(counts.keys())
    values = list(counts.values())
    colors = [EMOTION_COLORS.get(e, "#888888") for e in emotions]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0D1117")
    ax.set_facecolor("#161B22")

    bars = ax.bar(emotions, values, color=colors, width=0.55, edgecolor="#30363D", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=10,
                color="white", fontweight="bold")

    ax.set_title("Emotion Detection — Session Summary", color="white",
                 fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("Emotion", color="#8B949E", fontsize=11)
    ax.set_ylabel("Frequency", color="#8B949E", fontsize=11)
    ax.tick_params(colors="white", labelsize=10)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"[SAVED] Stats chart → {save_path}")
    plt.show()


if __name__ == "__main__":
    counts = load_log()
    print("Session emotion counts:", counts)
    plot_stats(counts)
