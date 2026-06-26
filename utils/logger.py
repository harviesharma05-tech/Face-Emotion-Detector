"""
utils/logger.py
Logs detected emotions with timestamps to a CSV file.
Useful for analysis and demo purposes.
"""

import csv
import os
import time
from datetime import datetime


class EmotionLogger:
    def __init__(self, log_path: str = "output/emotion_log.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "datetime", "dominant_emotion",
                    "happy", "sad", "angry", "fear",
                    "surprise", "disgust", "neutral"
                ])

    def log(self, dominant: str, emotions: dict):
        ts = time.time()
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        row = [
            ts, dt, dominant,
            round(emotions.get("happy", 0), 2),
            round(emotions.get("sad", 0), 2),
            round(emotions.get("angry", 0), 2),
            round(emotions.get("fear", 0), 2),
            round(emotions.get("surprise", 0), 2),
            round(emotions.get("disgust", 0), 2),
            round(emotions.get("neutral", 0), 2),
        ]
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def get_summary(self) -> dict:
        counts = {}
        try:
            with open(self.log_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    e = row.get("dominant_emotion", "")
                    counts[e] = counts.get(e, 0) + 1
        except FileNotFoundError:
            pass
        return counts
