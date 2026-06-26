import cv2
import numpy as np
from deepface import DeepFace
import time
import argparse
import os

# ── Emotion color map (BGR) ──────────────────────────────────────
EMOTION_COLORS = {
    "happy":     (0, 255, 150),
    "sad":       (255, 100, 50),
    "angry":     (0, 0, 255),
    "fear":      (200, 0, 200),
    "surprise":  (0, 200, 255),
    "disgust":   (0, 140, 0),
    "neutral":   (180, 180, 180),
}

EMOTION_EMOJI = {
    "happy": ":)",
    "sad": ":(",
    "angry": ">:(",
    "fear": "D:",
    "surprise": ":O",
    "disgust": ":S",
    "neutral": ":|",
}


def draw_emotion_bar(frame, emotions: dict, x: int, y: int, w: int):
    """Draw a mini bar chart of all emotion probabilities."""
    bar_x = x + w + 15
    if bar_x + 160 > frame.shape[1]:
        bar_x = x - 175
    bar_y = y

    cv2.rectangle(frame, (bar_x - 5, bar_y - 5),
                  (bar_x + 165, bar_y + len(emotions) * 22 + 5),
                  (30, 30, 30), -1)

    for i, (emotion, score) in enumerate(sorted(emotions.items(),
                                                  key=lambda x: x[1], reverse=True)):
        color = EMOTION_COLORS.get(emotion, (200, 200, 200))
        bar_len = int(score * 1.3)
        ey = bar_y + i * 22
        cv2.rectangle(frame, (bar_x, ey), (bar_x + bar_len, ey + 14), color, -1)
        cv2.putText(frame, f"{emotion[:4]} {score:.0f}%",
                    (bar_x + 2, ey + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)


def draw_face_box(frame, x, y, w, h, emotion: str, confidence: float):
    """Draw stylized bounding box with emotion label."""
    color = EMOTION_COLORS.get(emotion, (200, 200, 200))

    # Corner bracket style box
    corner = 18
    thickness = 2
    for (cx, cy, dx, dy) in [
        (x, y, 1, 1), (x+w, y, -1, 1),
        (x, y+h, 1, -1), (x+w, y+h, -1, -1)
    ]:
        cv2.line(frame, (cx, cy), (cx + dx*corner, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy*corner), color, thickness)

    # Label background
    label = f"{EMOTION_EMOJI.get(emotion, '')} {emotion.upper()}  {confidence:.0f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def process_frame(frame, analyze_every: int = 5, frame_count: list = [0],
                  last_result: list = [None]):
    """Analyze emotion every N frames, overlay results every frame."""
    frame_count[0] += 1

    if frame_count[0] % analyze_every == 0:
        try:
            results = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )
            last_result[0] = results if isinstance(results, list) else [results]
        except Exception:
            last_result[0] = None

    if last_result[0]:
        for face in last_result[0]:
            region = face.get("region", {})
            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 0)
            h = region.get("h", 0)

            dominant = face.get("dominant_emotion", "neutral")
            emotions = face.get("emotion", {})
            confidence = emotions.get(dominant, 0)

            if w > 0 and h > 0:
                draw_face_box(frame, x, y, w, h, dominant, confidence)
                draw_emotion_bar(frame, emotions, x, y, w)

    return frame


def run_webcam(analyze_every: int = 5, save_output: bool = False):
    """Main webcam loop."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Try --source with a video file path.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    writer = None
    if save_output:
        os.makedirs("output", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output/emotion_output.mp4", fourcc, 20,
                                  (int(cap.get(3)), int(cap.get(4))))

    fps_time = time.time()
    fps = 0
    print("[INFO] Starting emotion detection... Press 'q' to quit, 's' to save screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, analyze_every)

        # FPS counter
        fps = 1 / (time.time() - fps_time + 1e-6)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        cv2.putText(frame, "Press Q to quit | S to screenshot", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        if writer:
            writer.write(frame)

        cv2.imshow("Face Emotion Detector — harviesharma", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            os.makedirs("output", exist_ok=True)
            fname = f"output/screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[SAVED] {fname}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


def run_image(image_path: str):
    """Run emotion detection on a single image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return

    frame = process_frame(frame, analyze_every=1)
    out_path = f"output/result_{os.path.basename(image_path)}"
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(out_path, frame)
    print(f"[SAVED] Result saved to {out_path}")
    cv2.imshow("Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Emotion Detector")
    parser.add_argument("--source", type=str, default="webcam",
                        help="'webcam' or path to image/video file")
    parser.add_argument("--skip", type=int, default=5,
                        help="Analyze every N frames (default: 5, lower = slower but smoother)")
    parser.add_argument("--save", action="store_true",
                        help="Save output video to output/")
    args = parser.parse_args()

    if args.source == "webcam":
        run_webcam(analyze_every=args.skip, save_output=args.save)
    else:
        run_image(args.source)
