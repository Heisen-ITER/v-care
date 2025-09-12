import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import time
import base64

print("Starting MAVNI module...")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

MODEL_PATH = 'best_mobilenet_model'
try:
    emotion_model = tf.saved_model.load(MODEL_PATH)
    predict_fn = emotion_model.signatures['serving_default']
    print("Loaded emotion model.")
except Exception as e:
    print("Could not load model:", e)
    exit(1)

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

emotion_history = deque(maxlen=15)
blinks_in_last_minute = deque()
frames_below_threshold = 0

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


def calc_ear(landmarks, indices):
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in indices])
    v1 = np.linalg.norm(pts[1] - pts[15])
    v2 = np.linalg.norm(pts[2] - pts[14])
    v3 = np.linalg.norm(pts[3] - pts[13])
    h = np.linalg.norm(pts[0] - pts[8])
    return (v1 + v2 + v3) / (3.0 * h) if h != 0 else 0.0


def encode_frame(frame):
    h, w = frame.shape[:2]
    new_w = 320
    new_h = int((new_w / w) * h)
    resized = cv2.resize(frame, (new_w, new_h))
    _, buf = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return base64.b64encode(buf).decode("utf-8")


def analyze_frame(frame):
    global frames_below_threshold

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    output = {
        "fatigue_level": "Low",
        "stress_level": "Low",
        "anxiety_detected": False,
        "primary_emotion": "---",
        "blinks_per_minute": len(blinks_in_last_minute),
        "video_frame": encode_frame(frame)
    }

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        avg_ear = (calc_ear(lm, LEFT_EYE) + calc_ear(lm, RIGHT_EYE)) / 2
        if avg_ear < 0.25:
            frames_below_threshold += 1
        else:
            if frames_below_threshold >= 2:
                blinks_in_last_minute.append(time.time())
            frames_below_threshold = 0

        now = time.time()
        while blinks_in_last_minute and blinks_in_last_minute[0] < now - 60:
            blinks_in_last_minute.popleft()

        bpm = len(blinks_in_last_minute)
        output["blinks_per_minute"] = bpm
        if bpm >= 22:
            output["fatigue_level"] = "High"
        elif bpm >= 18:
            output["fatigue_level"] = "Moderate"

        x = [p.x * w for p in lm]
        y = [p.y * h for p in lm]
        face_crop = frame[int(min(y)):int(max(y)), int(min(x)):int(max(x))]

        if face_crop.size > 0:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            face_input = resized / 255.0
            face_input = np.expand_dims(face_input, axis=(0, -1))
            face_tensor = tf.convert_to_tensor(face_input, dtype=tf.float32)

            try:
                preds = predict_fn(face_tensor)
                out_key = list(preds.keys())[0]
                values = preds[out_key].numpy()[0]
                detected = EMOTION_LABELS[np.argmax(values)]
                emotion_history.append(detected)
            except Exception as e:
                print("Prediction error:", e)

    if emotion_history:
        stable_emotion = max(set(emotion_history), key=emotion_history.count)
        output["primary_emotion"] = stable_emotion
    
    # Remove the old stress/anxiety logic. The Fusion Engine handles this now.
    output.pop("stress_level", None)
    output.pop("anxiety_detected", None)

    return output


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    interval = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if int(time.time() * 1000) % interval == 0:
            result = analyze_frame(frame)
            print("Analysis:", result["primary_emotion"], "| Stress:", result["stress_level"])

        cv2.imshow("MAVNI Module", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()