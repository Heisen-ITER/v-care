import math
from collections import deque
import numpy as np
import time

# --- Configuration ---
BLINK_RATE_BASELINE = 18
BLINK_RATE_FATIGUE = 35
BLINK_RATE_HIGH_FATIGUE = 45

EMOTION_STRESS_MAP = {
    "Fear": 0.85, "Angry": 0.80, "Disgust": 0.65, "Sad": 0.55,
    "Surprise": 0.25, "Neutral": 0.0, "Happy": 0.0, "Unknown": 0.0
}

# --- Smoothing & Alerting Configuration ---
SCORE_HISTORY_LENGTH = 7 # Use a slightly shorter window for responsiveness
# --- NEW: Pre-fill history with calm values for a stable start ---
stress_history = deque([0.0] * SCORE_HISTORY_LENGTH, maxlen=SCORE_HISTORY_LENGTH)
fatigue_history = deque([0.0] * SCORE_HISTORY_LENGTH, maxlen=SCORE_HISTORY_LENGTH)

ALERT_THRESHOLD_STRESS = 75
ALERT_THRESHOLD_FATIGUE = 80
ALERT_PERSISTENCE_S = 4
last_stress_alert_time = 0
last_fatigue_alert_time = 0

# --- Helper Functions ---
def _normalize(value, min_val, max_val):
    if max_val == min_val: return 0.0
    score = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, score))

# --- Main Fusion Logic ---
def fuse_data(mavni_data, vani_data):
    global last_stress_alert_time, last_fatigue_alert_time
    
    # --- 1. Calculate RAW scores for the current frame ---
    emotion = mavni_data.get("primary_emotion", "Unknown")
    emotion_score = EMOTION_STRESS_MAP.get(emotion, 0.0)
    anomaly_score = vani_data.get("speech_anomaly_score", 0.0)
    
    # Evidence-Based Stress Logic
    raw_stress_score = emotion_score
    if emotion_score > 0.5 and anomaly_score > 0.4:
        stress_multiplier = 1.0 + (anomaly_score * 1.2)
        raw_stress_score *= stress_multiplier
    elif emotion_score < 0.2 and anomaly_score > 0.6:
        raw_stress_score = anomaly_score * 0.5
        
    raw_stress_score = min(1.0, raw_stress_score)
    
    blink_rate = mavni_data.get("blinks_per_minute", 0)
    blink_rate_score = _normalize(blink_rate, BLINK_RATE_BASELINE, BLINK_RATE_HIGH_FATIGUE)
    fatigue_level_map = {"Low": 0.0, "Moderate": 0.5, "High": 0.9}
    eye_closure_score = fatigue_level_map.get(mavni_data.get("fatigue_level", "Low"), 0.0)
    raw_fatigue_score = max(blink_rate_score, eye_closure_score)

    # --- 2. Add raw scores to history and calculate SMOOTHED score ---
    stress_history.append(raw_stress_score)
    fatigue_history.append(raw_fatigue_score)
    final_stress_score = np.median(list(stress_history))
    final_fatigue_score = np.median(list(fatigue_history))
    
    # --- 3. Calculate Final CWI based on smoothed scores ---
    highest_risk_factor = max(final_stress_score, final_fatigue_score)
    cwi = 100 * (1 - highest_risk_factor)
    
    # --- 4. Alerting System ---
    alert_message = None
    current_time = time.time()
    
    if len(stress_history) == SCORE_HISTORY_LENGTH and all(s * 100 > ALERT_THRESHOLD_STRESS for s in stress_history):
        if (current_time - last_stress_alert_time > ALERT_PERSISTENCE_S * 2):
            alert_message = f"HIGH STRESS DETECTED (Emotion: {emotion})"
            last_stress_alert_time = current_time

    if len(fatigue_history) == SCORE_HISTORY_LENGTH and all(f * 100 > ALERT_THRESHOLD_FATIGUE for f in fatigue_history):
        if (current_time - last_fatigue_alert_time > ALERT_PERSISTENCE_S * 2):
            alert_message = f"HIGH FATIGUE DETECTED (Blinks: {blink_rate} bpm)"
            last_fatigue_alert_time = current_time

    # --- 5. Assemble Final Data Packet ---
    return {
        "stress_level": round(final_stress_score * 100),
        "fatigue_level": round(final_fatigue_score * 100), 
        "cognitive_wellness_index": round(cwi),
        "alert": alert_message,
        "factors": {
            "emotion": emotion,
            "vocal_anomaly_factor": round(anomaly_score, 3),
            "blink_rate": blink_rate
        },
        "raw_data": {
            "video_frame": mavni_data.get("video_frame", None)
        }
    }