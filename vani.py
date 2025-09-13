import sounddevice as sd
import numpy as np
import librosa
import logging
from collections import deque
import threading
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SAMPLE_RATE = 22050
ANALYSIS_DURATION_S = 1.5 
ANALYSIS_SAMPLES = int(SAMPLE_RATE * ANALYSIS_DURATION_S)
BUFFER_SAMPLES = ANALYSIS_SAMPLES * 2 
N_MFCC = 13 
SILENCE_THRESHOLD = 0.005
ANOMALY_SMOOTHING_WINDOW = 3

audio_buffer = deque(maxlen=BUFFER_SAMPLES)
audio_capture_thread = None
is_capturing = False
is_calibrated = False
baseline_mfcc_mean = None
baseline_mfcc_std = None
anomaly_history = deque(maxlen=ANOMALY_SMOOTHING_WINDOW)

def _audio_callback(indata, frames, time, status):
    if status: logger.warning(f"Audio stream status: {status}")
    audio_buffer.extend(indata.flatten())

def start_audio_capture():
    global is_capturing, audio_capture_thread
    if is_capturing: return
    try:
        if not any(d['max_input_channels'] > 0 for d in sd.query_devices()):
            logger.error("No microphone found. VANI module will be disabled.")
            return

        def _capture_loop():
            global is_capturing
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=_audio_callback, dtype='float32'):
                is_capturing = True
                while is_capturing: time.sleep(0.1)
        audio_capture_thread = threading.Thread(target=_capture_loop, daemon=True)
        audio_capture_thread.start()
        logger.info("VANI audio capture started.")
    except Exception as e:
        logger.error(f"Failed to start audio stream: {e}")

def stop_audio_capture():
    global is_capturing
    if is_capturing:
        is_capturing = False
        if audio_capture_thread: audio_capture_thread.join(timeout=2)
        logger.info("VANI audio capture stopped.")

def calibrate(duration=12):
    global baseline_mfcc_mean, baseline_mfcc_std, is_calibrated
    if not is_capturing:
        start_audio_capture()
        time.sleep(1)

    print("\n" + "="*50)
    print(" VANI CALIBRATION ".center(50, '='))
    print(f"\nPlease speak normally and clearly for {duration} seconds.")
    print("This is the most important step for accuracy.")
    print("Starting in 3...2...1...")
    time.sleep(3)
    print(">>> SPEAK NOW <<<")

    calibration_end_time = time.time() + duration
    all_mfccs = []
    
    audio_buffer.clear()
    time.sleep(0.5)

    while time.time() < calibration_end_time:
        if len(audio_buffer) >= ANALYSIS_SAMPLES:
            audio_snapshot = np.array(list(audio_buffer))[-ANALYSIS_SAMPLES:]
            if np.sqrt(np.mean(audio_snapshot**2)) > SILENCE_THRESHOLD:
                mfccs = librosa.feature.mfcc(y=audio_snapshot, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
                all_mfccs.append(np.mean(mfccs, axis=1))
        time.sleep(0.25)

    print(">>> CALIBRATION ENDED <<<")
    print("="*50 + "\n")

    if len(all_mfccs) < 5:
        logger.error("VANI calibration failed: Not enough clear speech detected.")
        is_calibrated = False
        return

    baseline_mfcc_mean = np.mean(all_mfccs, axis=0)
    baseline_mfcc_std = np.std(all_mfccs, axis=0)
    is_calibrated = True
    logger.info("VANI calibration successful. Voice profile created.")

def analyze_audio():
    output = {"vocal_pitch": 0.0, "vocal_energy": 0.0, "speech_anomaly_score": 0.0}

    if not is_calibrated or len(audio_buffer) < ANALYSIS_SAMPLES:
        if anomaly_history:
            output["speech_anomaly_score"] = float(np.median(anomaly_history))
        return output

    audio_snapshot = np.array(list(audio_buffer))[-ANALYSIS_SAMPLES:]
    rms_energy = np.sqrt(np.mean(audio_snapshot**2))
    output["vocal_energy"] = float(rms_energy)
    
    raw_anomaly_score = 0.0

    if rms_energy > SILENCE_THRESHOLD:
        pitches, voiced_flags, _ = librosa.pyin(y=audio_snapshot, fmin=80, fmax=400, sr=SAMPLE_RATE)
        valid_pitches = pitches[voiced_flags]
        if len(valid_pitches) > 0:
            output["vocal_pitch"] = float(np.mean(valid_pitches))

        current_mfccs = librosa.feature.mfcc(y=audio_snapshot, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        current_mfcc_mean = np.mean(current_mfccs, axis=1)
        
        z_scores = (current_mfcc_mean - baseline_mfcc_mean) / (baseline_mfcc_std + 1e-6)
        anomaly_score = np.mean(np.abs(z_scores))
        normalized_anomaly_score = min(1.0, anomaly_score / 4.0)
        raw_anomaly_score = normalized_anomaly_score
    
    anomaly_history.append(raw_anomaly_score)
    output["speech_anomaly_score"] = float(np.median(anomaly_history))
        
    return output

if __name__ == '__main__':
    start_audio_capture()
    try:
        calibrate(duration=8)
        
        if is_calibrated:
            print("\n--- Live Analysis Started ---")
            print("1. Speak normally for a few seconds.")
            print("2. Then, try to stutter or speak in a strained voice.")
            for i in range(25):
                time.sleep(1)
                analysis = analyze_audio()
                status = "ANOMALY" if analysis['speech_anomaly_score'] > 0.4 else "Normal"
                print(f"--> VANI Analysis: Anomaly Score = {analysis['speech_anomaly_score']:.2f}  ({status})")
    finally:
        stop_audio_capture()
