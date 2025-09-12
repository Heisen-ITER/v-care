import asyncio
import socketio
from fastapi import FastAPI
import cv2
import logging
import traceback
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# --- Import Core Modules ---
try:
    import mavni
    import vani
    import engine
except ImportError as e:
    print(f"FATAL ERROR: A core module is missing. Details: {e}")
    exit(1)

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global State ---
cap = None
data_task = None
connected_clients = set()
# --- NEW: Thread pool for running heavy computations ---
executor = ThreadPoolExecutor(max_workers=2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cap
    try:
        # MAVNI Startup
        logger.info("Initializing MAVNI (Webcam)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): cap = cv2.VideoCapture(1)
        if not cap.isOpened(): raise IOError("Cannot open webcam.")
        logger.info("MAVNI webcam initialized.")

        # VANI Startup & Calibration
        logger.info("Initializing VANI (Microphone)...")
        vani.start_audio_capture()
        await asyncio.sleep(1)
        # --- NEW: Run blocking calibration in a separate thread ---
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, vani.calibrate, 10)
        logger.info("VANI calibration complete. Server is ready.")

    except Exception as e:
        logger.error(f"FATAL ERROR during server startup: {e}")
        if cap and cap.isOpened(): cap.release()
        cap = None
    
    yield
    
    logger.info("Server shutting down...")
    if cap: cap.release()
    vani.stop_audio_capture()
    executor.shutdown(wait=True)
    logger.info("All resources released.")

app = FastAPI(lifespan=lifespan)
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
app.mount('/socket.io', socketio.ASGIApp(sio))

async def send_data_updates():
    logger.info("Starting data update task...")
    loop = asyncio.get_running_loop()
    while connected_clients:
        try:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.5)
                continue
            
            # --- NEW: Run analysis in background threads to prevent blocking ---
            mavni_task = loop.run_in_executor(executor, mavni.analyze_frame, frame)
            vani_task = loop.run_in_executor(executor, vani.analyze_audio)
            
            mavni_data = await mavni_task
            vani_data = await vani_task
            
            # Fusion is lightweight and can run in the main loop
            fused_data = engine.fuse_data(mavni_data, vani_data)
            
            await sio.emit('update_data', fused_data)
            logger.info(f"CWI: {fused_data['cognitive_wellness_index']} | Stress: {fused_data['stress_level']}% | Fatigue: {fused_data['fatigue_level']}% | Anomaly: {fused_data['factors']['vocal_anomaly_factor']:.2f}")
            
            await asyncio.sleep(0.2)
            
        except Exception as e:
            logger.error(f"Error in data update loop: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(1)
            
    logger.info("Data update task stopped.")
    global data_task
    data_task = None

@sio.event
async def connect(sid, environ):
    global data_task
    logger.info(f"Client connected: {sid}")
    connected_clients.add(sid)
    if data_task is None or data_task.done():
        data_task = asyncio.create_task(send_data_updates())

@sio.event
def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")
    connected_clients.discard(sid)

@app.get("/")
def read_root():
    return {"status": "Cognitive Wellness Monitoring Backend is running"}