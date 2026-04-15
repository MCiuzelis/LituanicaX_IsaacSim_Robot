#!/usr/bin/env python3
"""
autopilot.py — Policy-driven RC car control.

Run on the Raspberry Pi:
    python autopilot.py

Open http://<pi-ip>:8080 in a browser. Hold SPACE to activate the policy.
Releasing SPACE (or losing browser focus) immediately stops the car.

Dependencies:
    pip install aiohttp onnxruntime ArducamDepthCamera opencv-python "numpy<2.0.0"
"""

import asyncio
import json
import os
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort
import ArducamDepthCamera as ac
import aiohttp
from aiohttp import web

from hardware import Robot

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
POLICY_PATH = os.path.join(SCRIPT_DIR, "Policy", "policy.onnx")

# ── Config ────────────────────────────────────────────────────────────────────
HOST           = "0.0.0.0"
PORT           = 8080
CAMERA_INDICES = [0, 8]
MAX_DISTANCE   = 2000   # mm — sensor range limit
CONF_THRESHOLD = 30
WARMUP_FRAMES  = 10
FPS_TARGET     = 15     # browser stream fps
POLICY_HZ      = 30

# ── Depth processing constants (must match training pipeline) ─────────────────
DEPTH_RAW_H    = 180
DEPTH_RAW_W    = 240
CROP_TOP_FRAC    = 0.35
CROP_BOTTOM_FRAC = 0.3
POLICY_W         = 120
# Crop: int() truncation mirrors sim's _process_depth_frame
CROP_TOP_ROWS    = int(CROP_TOP_FRAC    * DEPTH_RAW_H)
CROP_BOTTOM_ROWS = int(CROP_BOTTOM_FRAC * DEPTH_RAW_H)
CROPPED_H        = DEPTH_RAW_H - CROP_TOP_ROWS - CROP_BOTTOM_ROWS
POLICY_H         = round(POLICY_W * CROPPED_H / DEPTH_RAW_W)

# ── Policy observation layout ─────────────────────────────────────────────────
PER_CAM_DIM    = POLICY_W * POLICY_H   # 80*31 = 2480
ACTION_HIST    = 30                    # last 15 [throttle, steer] pairs
OBS_DIM        = PER_CAM_DIM * 2 + ACTION_HIST  # 4990

# ── Shared state ──────────────────────────────────────────────────────────────
# Written by camera threads, read by policy thread and broadcast coroutine
latest_frames: dict[int, bytes | None]                         = {0: None, 8: None}
latest_raw:    dict[int, tuple[np.ndarray, np.ndarray] | None] = {0: None, 8: None}
frames_lock = threading.Lock()

# WebSocket clients — only touched from asyncio coroutines, no lock needed
clients: set[web.WebSocketResponse] = set()

# Spacebar gate — set = policy active
space_active = threading.Event()

# Latest action for telemetry display (written by policy thread)
latest_action = [0.0, 0.0]
action_lock   = threading.Lock()

robot: Robot | None = None


# ── Depth processing ──────────────────────────────────────────────────────────

def depth_to_jpeg(depth_buf: np.ndarray, conf_buf: np.ndarray, depth_range: float) -> bytes:
    """Full-resolution rainbow depth image for the browser stream."""
    depth = np.nan_to_num(depth_buf)
    depth[conf_buf < CONF_THRESHOLD] = 0
    color = (depth * (255.0 / depth_range)).astype(np.uint8)
    color = cv2.applyColorMap(color, cv2.COLORMAP_RAINBOW)
    color[conf_buf < CONF_THRESHOLD] = (0, 0, 0)
    color = cv2.flip(color, -1)
    _, jpeg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes()


def process_for_policy(depth_buf: np.ndarray, conf_buf: np.ndarray) -> np.ndarray:
    """
    Reproduces the training camera pipeline:
      1. mm → meters
      2. Mask low-confidence pixels → max sensor range (reads as 'far/clear')
      3. Crop top CROP_TOP_FRAC and bottom CROP_BOTTOM_FRAC rows
      4. Bilinear downscale to POLICY_W × POLICY_H
      5. Normalize: 1.0 - depth_m / (MAX_DISTANCE/1000)
    """
    depth_m = np.nan_to_num(depth_buf).astype(np.float32) / 1000.0
    depth_m[conf_buf < CONF_THRESHOLD] = MAX_DISTANCE / 1000.0

    cropped    = depth_m[CROP_TOP_ROWS : DEPTH_RAW_H - CROP_BOTTOM_ROWS, :]
    resized    = cv2.resize(cropped, (POLICY_W, POLICY_H), interpolation=cv2.INTER_LINEAR)
    normalized = np.clip(1.0 - resized / 4.0, 0.0, 1.0)
    return normalized.flatten().astype(np.float32)


# ── Camera capture (single thread, sequential) ───────────────────────────────

def camera_worker(idx: int):
    while True:
        try:
            cam = ac.ArducamCamera()
            if cam.open(ac.Connection.CSI, idx) != 0:
                print(f"Camera {idx}: failed to open")
                time.sleep(2)
                continue

            cam.start(ac.FrameType.DEPTH)
            cam.setControl(ac.Control.RANGE, MAX_DISTANCE)
            depth_range = cam.getControl(ac.Control.RANGE)

            for _ in range(WARMUP_FRAMES):
                f = cam.requestFrame(2000)
                if f is not None:
                    cam.releaseFrame(f)

            print(f"Camera {idx}: ready")

            while True:
                frame = cam.requestFrame(2000)
                if frame is None or not isinstance(frame, ac.DepthData):
                    continue

                # 🔒 CRITICAL: isolate memory immediately
                depth_buf = np.array(frame.depth_data, copy=True).astype(np.float32, copy=False)
                conf_buf  = np.array(frame.confidence_data, copy=True)

                cam.releaseFrame(frame)

                jpeg = depth_to_jpeg(depth_buf, conf_buf, depth_range)

                depth_flipped = np.ascontiguousarray(np.flip(depth_buf, (0, 1)))
                conf_flipped  = np.ascontiguousarray(np.flip(conf_buf,  (0, 1)))

                with frames_lock:
                    latest_frames[idx] = jpeg
                    latest_raw[idx]    = (depth_flipped, conf_flipped)

        except Exception as e:
            print(f"Camera {idx} error: {e}, restarting in 2s")
            try:
                cam.stop()
                cam.close()
            except:
                pass
            time.sleep(2)


# ── Policy loop ───────────────────────────────────────────────────────────────

def policy_thread() -> None:
    print("Loading policy...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(
        POLICY_PATH, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"Policy loaded — obs={OBS_DIM}  act={2}  target={POLICY_HZ} Hz")

    action_history = np.zeros(ACTION_HIST, dtype=np.float32)
    step_interval  = 1.0 / POLICY_HZ

    while True:
        t_start = time.perf_counter()

        if not space_active.is_set():
            # Not active: zero throttle, disable servo, clear history for clean restart
            robot.set_throttle(0)
            robot.servo_pwm.ChangeDutyCycle(0)
            action_history[:] = 0.0
            with action_lock:
                latest_action[0] = 0.0
                latest_action[1] = 0.0
            time.sleep(step_interval)
            continue

        # Grab latest raw frames — camera 8 = physical left, camera 0 = physical right
        with frames_lock:
            raw_left  = latest_raw.get(8)
            raw_right = latest_raw.get(0)

        if raw_left is None or raw_right is None:
            time.sleep(step_interval)
            continue

        # Build observation vector: [left(2480) | right(2480) | action_hist(30)]
        left_obs  = process_for_policy(*raw_left)
        right_obs = process_for_policy(*raw_right)
        obs = np.concatenate([left_obs, right_obs, action_history])[np.newaxis, :]  # (1, 4990)

        action = sess.run([output_name], {input_name: obs})[0].clip(-1.0, 1.0)[0]
        throttle_raw, steer = float(action[0]), float(action[1])
        # Clamp throttle to [0, 1] — negative output means zero torque,
        # matching the sim's (raw + 1) / 2 remapping where raw=-1 → throttle_norm=0.
        throttle_motor = max(0.0, throttle_raw)

        robot.set_throttle(throttle_motor)
        robot.set_steering(steer)

        # Store RAW policy output in history (matches sim: _prev_action stores actions ∈ [-1, 1])
        action_history = np.roll(action_history, -2)
        action_history[-2] = throttle_raw
        action_history[-1] = steer

        with action_lock:
            latest_action[0] = throttle_motor
            latest_action[1] = steer

        elapsed = time.perf_counter() - t_start
        time.sleep(max(0.0, step_interval - elapsed))


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)
    print(f"Client connected: {request.remote}")

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "space":
                    if data.get("pressed"):
                        space_active.set()
                    else:
                        space_active.clear()
            elif msg.type == aiohttp.WSMsgType.ERROR:
                break
    finally:
        clients.discard(ws)
        space_active.clear()  # always stop on disconnect
        print(f"Client disconnected: {request.remote}")

    return ws


# ── MJPEG stream ──────────────────────────────────────────────────────────────

async def mjpeg_stream(request: web.Request) -> web.StreamResponse:
    index = int(request.match_info["index"])
    resp = web.StreamResponse()
    resp.content_type = "multipart/x-mixed-replace; boundary=frame"
    await resp.prepare(request)
    interval = 1.0 / FPS_TARGET
    try:
        while True:
            with frames_lock:
                jpeg = latest_frames.get(index)
            if jpeg is not None:
                await resp.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(interval)
    except (ConnectionResetError, asyncio.CancelledError, Exception):
        pass
    return resp


# ── Telemetry broadcaster (tiny JSON, no camera frames) ───────────────────────

async def broadcast_telemetry() -> None:
    interval = 1.0 / FPS_TARGET
    while True:
        await asyncio.sleep(interval)

        with action_lock:
            throttle = latest_action[0]
            steer    = latest_action[1]

        payload = json.dumps({
            "type":     "telemetry",
            "throttle": round(throttle, 3),
            "steer":    round(steer, 3),
            "active":   space_active.is_set(),
        })

        dead: set[web.WebSocketResponse] = set()
        for ws in list(clients):
            try:
                await ws.send_str(payload)
            except Exception:
                dead.add(ws)
        clients -= dead


# ── HTML UI ───────────────────────────────────────────────────────────────────

HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RC Car Autopilot</title>
<style>
  body {
    background: #111; color: #eee; font-family: monospace;
    margin: 0; display: flex; flex-direction: column;
    align-items: center; padding: 20px; user-select: none;
  }
  h1 { margin: 0 0 14px; }
  #cameras { display: flex; gap: 14px; }
  .cam-wrap img { width: 480px; border: 2px solid #444; display: block; background: #1a1a1a; min-height: 180px; }
  .label { text-align: center; margin-top: 5px; font-size: 13px; color: #aaa; }
  #status { margin-top: 12px; font-size: 14px; }
  #status.active   { color: #4f4; }
  #status.inactive { color: #f84; }
  #hint { margin-top: 4px; font-size: 12px; color: #666; }
  #telemetry {
    margin-top: 14px; display: grid;
    grid-template-columns: repeat(2, 190px); gap: 12px;
  }
  .gauge-box {
    background: #1e1e1e; border: 1px solid #444;
    border-radius: 6px; padding: 12px;
  }
  .gauge-label { font-size: 11px; color: #888; margin-bottom: 6px; letter-spacing: 1px; }
  .gauge-value { font-size: 22px; font-weight: bold; }
  .bar-bg { background: #2a2a2a; height: 8px; border-radius: 4px; margin-top: 8px; position: relative; }
  .bar-center { position: absolute; top: 0; left: 50%; width: 1px; height: 100%; background: #555; }
  .bar-fg { height: 8px; border-radius: 4px; transition: width 0.06s, margin 0.06s, background 0.1s; }
  #space-btn {
    margin-top: 16px; padding: 12px 40px;
    border: 2px solid #555; border-radius: 8px;
    font-size: 16px; font-family: monospace; color: #888;
    background: #1a1a1a; transition: all 0.05s;
  }
  #space-btn.active { border-color: #4f4; color: #4f4; background: #0a1f0a; }
</style>
</head>
<body>
<h1>RC Car Autopilot</h1>

<div id="cameras">
  <div class="cam-wrap">
    <img src="/stream/8" alt="Camera 8">
    <div class="label">Camera 8 (left)</div>
  </div>
  <div class="cam-wrap">
    <img src="/stream/0" alt="Camera 0">
    <div class="label">Camera 0 (right)</div>
  </div>
</div>

<div id="status" class="inactive">Connecting\u2026</div>
<div id="hint">Hold SPACE to activate autopilot \u00b7 click this tab first to capture keys</div>

<div id="telemetry">
  <div class="gauge-box">
    <div class="gauge-label">THROTTLE</div>
    <div class="gauge-value" id="tval">0.000</div>
    <div class="bar-bg">
      <div class="bar-fg" id="tbar" style="width:0%;background:#555;"></div>
    </div>
  </div>
  <div class="gauge-box">
    <div class="gauge-label">STEER</div>
    <div class="gauge-value" id="sval">0.000</div>
    <div class="bar-bg">
      <div class="bar-center"></div>
      <div class="bar-fg" id="sbar" style="width:50%;background:#555;margin-left:50%;"></div>
    </div>
  </div>
</div>

<div id="space-btn">SPACE \u2014 hold to activate</div>

<script>
const ws       = new WebSocket(`ws://${location.host}/ws`);
const statusEl = document.getElementById('status');
const tval     = document.getElementById('tval');
const tbar     = document.getElementById('tbar');
const sval     = document.getElementById('sval');
const sbar     = document.getElementById('sbar');
const spaceBtn = document.getElementById('space-btn');
let   held     = false;

ws.onopen  = () => { statusEl.textContent = 'Connected \u2014 hold SPACE to drive'; statusEl.className = 'inactive'; };
ws.onclose = () => { statusEl.textContent = 'Disconnected'; statusEl.className = 'inactive'; };
ws.onerror = () => { statusEl.textContent = 'Connection error'; statusEl.className = 'inactive'; };

ws.onmessage = ({ data }) => {
  const msg = JSON.parse(data);
  if (msg.type !== 'telemetry') return;

  // Throttle: map [-1,1] -> [0,100]%
  const tp = ((msg.throttle + 1) / 2 * 100).toFixed(1);
  tval.textContent = (msg.throttle >= 0 ? '+' : '') + msg.throttle.toFixed(3);
  tbar.style.width      = tp + '%';
  tbar.style.marginLeft = '0';
  tbar.style.background = msg.throttle >  0.05 ? '#4af'
                        : msg.throttle < -0.05 ? '#f64' : '#555';

  // Steer: center bar — grow left or right from 50%
  const sv = msg.steer;
  sval.textContent = (sv >= 0 ? '+' : '') + sv.toFixed(3);
  if (sv >= 0) {
    sbar.style.marginLeft = '50%';
    sbar.style.width      = (sv * 50).toFixed(1) + '%';
  } else {
    sbar.style.marginLeft = ((0.5 + sv / 2) * 100).toFixed(1) + '%';
    sbar.style.width      = (-sv * 50).toFixed(1) + '%';
  }
  sbar.style.background = Math.abs(sv) > 0.05 ? '#fa4' : '#555';

  statusEl.textContent = msg.active ? 'AUTOPILOT ACTIVE' : 'Policy inactive \u2014 hold SPACE';
  statusEl.className   = msg.active ? 'active' : 'inactive';

};

function sendSpace(pressed) {
  if (ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({ type: 'space', pressed }));
  spaceBtn.classList.toggle('active', pressed);
}

document.addEventListener('keydown', e => {
  if (e.code === 'Space' && !held) {
    e.preventDefault();
    held = true;
    sendSpace(true);
  }
});
document.addEventListener('keyup', e => {
  if (e.code === 'Space') {
    e.preventDefault();
    held = false;
    sendSpace(false);
  }
});
window.addEventListener('blur', () => {
  if (held) { held = false; sendSpace(false); }
});
</script>
</body>
</html>
"""


async def index_handler(request: web.Request) -> web.Response:
    return web.Response(text=HTML, content_type="text/html")


# ── Lifecycle ─────────────────────────────────────────────────────────────────

async def on_startup(app: web.Application) -> None:
    app["telemetry_task"] = asyncio.create_task(broadcast_telemetry())


async def on_cleanup(app: web.Application) -> None:
    app["telemetry_task"].cancel()
    try:
        await app["telemetry_task"]
    except asyncio.CancelledError:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global robot

    for idx in CAMERA_INDICES:
        threading.Thread(target=camera_worker, args=(idx,), daemon=True).start()

    robot = Robot()
    print("Hardware initialised.")

    threading.Thread(target=policy_thread, daemon=True).start()

    app = web.Application()
    app.router.add_get("/",               index_handler)
    app.router.add_get("/ws",             ws_handler)
    app.router.add_get("/stream/{index}", mjpeg_stream)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    print(f"\nOpen http://<raspberry-pi-ip>:{PORT} in your browser\n")
    try:
        web.run_app(app, host=HOST, port=PORT)
    finally:
        space_active.clear()
        if robot:
            robot.stop()


if __name__ == "__main__":
    main()
