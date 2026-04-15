#!/usr/bin/env python3
"""
manual_control.py — RC car browser-based manual control.

Run on the Raspberry Pi:
    python manual_control.py

Then open http://<pi-ip>:8080 in a browser on the same network.
WASD drives the car. Steering is calibrated automatically on startup.

Dependencies:
    pip install aiohttp ArducamDepthCamera opencv-python "numpy<2.0.0" ev3_dc
"""

import asyncio
import json
import threading
import time

import cv2
import numpy as np
import ArducamDepthCamera as ac
import ev3_dc as ev3
import aiohttp
from aiohttp import web

# ── Config ────────────────────────────────────────────────────────────────────
HOST             = "0.0.0.0"
PORT             = 8080
CAMERA_INDICES   = [0, 8]
MAX_DISTANCE     = 4000
CONF_THRESHOLD   = 30
WARMUP_FRAMES    = 10
DRIVE_SPEED      = 100
STEER_SPEED      = 50
FPS_TARGET       = 30

# ── Shared state ──────────────────────────────────────────────────────────────
# Camera frames (written by camera threads, read by asyncio broadcast)
latest_frames: dict[int, bytes | None] = {0: None, 8: None}
frames_lock = threading.Lock()

# Key state (accessed from both asyncio and executor threads)
keys_pressed: set[str] = set()
keys_lock = threading.Lock()

robot: "Robot | None" = None


# ── Camera capture ────────────────────────────────────────────────────────────

def depth_to_jpeg(depth_buf: np.ndarray, conf_buf: np.ndarray, depth_range: float) -> bytes:
    depth_buf = np.nan_to_num(depth_buf)
    depth_buf[conf_buf < CONF_THRESHOLD] = 0

    color = (depth_buf * (255.0 / depth_range)).astype(np.uint8)
    color = cv2.applyColorMap(color, cv2.COLORMAP_RAINBOW)
    color[conf_buf < CONF_THRESHOLD] = (0, 0, 0)

    # 🔁 Flip upside down (vertical + horizontal)
    color = cv2.flip(color, -1)

    _, jpeg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes()


def camera_thread() -> None:
    """
    Single thread that owns all cameras and captures them sequentially.
    This prevents the ArduCam SDK's shared C-level depth buffer from being
    written by one camera's internal capture thread while we are reading
    the other camera's frame, which causes both feeds to appear overlaid.
    """
    while True:
        # ── Open all cameras ──────────────────────────────────────────────────
        cams: dict[int, tuple[ac.ArducamCamera, float]] = {}
        for idx in CAMERA_INDICES:
            try:
                cam = ac.ArducamCamera()
                if cam.open(ac.Connection.CSI, idx) != 0:
                    print(f"Camera {idx}: failed to open")
                    continue
                cam.start(ac.FrameType.DEPTH)
                cam.setControl(ac.Control.RANGE, MAX_DISTANCE)
                depth_range = cam.getControl(ac.Control.RANGE)
                for _ in range(WARMUP_FRAMES):
                    f = cam.requestFrame(2000)
                    if f is not None:
                        cam.releaseFrame(f)
                cams[idx] = (cam, depth_range)
                print(f"Camera {idx}: ready")
            except Exception as exc:
                print(f"Camera {idx}: init error {exc!r}")

        if not cams:
            print("No cameras opened, retrying in 2s")
            time.sleep(2)
            continue

        # ── Capture loop: one camera at a time (STRICT isolation) ─────────────
        try:
            while True:
                for idx, (cam, depth_range) in cams.items():

                    frame = cam.requestFrame(2000)
                    if frame is None or not isinstance(frame, ac.DepthData):
                        continue

                    # 🚨 CRITICAL: copy AND process BEFORE releasing or switching camera
                    depth_buf = np.array(frame.depth_data, dtype=np.float32).copy()
                    conf_buf  = np.array(frame.confidence_data, dtype=np.uint8).copy()

                    # Process immediately while memory is still valid
                    jpeg = depth_to_jpeg(depth_buf, conf_buf, depth_range)

                    cam.releaseFrame(frame)  # release ONLY after processing

                    # Store result
                    with frames_lock:
                        latest_frames[idx] = jpeg

        except Exception as exc:
            print(f"Camera thread error: {exc!r}, restarting in 2s")
            for cam, _ in cams.values():
                try:
                    cam.stop()
                    cam.close()
                except Exception:
                    pass
            time.sleep(2)


# ── Robot / EV3 ───────────────────────────────────────────────────────────────

class Robot:
    """
    EV3 port assignment:
        PORT_A — steering motor
        PORT_B — left drive motor
        PORT_C — right drive motor
    """

    def __init__(self) -> None:
        self._ev3    = ev3.EV3(protocol=ev3.USB)
        self.steer   = ev3.Motor(ev3.PORT_D, ev3_obj=self._ev3)
        self.drive_b = ev3.Motor(ev3.PORT_B, ev3_obj=self._ev3)
        self.drive_c = ev3.Motor(ev3.PORT_C, ev3_obj=self._ev3)

        self.steer.speed   = STEER_SPEED
        self.drive_b.speed = DRIVE_SPEED
        self.drive_c.speed = DRIVE_SPEED

        # Set by calibrate_steering()
        self._steer_center: float    = 0.0
        self._steer_amplitude: float = 40  # degrees, fallback default

    def calibrate_steering(self) -> None:
        """
        Sweep steering to hardware limits, measure actual positions,
        compute center and amplitude, then park at center.
        After this call, set_steering(0) means mechanical center.
        """
        print("Calibrating steering...")

        print("  Moving to +180° (will stall at limit)...")
        self.steer.start_move_to(180, brake=True)
        time.sleep(2)
        pos_max = self.steer.position
        print(f"  Max position: {pos_max}°")

        print("  Moving to -180° (will stall at limit)...")
        self.steer.start_move_to(-180, brake=True)
        time.sleep(2)
        pos_min = self.steer.position
        print(f"  Min position: {pos_min}°")

        self._steer_center    = (pos_max + pos_min) / 2.0
        self._steer_amplitude = (pos_max - pos_min) / 2.0
        print(f"  Center={self._steer_center:.1f}°  Amplitude=±{self._steer_amplitude:.1f}°")

        print("  Moving to center...")
        self.steer.start_move_to(round(self._steer_center), brake=True)
        time.sleep(1)
        print("  Calibration complete. Steering zeroed at mechanical center.")

    def set_steering(self, value: float) -> None:
        """value in [-1, 1]; 0 = mechanical center, ±1 = hardware limit."""
        target = self._steer_center + value * self._steer_amplitude
        self.steer.start_move_to(round(target), brake=True)

    def drive(self, direction: int) -> None:
        """direction: +1 = forward, -1 = reverse."""
        self.drive_b.start_move(direction=direction)
        self.drive_c.start_move(direction=-direction)

    def stop_drive(self) -> None:
        self.drive_b.stop(brake=True)
        self.drive_c.stop(brake=True)

    def shutdown(self) -> None:
        """Coast all motors; call on program exit."""
        self.drive_b.stop(brake=False)
        self.drive_c.stop(brake=False)
        self.steer.stop(brake=False)


def apply_keys() -> None:
    """Translate current key state to motor commands. Runs in a thread."""
    try:
        with keys_lock:
            fwd  = "w" in keys_pressed
            back = "s" in keys_pressed
            left = "a" in keys_pressed
            rght = "d" in keys_pressed

        if fwd and not back:
            robot.drive(1)
        elif back and not fwd:
            robot.drive(-1)
        else:
            robot.stop_drive()

        if left and not rght:
            robot.set_steering(-1.0)
        elif rght and not left:
            robot.set_steering(1.0)
        else:
            robot.set_steering(0.0)

    except Exception as exc:
        print(f"Motor control error: {exc!r}")


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    loop = asyncio.get_event_loop()
    print(f"Client connected: {request.remote}")

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "key":
                    key     = data.get("key", "").lower()
                    pressed = bool(data.get("pressed"))
                    if key in ("w", "a", "s", "d"):
                        with keys_lock:
                            if pressed:
                                keys_pressed.add(key)
                            else:
                                keys_pressed.discard(key)
                        await loop.run_in_executor(None, apply_keys)

            elif msg.type == aiohttp.WSMsgType.ERROR:
                break

    finally:
        print(f"Client disconnected: {request.remote}")
        # Stop the car when the controlling browser tab closes
        with keys_lock:
            keys_pressed.clear()
        await loop.run_in_executor(None, robot.stop_drive)

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


# ── HTML UI ───────────────────────────────────────────────────────────────────

HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RC Car Control</title>
<style>
  body {
    background: #111; color: #eee; font-family: monospace;
    margin: 0; display: flex; flex-direction: column;
    align-items: center; padding: 20px; user-select: none;
  }
  h1 { margin: 0 0 14px; }
  #cameras { display: flex; gap: 14px; }
  .cam-wrap img { width: 480px; border: 2px solid #444; display: block; background: #222; }
  .label { text-align: center; margin-top: 5px; font-size: 13px; color: #aaa; }
  #status { margin-top: 12px; font-size: 14px; color: #6f6; }
  #hint   { margin-top: 4px; font-size: 12px; color: #666; }
  #keydisp {
    margin-top: 12px;
    display: grid;
    grid-template-columns: repeat(3, 42px);
    grid-template-rows: repeat(2, 42px);
    gap: 5px;
  }
  .key {
    background: #2a2a2a; border: 1px solid #555;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; border-radius: 5px; transition: background 0.05s;
  }
  .key.active { background: #2a6; border-color: #4c8; color: #fff; }
  #key-w { grid-column: 2; grid-row: 1; }
  #key-a { grid-column: 1; grid-row: 2; }
  #key-s { grid-column: 2; grid-row: 2; }
  #key-d { grid-column: 3; grid-row: 2; }
</style>
</head>
<body>
<h1>RC Car Control</h1>
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
<div id="status">Connecting\u2026</div>
<div id="hint">Click this tab to capture keyboard input \u00b7 WASD to drive</div>
<div id="keydisp">
  <div class="key" id="key-w">W</div>
  <div class="key" id="key-a">A</div>
  <div class="key" id="key-s">S</div>
  <div class="key" id="key-d">D</div>
</div>
<script>
const ws     = new WebSocket(`ws://${location.host}/ws`);
const status = document.getElementById('status');
const keyMap = { w:'key-w', a:'key-a', s:'key-s', d:'key-d' };
const VALID  = new Set(['w','a','s','d']);
const held   = new Set();

ws.onopen  = () => { status.textContent = 'Connected \u2014 use WASD to drive'; };
ws.onclose = () => { status.textContent = 'Disconnected'; };
ws.onerror = () => { status.textContent = 'Connection error'; };

function sendKey(key, pressed) {
  if (ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({ type: 'key', key, pressed }));
  const el = document.getElementById(keyMap[key]);
  if (el) el.classList.toggle('active', pressed);
}

document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (!VALID.has(k) || held.has(k)) return;
  e.preventDefault();
  held.add(k);
  sendKey(k, true);
});

document.addEventListener('keyup', e => {
  const k = e.key.toLowerCase();
  if (!VALID.has(k)) return;
  e.preventDefault();
  held.delete(k);
  sendKey(k, false);
});

// Release all keys when tab loses focus so the car doesn't drive away
window.addEventListener('blur', () => {
  held.forEach(k => sendKey(k, false));
  held.clear();
});
</script>
</body>
</html>
"""


async def index_handler(request: web.Request) -> web.Response:
    return web.Response(text=HTML, content_type="text/html")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global robot

    # Single camera thread owns all cameras sequentially — prevents SDK buffer mixing
    threading.Thread(target=camera_thread, daemon=True).start()

    # Connect to EV3 and calibrate steering before accepting connections
    robot = Robot()
    # robot.calibrate_steering()

    # Build and run the web application
    app = web.Application()
    app.router.add_get("/",              index_handler)
    app.router.add_get("/ws",            ws_handler)
    app.router.add_get("/stream/{index}", mjpeg_stream)

    print(f"\nOpen http://<pi-ip>:{PORT} in your browser\n")
    try:
        web.run_app(app, host=HOST, port=PORT)
    finally:
        if robot:
            robot.shutdown()


if __name__ == "__main__":
    main()