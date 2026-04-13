# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RC car platform running on a Raspberry Pi 5, controlled via LEGO EV3 motors over USB. Two ArduCam TOF depth cameras provide the vision input. A MuSHR-based neural network policy runs at 30 Hz to drive autonomously.

## Running Scripts

```bash
# Run trained policy (benchmark with random noise input)
python Policy/run_policy.py                  # ONNX backend (default)
python Policy/run_policy.py --backend torch  # TorchScript backend
python Policy/run_policy.py --steps 500      # exit after N steps

# Capture depth images from both cameras to disk
python Arducam_tof_camera/save_depth_image.py
# Outputs: depth_0.png, depth_8.png

# Run EV3 motor test (drives for 2s, then steers to -90°)
python Ev3/ev3_test.py
```

## Architecture

### Policy (`Policy/`)
- **Input**: 4,990-dim float32 vector — `[0:2480]` left camera depth, `[2480:4960]` right camera depth, `[4960:4990]` action history
- **Output**: `[throttle, steer]` both in `[-1, 1]`, clipped from raw actor mean
- **Backends**: `policy.onnx` (preferred, 4-thread ONNX Runtime) or `policy.pt` (TorchScript)
- **Target frequency**: 30 Hz

### Camera (`Arducam_tof_camera/`)
- Two cameras on CSI indices **0** and **8**, opened in parallel threads
- `save_depth_image.py` is the reference for camera capture: 10-frame warmup, then one frame per camera
- Each frame provides `depth_data` (mm float array) and `confidence_data`; pixels below `CONFIDENCE_THRESHOLD = 30` are masked to 0
- Colorized with `cv2.COLORMAP_RAINBOW` scaled to `MAX_DISTANCE = 4000` mm
- `ArducamDepthCamera` Python library (`ac`) wraps the CSI cameras; `ac.Connection.CSI`, `ac.FrameType.DEPTH`, `ac.Control.RANGE` are the key enums

### EV3 Motor Control (`Ev3/`)
- Uses the `ev3_dc` library over USB (`ev3.EV3(protocol=ev3.USB)`)
- **Port mapping**:
  - `PORT_A` — steering motor
  - `PORT_B` — left drive motor
  - `PORT_C` — right drive motor
- Drive: run PORT_B forward and PORT_C backward (differential) for forward motion
- Steering: `motor.start_move_to(angle, brake=True)` — absolute encoder degrees from power-on position; calibration determines center and amplitude
- `Robot` is a context manager; `__exit__` coasts all motors to stop

### Manual Control Script (`manual_control.py`)
Run on the Pi, then open `http://<pi-ip>:8080` in a browser on the same network.

```bash
python manual_control.py
```

- On startup: calibrates steering by sweeping to +180° (stalls), reading encoder, sweeping to -180°, reading encoder — computes center and amplitude, parks at center
- Serves a browser UI (aiohttp) showing both depth camera feeds side-by-side as live JPEG frames (~15 fps via WebSocket)
- WASD keyboard input from browser → `drive()`/`stop_drive()`/`set_steering()` calls; car stops automatically when the browser tab loses focus or disconnects
- `set_steering(value)` maps `[-1, 1]` to the calibrated hardware range; 0 = mechanical center

## Dependencies

```
ArducamDepthCamera    # camera SDK
opencv-python
numpy<2.0.0
ev3_dc                # EV3 USB control
onnxruntime           # for Policy ONNX backend
torch                 # for Policy TorchScript backend (optional)
```
