"""
Local NVIDIA GPU Vision Server (GTX 1650 Ti 4GB VRAM optimized)
- Open-vocabulary real-time text segmentation: YOLOE-11m-seg (any English object prompt, exact pixel polygon mask)
- Stateful Visual Memory Bank tracking: SAM 2.1 Tiny (SAM2DynamicInteractivePredictor)
  -> Remembers the exact visual appearance of a clicked or YOLOE-detected object across frames,
     fast motion, and temporary occlusion!
- Zero cloud API calls for vision/tracking; streams lightweight 0..1000 polygon vertices over WebSocket.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

# Keep all model weights & caches on D:\ai-glasses-gpu when available (protects low-space C: drive)
_DEFAULT_GPU_ROOT = Path(os.environ.get("VISION_GPU_ROOT", r"D:\ai-glasses-gpu"))
if _DEFAULT_GPU_ROOT.exists():
    MODELS_DIR = _DEFAULT_GPU_ROOT / "models"
    CACHE_DIR = _DEFAULT_GPU_ROOT / "cache"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
    os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))
    os.environ.setdefault("TIMM_CACHE_DIR", str(CACHE_DIR / "timm"))
else:
    MODELS_DIR = Path(__file__).resolve().parent / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Ensure Ultralytics auxiliary files (e.g. mobileclip_blt.ts) resolve inside MODELS_DIR on D:
os.chdir(MODELS_DIR)

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLOE
from ultralytics.models.sam import SAM2DynamicInteractivePredictor

app = FastAPI(title="AI Glasses GPU Vision Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
YOLOE_WEIGHTS = "yoloe-11m-seg.pt"
SAM_WEIGHTS = "sam2.1_t.pt"

yoloe_model: YOLOE | None = None
sam_predictor: SAM2DynamicInteractivePredictor | None = None
current_yoloe_key: str | None = None
prompt_pe_cache: dict[str, Any] = {}


def get_yoloe() -> YOLOE:
    global yoloe_model
    if yoloe_model is None:
        weights_path = str(MODELS_DIR / YOLOE_WEIGHTS)
        print(f"[VISION] Loading {weights_path} on {DEVICE} ...")
        yoloe_model = YOLOE(weights_path)
        yoloe_model.to(DEVICE)
        print(f"[VISION] {YOLOE_WEIGHTS} ready!")
    return yoloe_model


def get_sam_memory() -> SAM2DynamicInteractivePredictor:
    global sam_predictor
    if sam_predictor is None:
        weights_path = str(MODELS_DIR / SAM_WEIGHTS)
        print(f"[VISION] Loading stateful SAM 2.1 Memory Tracker ({weights_path}) on {DEVICE} ...")
        sam_predictor = SAM2DynamicInteractivePredictor(
            overrides=dict(
                model=weights_path,
                imgsz=512,
                conf=0.08,  # SAM2 object_score_logits/32 is ~0.15-0.32 for present objects, 0 for absent
                device=DEVICE,
                verbose=False,
                save=False,
            ),
            max_obj_num=1,
        )
        print("[VISION] SAM 2.1 Memory Tracker ready!")
    return sam_predictor


def reset_sam_memory(pred: SAM2DynamicInteractivePredictor) -> None:
    pred.memory_bank.clear()
    pred.obj_idx_set.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mask_to_norm_polygon(mask_bool: np.ndarray, img_w: int, img_h: int) -> tuple[list[list[int]], list[int], list[int]]:
    """Convert binary HxW mask to simplified 0..1000 polygon [[x,y],...], center [x,y], and box [w,h]."""
    mask_u8 = (mask_bool.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], [500, 500], [80, 80]

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 25:
        return [], [500, 500], [80, 80]

    # Simplify contour to clean ~12-36 vertices for silky-smooth SVG rendering
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.011 * peri, True)
    pts = approx.reshape(-1, 2)

    poly: list[list[int]] = []
    for px, py in pts:
        nx = int(max(0, min(1000, round((px / img_w) * 1000))))
        ny = int(max(0, min(1000, round((py / img_h) * 1000))))
        poly.append([nx, ny])

    x, y, w, h = cv2.boundingRect(largest)
    cx = int(max(0, min(1000, round(((x + w / 2) / img_w) * 1000))))
    cy = int(max(0, min(1000, round(((y + h / 2) / img_h) * 1000))))
    nw = int(max(20, min(1000, round((w / img_w) * 1000))))
    nh = int(max(20, min(1000, round((h / img_h) * 1000))))
    return poly, [cx, cy], [nw, nh]


def decode_jpeg(b64_str: str) -> np.ndarray | None:
    try:
        raw = base64.b64decode(b64_str)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


@app.get("/health")
def health() -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return {
        "ok": True,
        "device": DEVICE,
        "gpu": gpu_name,
        "yoloeModel": YOLOE_WEIGHTS,
        "samModel": "SAM-2.1-Memory",
        "yoloeLoaded": yoloe_model is not None,
        "samLoaded": sam_predictor is not None,
    }


@app.websocket("/ws/vision")
async def ws_vision(ws: WebSocket) -> None:
    global current_yoloe_key
    await ws.accept()

    mode: str = "idle"  # "idle" | "text" | "sam_memory"
    pending_point_norm: list[float] | None = None
    sam_initialized = False
    last_center_norm: tuple[float, float] = (500.0, 500.0)
    last_yoloe_box_px: list[float] | None = None
    last_frame_bgr: np.ndarray | None = None
    miss_count = 0

    try:
        while True:
            raw_msg = await ws.receive_text()
            msg = json.loads(raw_msg)
            cmd = msg.get("cmd")

            if cmd == "clear":
                mode = "idle"
                pending_point_norm = None
                sam_initialized = False
                last_yoloe_box_px = None
                last_frame_bgr = None
                miss_count = 0
                if sam_predictor is not None:
                    reset_sam_memory(sam_predictor)
                await ws.send_json({"type": "cleared"})
                continue

            if cmd == "track_text":
                raw_prompt = str(msg.get("prompt", "")).strip()
                if not raw_prompt:
                    continue
                prompts = [p.strip() for p in raw_prompt.split(",") if p.strip()]
                key = "|".join(p.lower() for p in prompts)
                mode = "text"
                pending_point_norm = None
                sam_initialized = False
                last_yoloe_box_px = None
                miss_count = 0

                model = get_yoloe()
                if current_yoloe_key != key:
                    pe = prompt_pe_cache.get(key)
                    if pe is None:
                        pe = model.get_text_pe(prompts)
                        prompt_pe_cache[key] = pe
                    model.set_classes(prompts, pe)
                    current_yoloe_key = key

                await ws.send_json({"type": "mode_set", "mode": "text", "prompt": raw_prompt})
                continue

            if cmd == "track_point":
                nx = float(msg.get("x", 500))
                ny = float(msg.get("y", 500))
                mode = "sam_memory"
                pending_point_norm = [nx, ny]
                sam_initialized = False
                last_center_norm = (nx, ny)
                miss_count = 0
                pred = get_sam_memory()
                reset_sam_memory(pred)
                await ws.send_json({"type": "mode_set", "mode": "sam_memory", "x": nx, "y": ny})
                continue

            if cmd == "lock_current":
                # Hand off the last YOLOE detection straight into SAM 2.1 Visual Memory Bank!
                if last_yoloe_box_px is None or last_frame_bgr is None:
                    await ws.send_json({"type": "error", "message": "No active YOLOE detection available to lock."})
                    continue
                pred = get_sam_memory()
                reset_sam_memory(pred)
                with torch.inference_mode():
                    pred(
                        source=last_frame_bgr,
                        bboxes=[last_yoloe_box_px],
                        obj_ids=[0],
                        update_memory=True,
                    )
                mode = "sam_memory"
                pending_point_norm = None
                sam_initialized = True
                miss_count = 0
                await ws.send_json({"type": "mode_set", "mode": "sam_memory", "lockedFromYoloe": True})
                continue

            if cmd == "frame":
                if mode == "idle":
                    await ws.send_json({"type": "result", "found": False, "mode": "idle"})
                    continue

                b64 = msg.get("jpeg", "")
                frame = decode_jpeg(b64)
                if frame is None:
                    await ws.send_json({"type": "result", "found": False, "error": "bad_jpeg"})
                    continue

                img_h, img_w = frame.shape[:2]
                t0 = time.perf_counter()

                if mode == "text":
                    model = get_yoloe()
                    with torch.inference_mode():
                        results = model.predict(
                            frame,
                            conf=0.16,
                            imgsz=480,
                            verbose=False,
                            device=DEVICE,
                        )
                    r = results[0]
                    best_idx = -1
                    best_score = -1.0
                    if r.boxes is not None and len(r.boxes) > 0:
                        confs = r.boxes.conf.detach().cpu().numpy()
                        xyxy = r.boxes.xyxy.detach().cpu().numpy()
                        for i, c in enumerate(confs):
                            bx = ((xyxy[i][0] + xyxy[i][2]) / 2.0) / img_w * 1000.0
                            by = ((xyxy[i][1] + xyxy[i][3]) / 2.0) / img_h * 1000.0
                            dist_penalty = np.hypot(bx - last_center_norm[0], by - last_center_norm[1]) / 3500.0
                            rank = float(c) - (dist_penalty if miss_count < 8 else 0.0)
                            if rank > best_score:
                                best_score = rank
                                best_idx = i

                    ms = int((time.perf_counter() - t0) * 1000)
                    if best_idx >= 0:
                        miss_count = 0
                        conf_val = float(r.boxes.conf[best_idx].item())
                        x1, y1, x2, y2 = r.boxes.xyxy[best_idx].detach().cpu().numpy().tolist()
                        last_yoloe_box_px = [x1, y1, x2, y2]
                        last_frame_bgr = frame

                        poly: list[list[int]] = []
                        cx = int(max(0, min(1000, round(((x1 + x2) / 2.0 / img_w) * 1000))))
                        cy = int(max(0, min(1000, round(((y1 + y2) / 2.0 / img_h) * 1000))))
                        nw = int(max(20, min(1000, round(((x2 - x1) / img_w) * 1000))))
                        nh = int(max(20, min(1000, round(((y2 - y1) / img_h) * 1000))))

                        if r.masks is not None and r.masks.data is not None and best_idx < len(r.masks.data):
                            m_raw = r.masks.data[best_idx].detach().cpu().numpy()
                            m_resized = cv2.resize(m_raw, (img_w, img_h), interpolation=cv2.INTER_NEAREST) > 0.5
                            poly, center_m, size_m = mask_to_norm_polygon(m_resized, img_w, img_h)
                            if poly:
                                cx, cy = center_m
                                nw, nh = size_m

                        last_center_norm = (float(cx), float(cy))
                        await ws.send_json(
                            {
                                "type": "result",
                                "mode": "text",
                                "found": True,
                                "x": cx,
                                "y": cy,
                                "width": nw,
                                "height": nh,
                                "polygon": poly,
                                "score": round(conf_val, 3),
                                "ms": ms,
                                "model": "YOLOE-11m-seg",
                            }
                        )
                    else:
                        miss_count += 1
                        await ws.send_json(
                            {
                                "type": "result",
                                "mode": "text",
                                "found": False,
                                "ms": ms,
                                "model": "YOLOE-11m-seg",
                            }
                        )

                elif mode == "sam_memory":
                    pred = get_sam_memory()
                    with torch.inference_mode():
                        if not sam_initialized:
                            if pending_point_norm is None:
                                results = []
                            else:
                                pt_x = (pending_point_norm[0] / 1000.0) * img_w
                                pt_y = (pending_point_norm[1] / 1000.0) * img_h
                                reset_sam_memory(pred)
                                results = pred(
                                    source=frame,
                                    points=[[pt_x, pt_y]],
                                    labels=[1],
                                    obj_ids=[0],
                                    update_memory=True,
                                )
                                sam_initialized = True
                        else:
                            # Pure Visual Memory Bank attention across the ENTIRE frame!
                            results = pred(source=frame)

                    ms = int((time.perf_counter() - t0) * 1000)
                    r = results[0] if results else None
                    if r is not None and r.masks is not None and len(r.masks.data) > 0:
                        m_raw = r.masks.data[0].detach().cpu().numpy()
                        m_resized = cv2.resize(m_raw, (img_w, img_h), interpolation=cv2.INTER_NEAREST) > 0.5
                        poly, [cx, cy], [nw, nh] = mask_to_norm_polygon(m_resized, img_w, img_h)
                        raw_conf = float(r.boxes.conf[0].item()) if r.boxes is not None and len(r.boxes) > 0 else 0.2
                        # Map SAM2's scaled logit score (0.08..0.32) to friendly 0..1 display score
                        display_score = round(min(1.0, raw_conf / 0.30), 3)
                        if poly and (nw * nh) < 750000:
                            last_center_norm = (float(cx), float(cy))
                            await ws.send_json(
                                {
                                    "type": "result",
                                    "mode": "sam_memory",
                                    "found": True,
                                    "x": cx,
                                    "y": cy,
                                    "width": nw,
                                    "height": nh,
                                    "polygon": poly,
                                    "score": display_score,
                                    "ms": ms,
                                    "model": "SAM2-Memory",
                                }
                            )
                            continue

                    await ws.send_json(
                        {
                            "type": "result",
                            "mode": "sam_memory",
                            "found": False,
                            "ms": ms,
                            "model": "SAM2-Memory",
                        }
                    )

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print("[VISION] WebSocket error:", exc)
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
