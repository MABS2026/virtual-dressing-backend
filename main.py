import os
import io
import gc
import time
import math
import base64
import traceback
from typing import Optional

import nest_asyncio
import uvicorn
import requests
import cv2
import numpy as np
import mediapipe as mp

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse


nest_asyncio.apply()
app = FastAPI()

# ==========================================
# FASHN AI TRY-ON MAX CONFIG
# ==========================================

FASHN_API_KEY = os.environ.get(
    "FASHN_API_KEY",
    "fa-LKYXdavvGOc4-RaLBOccIDvo6lcb43OCd3nHj",
).strip()
FASHN_RUN_URL = "https://api.fashn.ai/v1/run"
FASHN_STATUS_BASE = "https://api.fashn.ai/v1/status"
MAX_IMAGE_BYTES = 30 * 1024 * 1024  # 30 MiB limit per FASHN docs


def _to_data_uri(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    b64_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64_str}"


def _run_fashn_tryon(
    user_bytes: bytes,
    garment_bytes: bytes,
    prompt: Optional[str] = None,
    resolution: str = "1k",
    generation_mode: str = "fast",
    output_format: str = "jpeg",
) -> tuple[bytes, str]:
    if not FASHN_API_KEY:
        raise RuntimeError("FASHN_API_KEY environment variable set nahi hai.")

    headers = {
        "Authorization": f"Bearer {FASHN_API_KEY}",
        "Content-Type": "application/json",
    }

    inputs = {
        "model_image": _to_data_uri(user_bytes, "image/jpeg"),
        "product_image": _to_data_uri(garment_bytes, "image/jpeg"),
        "resolution": resolution,
        "generation_mode": generation_mode,
        "output_format": output_format,
    }
    if prompt and prompt.strip():
        inputs["prompt"] = prompt.strip()

    payload = {
        "model_name": "tryon-max",
        "inputs": inputs,
    }

    run_resp = requests.post(FASHN_RUN_URL, headers=headers, json=payload, timeout=60)
    if run_resp.status_code not in (200, 201):
        raise RuntimeError(
            f"FASHN /v1/run error ({run_resp.status_code}): {run_resp.text}"
        )

    run_data = run_resp.json()
    if run_data.get("error"):
        raise RuntimeError(f"FASHN run error: {run_data.get('error')}")

    prediction_id = run_data.get("id")
    if not prediction_id:
        raise RuntimeError(f"Prediction ID nahi mila: {run_data}")

    max_attempts = 50
    poll_interval = 3.0

    for _ in range(max_attempts):
        time.sleep(poll_interval)
        status_resp = requests.get(
            f"{FASHN_STATUS_BASE}/{prediction_id}",
            headers={"Authorization": f"Bearer {FASHN_API_KEY}"},
            timeout=30,
        )
        if status_resp.status_code != 200:
            raise RuntimeError(
                f"FASHN status error ({status_resp.status_code}): {status_resp.text}"
            )

        status_data = status_resp.json()
        status = (status_data.get("status") or "").lower()

        if status == "completed":
            output = status_data.get("output") or []
            if not output:
                raise RuntimeError(
                    f"Completed response mein output empty hai: {status_data}"
                )

            img_url = output[0]
            img_resp = requests.get(img_url, timeout=60)
            if img_resp.status_code != 200:
                raise RuntimeError(
                    f"Output image download fail ({img_resp.status_code})"
                )

            media_type = (
                "image/jpeg" if output_format.lower() == "jpeg" else "image/png"
            )
            return img_resp.content, media_type

        if status in ("failed", "error", "canceled", "cancelled"):
            err_info = status_data.get("error") or status_data
            raise RuntimeError(f"FASHN job fail ho gaya ({status}): {err_info}")

    raise TimeoutError("FASHN try-on polling time out ho gaya.")


# ==========================================
# ELLIPSE CIRCUMFERENCE
# ==========================================

def ellipse_circumference_cm(
    width_cm: float,
    depth_cm: float
) -> float:
    a = max(width_cm / 2.0, 0.1)
    b = max(depth_cm / 2.0, 0.1)

    h = ((a - b) ** 2) / ((a + b) ** 2)

    circumference = math.pi * (a + b) * (
        1.0 +
        (3.0 * h) /
        (10.0 + math.sqrt(max(0.0, 4.0 - 3.0 * h)))
    )

    return circumference


# ==========================================
# HOME ROUTE
# ==========================================

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Virtual Dressing Backend is running successfully!"
    }


# ==========================================
# ROUTE 1: VIRTUAL TRY-ON (FASHN TRY-ON MAX)
# ==========================================

@app.post("/api/try-on")
async def try_on(
    user: UploadFile = File(...),
    garment: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    resolution: str = Form("1k"),
    generation_mode: str = Form("fast"),
    output_format: str = Form("jpeg"),
):
    try:
        user_bytes = await user.read()
        garment_bytes = await garment.read()

        if not user_bytes or not garment_bytes:
            raise HTTPException(
                status_code=400,
                detail="User aur garment dono images zaroori hain."
            )

        if len(user_bytes) > MAX_IMAGE_BYTES or len(garment_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=400,
                detail="Har image ka size 30 MiB se kam hona chahiye."
            )

        valid_res = {"1k", "2k", "4k"}
        valid_modes = {"fast", "balanced", "quality"}
        valid_formats = {"jpeg", "png"}

        res_clean = resolution.lower() if resolution.lower() in valid_res else "1k"
        mode_clean = generation_mode.lower() if generation_mode.lower() in valid_modes else "fast"
        fmt_clean = output_format.lower() if output_format.lower() in valid_formats else "jpeg"

        result_bytes, media_type = _run_fashn_tryon(
            user_bytes=user_bytes,
            garment_bytes=garment_bytes,
            prompt=prompt,
            resolution=res_clean,
            generation_mode=mode_clean,
            output_format=fmt_clean,
        )

        del user_bytes, garment_bytes
        gc.collect()

        return StreamingResponse(
            io.BytesIO(result_bytes),
            media_type=media_type
        )

    except HTTPException:
        raise
    except Exception as e:
        error_details = traceback.format_exc()
        print("TRY-ON ERROR:\n", error_details)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================
# MEDIAPIPE POSE + SEGMENTATION
# ==========================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
)


def segment_body_grabcut(
    img: np.ndarray,
    x1: int,
    top_y: int,
    x2: int,
    bottom_y: int
) -> Optional[np.ndarray]:
    h, w, _ = img.shape
    rect_w = x2 - x1
    rect_h = bottom_y - top_y

    if rect_w < 20 or rect_h < 50:
        return None

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (x1, top_y, rect_w, rect_h)

    try:
        cv2.grabCut(
            img,
            mask,
            rect,
            bgd_model,
            fgd_model,
            2,
            cv2.GC_INIT_WITH_RECT
        )
        person_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)

        if np.count_nonzero(person_mask) < 100:
            return None

        return person_mask
    except Exception:
        return None


def measure_horizontal_width_px(
    person_mask: Optional[np.ndarray],
    y_pos: int,
    fallback_px: float,
    x_center_hint: int,
    max_half_span: int,
    min_px: float,
    max_px: float
) -> float:
    if person_mask is None:
        return float(np.clip(fallback_px, min_px, max_px))

    h, w = person_mask.shape
    y0 = max(0, y_pos - 4)
    y1 = min(h, y_pos + 5)
    if y0 >= y1:
        return float(np.clip(fallback_px, min_px, max_px))

    band = person_mask[y0:y1, :]
    active = np.any(band > 0, axis=0).astype(np.uint8)

    x_min_lim = max(0, x_center_hint - max_half_span)
    x_max_lim = min(w, x_center_hint + max_half_span + 1)
    if x_min_lim >= x_max_lim:
        return float(np.clip(fallback_px, min_px, max_px))

    sub = active[x_min_lim:x_max_lim]
    padded = np.concatenate(([0], sub, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    if len(starts) == 0:
        return float(np.clip(fallback_px, min_px, max_px))

    rel_center = float(x_center_hint - x_min_lim)
    best_span = None
    best_dist = float("inf")
    for s, e in zip(starts, ends):
        span = float(e - s)
        if span < 3:
            continue
        seg_center = 0.5 * (s + e)
        dist = abs(seg_center - rel_center)
        if dist < best_dist:
            best_dist = dist
            best_span = span

    if best_span is None:
        return float(np.clip(fallback_px, min_px, max_px))

    blended = 0.70 * best_span + 0.30 * fallback_px
    return float(np.clip(blended, min_px, max_px))


def extract_pose_metrics(
    image_bytes: bytes,
    height_cm: float,
    weight_kg: float,
):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None, 0.0, "Invalid image file"

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        return None, 0.0, (
            "Body clearly detect nahi hui. "
            "Full body photo seedhay kharay ho kar upload karein."
        )

    landmarks = results.pose_landmarks.landmark

    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    key_points = [
        nose,
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip,
        left_ankle,
        right_ankle,
    ]

    for pt in key_points:
        if pt.visibility < 0.35:
            return None, 0.0, (
                "Kuch body parts clearly nazar nahi aa rahe. "
                "Poori body (sar se paon tak) frame mein rakhein."
            )

    avg_vis = sum(pt.visibility for pt in key_points) / len(key_points)
    confidence_pct = round(float(np.clip(avg_vis * 100.0, 10.0, 99.5)), 1)

    nose_y = nose.y * h
    shoulder_y = ((left_shoulder.y + right_shoulder.y) / 2.0) * h
    hip_y = ((left_hip.y + right_hip.y) / 2.0) * h
    ankle_y = ((left_ankle.y + right_ankle.y) / 2.0) * h

    # Full height calibration from
