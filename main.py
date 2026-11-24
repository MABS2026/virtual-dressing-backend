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
# ELLIPSE CIRCUMFERENCE (RAMANUJAN II)
# ==========================================

def ellipse_circumference_cm(width_cm: float, depth_cm: float) -> float:
    a = max(width_cm / 2.0, 0.1)
    b = max(depth_cm / 2.0, 0.1)
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (
        1.0 + (3.0 * h) / (10.0 + math.sqrt(max(0.0, 4.0 - 3.0 * h)))
    )


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
# MEDIAPIPE POSE + GRABCUT (480PX MEMORY SAFE)
# ==========================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=0,
    min_detection_confidence=0.5,
)


def _resize_for_processing(img: np.ndarray, max_dim: int = 480) -> np.ndarray:
    h, w, _ = img.shape
    largest = max(h, w)
    if largest <= max_dim:
        return img
    scale = max_dim / float(largest)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


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
            1,
            cv2.GC_INIT_WITH_RECT
        )
        person_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)

        del mask, bgd_model, fgd_model

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
    safe_fallback = float(np.clip(fallback_px, min_px, max_px))
    if person_mask is None:
        return safe_fallback

    h, w = person_mask.shape
    y0 = max(0, y_pos - 3)
    y1 = min(h, y_pos + 4)
    if y0 >= y1:
        return safe_fallback

    band = person_mask[y0:y1, :]
    active = np.any(band > 0, axis=0).astype(np.uint8)

    x_min_lim = max(0, x_center_hint - max_half_span)
    x_max_lim = min(w, x_center_hint + max_half_span + 1)
    if x_min_lim >= x_max_lim:
        return safe_fallback

    sub = active[x_min_lim:x_max_lim]
    padded = np.concatenate(([0], sub, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    if len(starts) == 0:
        return safe_fallback

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
        return safe_fallback

    if best_span > safe_fallback * 1.40 or best_span < safe_fallback * 0.65:
        best_span = safe_fallback

    blended = 0.50 * best_span + 0.50 * safe_fallback
    return float(np.clip(blended, min_px, max_px))


def _measure_side_depth_cm(
    side_bytes: Optional[bytes],
    height_cm: float
) -> Optional[tuple[float, float]]:
    """Side-view image + grabCut se chest aur waist ki front-to-back depth (cm) nikalta hai."""
    if not side_bytes:
        return None
    try:
        nparr = np.frombuffer(side_bytes, np.uint8)
        raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del nparr
        if raw_img is None:
            return None

        img = _resize_for_processing(raw_img, max_dim=480)
        del raw_img

        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        del img_rgb

        if not res.pose_landmarks:
            del img
            return None

        lm = res.pose_landmarks.landmark
        nose = lm[mp_pose.PoseLandmark.NOSE]
        l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        l_ank = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
        r_ank = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

        nose_y = nose.y * h
        sh_y = ((l_sh.y + r_sh.y) / 2.0) * h
        hip_y = ((l_hip.y + r_hip.y) / 2.0) * h
        ank_y = ((l_ank.y + r_ank.y) / 2.0) * h

        top_y = max(0.0, nose_y - 0.11 * h)
        bot_y = min(float(h), ank_y + 0.03 * h)
        body_h_px = max(20.0, bot_y - top_y)
        cm_per_px = height_cm / body_h_px

        center_x = int((((l_sh.x + r_sh.x + l_hip.x + r_hip.x) / 4.0)) * w)
        x1 = max(0, center_x - int(0.18 * w))
        x2 = min(w, center_x + int(0.18 * w))

        side_mask = segment_body_grabcut(img, x1, int(top_y), x2, int(bot_y))
        del img

        if side_mask is None:
            return None

        chest_y = int(sh_y + 0.28 * (hip_y - sh_y))
        waist_y = int(sh_y + 0.65 * (hip_y - sh_y))

        chest_depth_px = measure_horizontal_width_px(
            person_mask=side_mask,
            y_pos=chest_y,
            fallback_px=22.0 / max(cm_per_px, 0.05),
            x_center_hint=center_x,
            max_half_span=max(15, int(0.18 * w)),
            min_px=10.0,
            max_px=float(w) * 0.45,
        )
        waist_depth_px = measure_horizontal_width_px(
            person_mask=side_mask,
            y_pos=waist_y,
            fallback_px=20.0 / max(cm_per_px, 0.05),
            x_center_hint=center_x,
            max_half_span=max(15, int(0.18 * w)),
            min_px=10.0,
            max_px=float(w) * 0.45,
        )
        del side_mask

        return (chest_depth_px * cm_per_px, waist_depth_px * cm_per_px)
    except Exception:
        return None


def extract_pose_metrics(
    image_bytes: bytes,
    height_cm: float,
    weight_kg: float,
    side_bytes: Optional[bytes] = None,
):
    nparr = np.frombuffer(image_bytes, np.uint8)
    raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    del nparr

    if raw_img is None:
        return None, 0.0, "Invalid image file"

    img = _resize_for_processing(raw_img, max_dim=480)
    del raw_img

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    del img_rgb

    if not results.pose_landmarks:
        del img
        gc.collect()
        return None, 0.0, (
            "Body not clearly detected. "
            "Please stand straight and upload a full-body photo."
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
            del img
            gc.collect()
            return None, 0.0, (
                "Some body parts aren't clearly visible. "
                "Please keep your full body (head to toe) inside the frame."
            )

    avg_vis = sum(pt.visibility for pt in key_points) / len(key_points)

    nose_y = nose.y * h
    shoulder_y = ((left_shoulder.y + right_shoulder.y) / 2.0) * h
    hip_y = ((left_hip.y + right_hip.y) / 2.0) * h
    ankle_y = ((left_ankle.y + right_ankle.y) / 2.0) * h

    top_head_y = max(0.0, nose_y - 0.11 * h)
    bottom_foot_y = min(float(h), ankle_y + 0.03 * h)
    body_height_px = max(20.0, bottom_foot_y - top_head_y)
    cm_per_px = height_cm / body_height_px

    shoulder_width_px = abs(left_shoulder.x - right_shoulder.x) * w
    hip_width_px = abs(left_hip.x - right_hip.x) * w

    x1 = max(
        0,
        int(min(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * w - 0.10 * w),
    )
    x2 = min(
        w,
        int(max(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * w + 0.10 * w),
    )
    seg_top_y = max(0, int(top_head_y))
    seg_bot_y = min(h, int(bottom_foot_y))

    person_mask = segment_body_grabcut(img, x1, seg_top_y, x2, seg_bot_y)
    x_center_hint = int(((left_hip.x + right_hip.x) / 2.0) * w)
    max_half_span = max(20, int(0.28 * w))

    chest_y = int(shoulder_y + 0.26 * (hip_y - shoulder_y))
    waist_y = int(shoulder_y + 0.64 * (hip_y - shoulder_y))

    height_m = max(height_cm / 100.0, 1.2)
    bmi = weight_kg / (height_m * height_m)

    shoulder_cal = float(np.clip(1.14 + (bmi - 20.0) * 0.007, 1.08, 1.25))
    hip_cal = float(np.clip(1.22 + (bmi - 20.0) * 0.009, 1.15, 1.36))
    taper = float(np.clip(0.84 + (bmi - 20.0) * 0.011, 0.76, 0.98))
    fallback_waist_px = (
        (hip_width_px * hip_cal * taper)
        if hip_width_px > 0
        else (shoulder_width_px * 0.80)
    )

    measured_shoulder_px = measure_horizontal_width_px(
        person_mask=person_mask,
        y_pos=int(shoulder_y),
        fallback_px=(shoulder_width_px if shoulder_width_px > 0 else 80.0) * shoulder_cal,
        x_center_hint=x_center_hint,
        max_half_span=max_half_span,
        min_px=20.0,
        max_px=float(w),
    )
    measured_chest_px = measure_horizontal_width_px(
        person_mask=person_mask,
        y_pos=chest_y,
        fallback_px=measured_shoulder_px * 0.88,
        x_center_hint=x_center_hint,
        max_half_span=max_half_span,
        min_px=18.0,
        max_px=float(w),
    )
    measured_waist_px = measure_horizontal_width_px(
        person_mask=person_mask,
        y_pos=waist_y,
        fallback_px=(fallback_waist_px if fallback_waist_px > 0 else 65.0),
        x_center_hint=x_center_hint,
        max_half_span=max_half_span,
        min_px=16.0,
        max_px=float(w),
    )
    measured_hip_px = measure_horizontal_width_px(
        person_mask=person_mask,
        y_pos=int(hip_y),
        fallback_px=(hip_width_px if hip_width_px > 0 else 75.0) * hip_cal,
        x_center_hint=x_center_hint,
        max_half_span=max_half_span,
        min_px=20.0,
        max_px=float(w),
    )

    if person_mask is not None:
        del person_mask
    del img
    gc.collect()

    raw_shoulder_cm = measured_shoulder_px * cm_per_px
    raw_chest_width_cm = measured_chest_px * cm_per_px
    raw_waist_width_cm = measured_waist_px * cm_per_px
    raw_hip_width_cm = measured_hip_px * cm_per_px

    # Torso depth from BMI + optional Side View GrabCut measurement
    depth_scale = float(np.clip(0.68 + (bmi - 20.0) * 0.012, 0.62, 0.86))
    est_chest_depth_cm = raw_chest_width_cm * depth_scale
    est_waist_depth_cm = raw_waist_width_cm * (depth_scale * 0.98)
    est_hip_depth_cm = raw_hip_width_cm * depth_scale

    side_depths = _measure_side_depth_cm(side_bytes, height_cm)
    has_side_view = side_depths is not None
    if side_depths is not None:
        side_chest_d, side_waist_d = side_depths
        side_chest_d = float(np.clip(side_chest_d, est_chest_depth_cm * 0.78, est_chest_depth_cm * 1.22))
        side_waist_d = float(np.clip(side_waist_d, est_waist_depth_cm * 0.78, est_waist_depth_cm * 1.22))
        est_chest_depth_cm = 0.55 * est_chest_depth_cm + 0.45 * side_chest_d
        est_waist_depth_cm = 0.55 * est_waist_depth_cm + 0.45 * side_waist_d

    raw_chest_cm = ellipse_circumference_cm(raw_chest_width_cm, est_chest_depth_cm)
    raw_waist_cm = ellipse_circumference_cm(raw_waist_width_cm, est_waist_depth_cm)
    raw_hip_cm = ellipse_circumference_cm(raw_hip_width_cm, est_hip_depth_cm)

    # Calibrated Anthropometric Regression (no 70cm floor lock; responsive to height & weight)
    anthro_chest_cm = 0.24 * height_cm + 0.62 * weight_kg + 13.5
    anthro_waist_cm = 0.14 * height_cm + 0.68 * weight_kg + 12.2
    anthro_hip_cm = 0.22 * height_cm + 0.60 * weight_kg + 22.5
    anthro_shoulder_cm = 0.17 * height_cm + 0.21 * weight_kg + 3.2

    # Bound loose clothing / camera perspective distortion within +/- 6.5 cm of anthropometric baseline
    raw_chest_cm = float(np.clip(raw_chest_cm, anthro_chest_cm - 6.5, anthro_chest_cm + 6.5))
    raw_waist_cm = float(np.clip(raw_waist_cm, anthro_waist_cm - 6.0, anthro_waist_cm + 6.0))
    raw_hip_cm = float(np.clip(raw_hip_cm, anthro_hip_cm - 6.5, anthro_hip_cm + 6.5))
    raw_shoulder_cm = float(np.clip(raw_shoulder_cm, anthro_shoulder_cm - 3.5, anthro_shoulder_cm + 3.5))

    # Fuse GrabCut + Pose circumference (35%) with calibrated anthropometric model (65%)
    chest_cm = float(np.clip(0.35 * raw_chest_cm + 0.65 * anthro_chest_cm, 72.0, 148.0))
    waist_cm = float(np.clip(0.35 * raw_waist_cm + 0.65 * anthro_waist_cm, 58.0, 140.0))
    hip_cm = float(np.clip(0.35 * raw_hip_cm + 0.65 * anthro_hip_cm, 74.0, 148.0))
    shoulder_cm = float(np.clip(0.40 * raw_shoulder_cm + 0.60 * anthro_shoulder_cm, 34.0, 62.0))

    shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
    hip_tilt = abs(left_hip.y - right_hip.y)
    tilt_penalty = min(8.0, (shoulder_tilt + hip_tilt) * 85.0)

    chest_diff_in = abs((raw_chest_cm - anthro_chest_cm) / 2.54)
    waist_diff_in = abs((raw_waist_cm - anthro_waist_cm) / 2.54)
    consistency_penalty = min(10.0, (chest_diff_in + waist_diff_in) * 1.2)

    side_bonus = 2.5 if has_side_view else -1.5
    raw_conf = (avg_vis * 95.0) - tilt_penalty - consistency_penalty + side_bonus
    confidence_pct = round(float(np.clip(raw_conf, 72.0, 97.5)), 1)

    return {
        "shoulder_cm": round(shoulder_cm, 1),
        "chest_cm": round(chest_cm, 1),
        "waist_cm": round(waist_cm, 1),
        "hip_cm": round(hip_cm, 1),
        "chest_inches": round(chest_cm / 2.54, 1),
        "waist_inches": round(waist_cm / 2.54, 1),
        "hip_inches": round(hip_cm / 2.54, 1),
        "shoulder_inches": round(shoulder_cm / 2.54, 1),
    }, confidence_pct, None


def _pant_size_from_waist_inches(waist_in: float) -> str:
    if waist_in < 28.0:
        return "XS"
    elif waist_in <= 29.5:
        return "Small"
    elif waist_in <= 32.5:
        return "M"
    elif waist_in <= 35.5:
        return "Large"
    elif waist_in <= 38.5:
        return "L"
    elif waist_in <= 42.5:
        return "XL"
    else:
        return "XXL"


def _shirt_size_from_chest_inches(chest_in: float) -> str:
    if chest_in < 34.0:
        return "XS"
    elif chest_in <= 36.5:
        return "S"
    elif chest_in <= 39.5:
        return "M"
    elif chest_in <= 42.5:
        return "L"
    elif chest_in <= 46.0:
        return "XL"
    else:
        return "XXL"


@app.post("/api/predict-size")
async def predict_size(
    image: UploadFile = File(...),
    side_image: Optional[UploadFile] = File(None),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
):
    try:
        if height_cm < 120 or height_cm > 230:
            raise HTTPException(
                status_code=400,
                detail="Height 120 cm se 230 cm ke darmiyan honi chahiye."
            )
        if weight_kg < 30 or weight_kg > 200:
            raise HTTPException(
                status_code=400,
                detail="Weight 30 kg se 200 kg ke darmiyan hona chahiye."
            )

        front_bytes = await image.read()
        if not front_bytes:
            raise HTTPException(status_code=400, detail="Front photo required hai.")

        side_bytes = None
        if side_image is not None:
            raw_side = await side_image.read()
            if raw_side:
                side_bytes = raw_side

        metrics, confidence_pct, err = extract_pose_metrics(
            front_bytes,
            height_cm,
            weight_kg,
            side_bytes=side_bytes,
        )
        del front_bytes, side_bytes

        if err or metrics is None:
            raise HTTPException(status_code=400, detail=err or "Pose detection failed.")

        chest_in = metrics["chest_inches"]
        waist_in = metrics["waist_inches"]

        shirt_size = _shirt_size_from_chest_inches(chest_in)
        pant_size = _pant_size_from_waist_inches(waist_in)

        return JSONResponse(
            content={
                "status": "success",
                "height_cm": round(height_cm, 1),
                "weight_kg": round(weight_kg, 1),
                "confidence": confidence_pct,
                "confidence_percentage": confidence_pct,
                "measurements": metrics,
                "recommended_shirt_size": shirt_size,
                "recommended_pant_size": pant_size,
                "recommended_size": shirt_size,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()


@app.post("/api/skin-tone-analysis")
async def skin_tone_analysis(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        if not img_bytes:
            raise HTTPException(status_code=400, detail="Image file required hai.")

        nparr = np.frombuffer(img_bytes, np.uint8)
        raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del nparr, img_bytes

        if raw_img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        img = _resize_for_processing(raw_img, max_dim=480)
        del raw_img

        h, w, _ = img.shape
        crop = img[int(h * 0.18):int(h * 0.62), int(w * 0.28):int(w * 0.72)]
        if crop.size == 0:
            crop = img

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_v = float(hsv[:, :, 2].mean())

        mean_bgr = crop.mean(axis=(0, 1))
        b, g, r = float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])
        del hsv, crop, img

        if mean_v >= 165:
            tone_band = "Fair to Light"
        elif mean_v >= 115:
            tone_band = "Medium to Olive"
        else:
            tone_band = "Deep / Rich"

        if (r - b) > 12:
            undertone = f"Warm Undertone ({tone_band})"
            colors = [
                {
                    "name": "Warm Olive & Moss Green",
                    "hex": "#5A6B3C",
                    "desc": "Warm undertones ke sath natural harmony banata hai aur smart casual look deta hai.",
                },
                {
                    "name": "Terracotta & Rust Brown",
                    "hex": "#B8502D",
                    "desc": "Golden aur warm skin tones ko rich, confident depth deta hai.",
                },
                {
                    "name": "Warm Ivory & Camel",
                    "hex": "#E3C598",
                    "desc": "Harsh contrast ke baghair clean aur refined appearance deta hai.",
                },
                {
                    "name": "Deep Burgundy / Wine",
                    "hex": "#6D213C",
                    "desc": "Evening aur formal wear ke liye balanced aur classy option hai.",
                },
            ]
        elif (b - r) > 8:
            undertone = f"Cool Undertone ({tone_band})"
            colors = [
                {
                    "name": "Classic Navy & Slate Blue",
                    "hex": "#1E3A5F",
                    "desc": "Cool undertones ke liye crisp, formal aur trustworthy palette hai.",
                },
                {
                    "name": "Emerald & Pine Teal",
                    "hex": "#15616D",
                    "desc": "Skin tone ki clarity ko ubharta hai aur fresh look deta hai.",
                },
                {
                    "name": "Crisp White & Cool Grey",
                    "hex": "#D8DEE9",
                    "desc": "Clean contrast ke sath modern aur sharp styling deta hai.",
                },
                {
                    "name": "Plum & Berry Slate",
                    "hex": "#5E3A5A",
                    "desc": "Smart semi-formal outfits ke liye balanced accent shade hai.",
                },
            ]
        else:
            undertone = f"Neutral Undertone ({tone_band})"
            colors = [
                {
                    "name": "Charcoal & Graphite Grey",
                    "hex": "#363945",
                    "desc": "Neutral skin tones par versatile, clean aur formal look deta hai.",
                },
                {
                    "name": "Dusty Teal & Sage",
                    "hex": "#3B6E6C",
                    "desc": "Balanced undertones ke sath calm aur polished contrast banata hai.",
                },
                {
                    "name": "Soft Rose & Mauve",
                    "hex": "#A84A6A",
                    "desc": "App ke signature aesthetic ke sath subtle aur stylish match hai.",
                },
                {
                    "name": "Warm Sand & Taupe",
                    "hex": "#9C846E",
                    "desc": "Everyday wear ke liye effortless aur cohesive tone deta hai.",
                },
            ]

        return JSONResponse(
            content={
                "status": "success",
                "detected_undertone": undertone,
                "recommended_colors": colors,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
