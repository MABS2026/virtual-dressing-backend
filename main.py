import os
import io
import math
import traceback
from typing import Optional

import nest_asyncio
import uvicorn
import cv2
import numpy as np
import mediapipe as mp

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from gradio_client import Client, handle_file


# Hugging Face Token
if os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")


nest_asyncio.apply()
app = FastAPI()

client = None


# ==========================================
# IDM-VTON CLIENT
# ==========================================

def get_gradio_client():
    global client

    if client is None:
        try:
            print("Connecting to IDM-VTON Model...")
            client = Client("yisol/IDM-VTON")
            print("Connected Successfully!")

        except Exception as e:
            print("Failed to connect:", str(e))

            raise HTTPException(
                status_code=500,
                detail="AI Model service is currently unavailable."
            )

    return client


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
# ROUTE 1: VIRTUAL TRY-ON
# ==========================================

@app.post("/api/try-on")
async def try_on(
    user: UploadFile = File(...),
    garment: UploadFile = File(...)
):

    try:

        user_path = "user.jpg"
        garment_path = "garment.jpg"

        with open(user_path, "wb") as f:
            f.write(await user.read())

        with open(garment_path, "wb") as f:
            f.write(await garment.read())

        print("Images received, sending to IDM-VTON...")

        vton_client = get_gradio_client()

        result = vton_client.predict(
            dict={
                "background": handle_file(user_path),
                "layers": [],
                "composite": None,
            },
            garm_img=handle_file(garment_path),
            garment_des="a black cotton crewneck t-shirt with short sleeves",
            is_checked=True,
            denoise_steps=40,
            seed=42,
            api_name="/tryon",
        )

        with open(result[0], "rb") as f:

            print("Virtual Try-On Successful!")

            return StreamingResponse(
                io.BytesIO(f.read()),
                media_type="image/jpeg"
            )

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
        # Only definite foreground to avoid background/arm bleed
        person_mask = np.where(
            mask == cv2.GC_FGD,
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
    y0 = max(0, y_pos - 3)
    y1 = min(h, y_pos + 4)
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

    # Pick contiguous foreground segment closest to torso center
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

    # Blend 60% measured central span + 40% pose-derived fallback for stability
    blended = 0.60 * best_span + 0.40 * fallback_px
    return float(np.clip(blended, min_px, max_px))


def extract_pose_metrics(
    image_bytes: bytes,
    height_cm: float
):

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None, "Invalid image file"

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        return None, (
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
        right_ankle
    ]

    avg_vis = sum(p.visibility for p in key_points) / len(key_points)
    if avg_vis < 0.45:
        return None, (
            "Body visibility low hai. "
            "Clear full-body photo upload karein."
        )

    nose_y = int(nose.y * h)
    shoulder_y = int(((left_shoulder.y + right_shoulder.y) / 2.0) * h)
    hip_y = int(((left_hip.y + right_hip.y) / 2.0) * h)
    ankle_y = int(((left_ankle.y + right_ankle.y) / 2.0) * h)

    shoulder_width_px = abs(left_shoulder.x - right_shoulder.x) * w
    hip_width_px = abs(left_hip.x - right_hip.x) * w

    if hip_y <= shoulder_y or ankle_y <= nose_y:
        return None, (
            "Invalid body position. "
            "Seedha kharay ho kar photo upload karein."
        )

    torso_pixels = max(hip_y - shoulder_y, 1)

    # Tighter bounding box around body to reduce background inclusion
    body_width = max(shoulder_width_px, hip_width_px) * 1.45
    center_x = int(
        (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4.0 * w
    )
    top_y = max(0, int(nose_y - (ankle_y - nose_y) * 0.12))
    bottom_y = min(h - 1, int(ankle_y + (ankle_y - nose_y) * 0.04))

    x1 = max(0, int(center_x - body_width / 2.0))
    x2 = min(w - 1, int(center_x + body_width / 2.0))

    if x2 - x1 < 20 or bottom_y - top_y < 50:
        return None, "Full body photo properly visible nahi hai."

    person_mask = segment_body_grabcut(img, x1, top_y, x2, bottom_y)

    # Segmentation-based vertical body height if mask spans cleanly, else fallback
    if person_mask is not None:
        ys, _ = np.where(person_mask > 0)
        seg_h = float(int(ys.max()) - int(ys.min())) if len(ys) > 50 else 0.0
        pose_est_h = max((ankle_y - nose_y) / 0.84, 1.0)
        # Ensure mask height is plausible relative to pose span
        if seg_h > pose_est_h * 0.65 and seg_h < pose_est_h * 1.35:
            pixel_height = seg_h
        else:
            pixel_height = pose_est_h
    else:
        pixel_height = max((ankle_y - nose_y) / 0.84, 1.0)

    cm_per_pixel = height_cm / pixel_height

    chest_y = int(np.clip(shoulder_y + torso_pixels * 0.28, 0, h - 1))
    waist_y = int(np.clip(shoulder_y + torso_pixels * 0.65, 0, h - 1))

    # Keep horizontal search tight to torso so arms don't inflate width/depth
    sh_half_span = max(int(shoulder_width_px * 0.65), 15)
    hip_half_span = max(int(hip_width_px * 0.72), 15)

    shoulder_fallback = shoulder_width_px * 1.15
    chest_fallback = shoulder_width_px * 1.06
    waist_fallback = hip_width_px * 1.05

    shoulder_px = measure_horizontal_width_px(
        person_mask,
        shoulder_y,
        shoulder_fallback,
        center_x,
        sh_half_span,
        min_px=shoulder_width_px * 0.90,
        max_px=shoulder_width_px * 1.30,
    )
    chest_px = measure_horizontal_width_px(
        person_mask,
        chest_y,
        chest_fallback,
        center_x,
        sh_half_span,
        min_px=shoulder_width_px * 0.90,
        max_px=shoulder_width_px * 1.22,
    )
    waist_px = measure_horizontal_width_px(
        person_mask,
        waist_y,
        waist_fallback,
        center_x,
        hip_half_span,
        min_px=hip_width_px * 0.80,
        max_px=hip_width_px * 1.25,
    )

    torso_len_cm = max(torso_pixels * cm_per_pixel, 1.0)

    return {
        "cm_per_pixel": cm_per_pixel,
        "shoulder_px": shoulder_px,
        "chest_px": chest_px,
        "waist_px": waist_px,
        "torso_len_cm": torso_len_cm,
        "avg_vis": avg_vis,
    }, None


# ==========================================
# ROUTE 2: AI BODY SIZE PREDICTION
# ==========================================

@app.post("/api/predict-size")
async def predict_size(
    image: UploadFile = File(...),
    side_image: Optional[UploadFile] = File(None),
    height_cm: float = Form(...),
    weight_kg: Optional[float] = Form(None),
    gender: str = Form("male"),
):
    try:
        front_bytes = await image.read()
        front_data, front_err = extract_pose_metrics(front_bytes, height_cm)

        if front_err:
            if front_err == "Invalid image file":
                return JSONResponse(status_code=400, content={"error": front_err})
            return {"status": "error", "message": front_err}

        cm_per_pixel = front_data["cm_per_pixel"]
        torso_len_cm = front_data["torso_len_cm"]
        avg_vis = front_data["avg_vis"]

        outer_shoulder_cm = max(front_data["shoulder_px"] * cm_per_pixel, 25.0)
        chest_width_cm = max(front_data["chest_px"] * cm_per_pixel, 22.0)
        waist_width_cm = max(front_data["waist_px"] * cm_per_pixel, 20.0)

        used_dual_photo = False
        side_note = "Single-photo proportional estimation"

        if side_image is not None and side_image.filename:
            side_bytes = await side_image.read()
            if side_bytes:
                side_data, side_err = extract_pose_metrics(side_bytes, height_cm)
                if side_data is not None:
                    side_chest_cm = side_data["chest_px"] * side_data["cm_per_pixel"]
                    side_waist_cm = side_data["waist_px"] * side_data["cm_per_pixel"]
                    ref_torso_cm = 0.5 * (torso_len_cm + side_data["torso_len_cm"])

                    # Blend side mask width with width-proportional depth to avoid arm inflation
                    chest_depth_cm = float(
                        np.clip(
                            0.50 * side_chest_cm + 0.50 * (chest_width_cm * 0.64),
                            15.0,
                            32.0
                        )
                    )
                    waist_depth_cm = float(
                        np.clip(
                            0.50 * side_waist_cm + 0.50 * (waist_width_cm * 0.65),
                            14.0,
                            32.0
                        )
                    )

                    chest_circ_cm = ellipse_circumference_cm(
                        chest_width_cm, chest_depth_cm
                    )
                    waist_circ_cm = ellipse_circumference_cm(
                        waist_width_cm, waist_depth_cm
                    )

                    chest_in = round(chest_circ_cm / 2.54, 1)
                    waist_in = round(waist_circ_cm / 2.54, 1)

                    avg_vis = (avg_vis + side_data["avg_vis"]) / 2.0
                    used_dual_photo = True
                    side_note = "Dual-photo (Front + Side) GrabCut + Ellipse estimation"
                else:
                    chest_in = round((chest_width_cm * 2.35) / 2.54, 1)
                    waist_in = round((waist_width_cm * 2.30) / 2.54, 1)
            else:
                chest_in = round((chest_width_cm * 2.35) / 2.54, 1)
                waist_in = round((waist_width_cm * 2.30) / 2.54, 1)
        else:
            chest_in = round((chest_width_cm * 2.35) / 2.54, 1)
            waist_in = round((waist_width_cm * 2.30) / 2.54, 1)

        if waist_in < 26.0:
            waist_in = round(max(28.0, chest_in * 0.80), 1)

        shoulder_in = round(outer_shoulder_cm / 2.54, 1)

        bmi_val = None
        if weight_kg is not None and weight_kg > 0 and height_cm > 0:
            height_m = height_cm / 100.0
            bmi_val = round(weight_kg / (height_m * height_m), 1)

        base_conf = 82.0 + (avg_vis * 9.0) + (1.5 if used_dual_photo else 0.0)
        accuracy_percentage = round(min(max(base_conf, 82.0), 93.0), 1)

        chest_waist_diff = chest_in - waist_in
        if chest_waist_diff >= 8.0:
            body_type = "Athletic / V-Taper Physique"
        elif waist_in >= chest_in - 1.5:
            body_type = "Comfort / Relaxed Fit"
        else:
            body_type = "Standard / Balanced Fit"

        if chest_in < 36:
            recommended_size = "S (Small)"
        elif 36 <= chest_in < 39.5:
            recommended_size = "M (Medium)"
        elif 39.5 <= chest_in < 43:
            recommended_size = "L (Large)"
        elif 43 <= chest_in < 47:
            recommended_size = "XL (Extra Large)"
        else:
            recommended_size = "XXL (Double Extra Large)"

        if waist_in < 28.0:
            pant_size = 'XS (Under 28" Waist)'
        elif waist_in < 30.0:
            pant_size = 'S (28-29" Waist)'
        elif waist_in < 33.0:
            pant_size = 'M (30-32" Waist)'
        elif waist_in < 36.0:
            pant_size = 'L (33-35" Waist)'
        elif waist_in < 39.0:
            pant_size = 'XL (36-38" Waist)'
        elif waist_in <= 42.5:
            pant_size = 'XXL (39-42" Waist)'
        else:
            pant_size = '3XL (42"+ Waist)'

        measurements_payload = {
            "height_cm": height_cm,
            "shoulder_inches": shoulder_in,
            "chest_inches": chest_in,
            "waist_inches": waist_in,
        }
        if weight_kg is not None and weight_kg > 0:
            measurements_payload["weight_kg"] = round(weight_kg, 1)
        if bmi_val is not None:
            measurements_payload["bmi"] = bmi_val

        return {
            "status": "success",
            "body_type": body_type,
            "recommended_size": recommended_size,
            "recommended_shirt_size": recommended_size,
            "recommended_pant_size": pant_size,
            "accuracy_percentage": accuracy_percentage,
            "dual_photo_used": used_dual_photo,
            "measurements": measurements_payload,
            "confidence": f"Moderate-High ({side_note})",
        }

    except Exception as e:
        error_details = traceback.format_exc()
        print("PREDICT SIZE ERROR:\n", error_details)
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ROUTE 3: AI SKIN TONE & COLOR ANALYSIS
# ==========================================

@app.post("/api/skin-tone-analysis")
async def skin_tone_analysis(file: UploadFile = File(...)):

    try:

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        h, w, _ = img.shape
        center_crop = img[int(h * 0.3) : int(h * 0.6), int(w * 0.3) : int(w * 0.6)]

        avg_color_per_row = np.average(center_crop, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        b, g, r = avg_color

        if r > g and r > b:
            tone = "Warm Undertone (Medium/Olive Complexion)"
            recommended_colors = [
                {"name": "Navy Blue", "hex": "#1B365D", "desc": "Creates a sharp, rich contrast."},
                {"name": "Olive Green", "hex": "#556B2F", "desc": "Complements earthy tones perfectly."},
                {"name": "Burgundy", "hex": "#800020", "desc": "Adds a sophisticated warmth."},
                {"name": "Mustard Yellow", "hex": "#FFDB58", "desc": "Pops out brilliantly against your skin."},
            ]
        else:
            tone = "Cool Undertone (Fair/Light Complexion)"
            recommended_colors = [
                {"name": "Charcoal Grey", "hex": "#36454F", "desc": "Provides a balanced, striking look."},
                {"name": "Emerald Green", "hex": "#50C878", "desc": "Pops out brilliantly on cool skin."},
                {"name": "True White", "hex": "#F8F9FA", "desc": "Gives a crisp, clean appearance."},
                {"name": "Soft Pink", "hex": "#FFB6C1", "desc": "Adds a subtle, elegant contrast."},
            ]

        return {
            "status": "success",
            "detected_undertone": tone,
            "recommended_colors": recommended_colors,
        }

    except Exception as e:
        error_details = traceback.format_exc()
        print("SKIN TONE ERROR:\n", error_details)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ==========================================
# SERVER START FOR RAILWAY
# ==========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
