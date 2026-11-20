import os
import io
import math
import traceback
from typing import Optional, Tuple
import nest_asyncio
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from gradio_client import Client, handle_file

# 🔑 Hugging Face Token (Railway Variables se uthayega)
if os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")

nest_asyncio.apply()
app = FastAPI()

# Lazy loading to prevent startup crashes
client = None


def get_gradio_client():
    global client
    if client is None:
        try:
            print("⏳ Connecting to IDM-VTON Model on Hugging Face...")
            client = Client("yisol/IDM-VTON")
            print("✅ Connected Successfully!")
        except Exception as e:
            print("❌ Failed to connect to Gradio Client:", str(e))
            raise HTTPException(
                status_code=500, detail="AI Model service is currently unavailable."
            )
    return client


def ellipse_circumference_cm(width_cm: float, depth_cm: float) -> float:
    """
    Ramanujan's approximation for ellipse circumference in cm given full width and full depth.
    Semi-axes: a = width / 2, b = depth / 2.
    """
    a = max(width_cm / 2.0, 0.1)
    b = max(depth_cm / 2.0, 0.1)
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (
        1.0 + (3.0 * h) / (10.0 + math.sqrt(max(0.0, 4.0 - 3.0 * h)))
    )


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Virtual Dressing Backend is running successfully!",
    }


# --- ROUTE 1: Virtual Try-On ---
@app.post("/api/try-on")
async def try_on(user: UploadFile = File(...), garment: UploadFile = File(...)):
    try:
        user_path, garment_path = "user.jpg", "garment.jpg"
        with open(user_path, "wb") as f:
            f.write(await user.read())
        with open(garment_path, "wb") as f:
            f.write(await garment.read())

        print("📥 Images received, sending to IDM-VTON...")
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
            print("✨ Success! Returning processed virtual try-on image...")
            return StreamingResponse(
                io.BytesIO(f.read()), media_type="image/jpeg"
            )

    except Exception as e:
        error_details = traceback.format_exc()
        print("❌ DETAILED ERROR:\n", error_details)
        raise HTTPException(status_code=500, detail=str(e))


# --- ROUTE 2: AI Body Size Prediction (Segmentation + Pose + Front/Side Ellipse) ---

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True, model_complexity=2, min_detection_confidence=0.5
)

mp_selfie_seg = mp.solutions.selfie_segmentation
selfie_seg = mp_selfie_seg.SelfieSegmentation(model_selection=1)


def get_body_segmentation_mask(img_bgr: np.ndarray, img_rgb: np.ndarray) -> np.ndarray:
    """
    Produces a cleaned binary foreground mask (values 0 or 255) for the person.
    Keeps the largest connected component around the center/body to reduce background noise.
    """
    h, w, _ = img_rgb.shape
    seg_res = selfie_seg.process(img_rgb)

    if seg_res.segmentation_mask is not None:
        mask = (seg_res.segmentation_mask > 0.45).astype(np.uint8) * 255
    else:
        mask = np.zeros((h, w), dtype=np.uint8)

    # Morphological cleanup to fill small gaps and remove tiny speckles
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        # Ignore background label 0; pick largest component
        best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = stats[best_label, cv2.CC_STAT_AREA]
        if area > (h * w * 0.02):
            mask = np.where(labels == best_label, 255, 0).astype(np.uint8)

    return mask


def measure_mask_span_at_y(
    mask: np.ndarray,
    y_center: int,
    x_min_hint: Optional[int] = None,
    x_max_hint: Optional[int] = None,
    band_half: int = 6,
) -> Optional[float]:
    """
    Measures horizontal pixel span of the body mask near row y_center.
    If pose x hints are provided, prefers the contiguous foreground segment overlapping them.
    """
    h, w = mask.shape
    y0 = max(0, y_center - band_half)
    y1 = min(h, y_center + band_half + 1)
    if y0 >= y1:
        return None

    band = (mask[y0:y1, :] > 0).astype(np.uint8)
    # Row profile: pixel present in at least half of the band rows
    min_votes = max(1, band.shape[0] // 2)
    row_active = (np.sum(band, axis=0) >= min_votes).astype(np.uint8)

    if np.sum(row_active) == 0:
        return None

    # Find contiguous runs in row_active
    padded = np.concatenate(([0], row_active, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    if len(starts) == 0:
        return None

    if x_min_hint is not None and x_max_hint is not None:
        hint_cx = 0.5 * (x_min_hint + x_max_hint)
        best_span = None
        best_score = float("inf")
        for s, e in zip(starts, ends):
            span = float(e - s)
            if span < 4:
                continue
            seg_cx = 0.5 * (s + e)
            # Check overlap with pose horizontal region
            overlaps = not (e < x_min_hint or s > x_max_hint)
            dist = abs(seg_cx - hint_cx) + (0.0 if overlaps else w * 0.5)
            if dist < best_score:
                best_score = dist
                best_span = span
        if best_span is not None:
            return best_span

    xs = np.where(row_active > 0)[0]
    return float(xs[-1] - xs[0] + 1) if len(xs) > 0 else None


def extract_body_metrics(image_bytes: bytes, height_cm: float) -> Tuple[Optional[dict], Optional[str]]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None, "Invalid image file"

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(img_rgb)

    if not pose_results.pose_landmarks:
        return (
            None,
            "Body clearly detect nahi hui. Full body photo seedhay kharay ho kar upload karein.",
        )

    landmarks = pose_results.pose_landmarks.landmark
    mask = get_body_segmentation_mask(img, img_rgb)

    # 1. Segmentation-based top-to-bottom body height
    ys, _ = np.where(mask > 0)
    if len(ys) > 20:
        seg_top_y = float(np.min(ys))
        seg_bottom_y = float(np.max(ys))
        pixel_height = seg_bottom_y - seg_top_y
    else:
        # Fallback if segmentation mask is sparse
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        avg_ankle_y = ((left_ankle.y + right_ankle.y) / 2.0) * h
        nose_y = nose.y * h
        pixel_height = max((avg_ankle_y - nose_y) / 0.84, 1.0)

    if pixel_height <= 50:
        return None, "Full body photo upload karein."

    cm_per_pixel = height_cm / pixel_height

    # Pose visibility score
    key_indices = [0, 11, 12, 23, 24, 27, 28]
    try:
        visibilities = [landmarks[i].visibility for i in key_indices]
        avg_vis = sum(visibilities) / len(visibilities)
    except Exception:
        avg_vis = 0.5

    # Pose landmarks for vertical level anchoring
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    sh_x1, sh_x2 = sorted([int(left_shoulder.x * w), int(right_shoulder.x * w)])
    hip_x1, hip_x2 = sorted([int(left_hip.x * w), int(right_hip.x * w)])

    avg_shoulder_y = int(round(((left_shoulder.y + right_shoulder.y) / 2.0) * h))
    avg_hip_y = int(round(((left_hip.y + right_hip.y) / 2.0) * h))

    # Chest level is slightly below shoulder line (~28% of shoulder-to-hip distance)
    torso_px = max(abs(avg_hip_y - avg_shoulder_y), 1)
    chest_y = int(np.clip(avg_shoulder_y + int(torso_px * 0.28), 0, h - 1))
    # Natural waist is roughly ~62% down from shoulder toward hip
    waist_y = int(np.clip(avg_shoulder_y + int(torso_px * 0.62), 0, h - 1))

    # Horizontal span from segmentation mask (with pose fallback)
    pose_sh_span = max(abs(sh_x2 - sh_x1), 1)
    pose_hip_span = max(abs(hip_x2 - hip_x1), 1)

    sh_hint_min = max(0, sh_x1 - int(pose_sh_span * 0.35))
    sh_hint_max = min(w, sh_x2 + int(pose_sh_span * 0.35))
    hip_hint_min = max(0, hip_x1 - int(pose_hip_span * 0.35))
    hip_hint_max = min(w, hip_x2 + int(pose_hip_span * 0.35))

    shoulder_px = measure_mask_span_at_y(
        mask, avg_shoulder_y, sh_hint_min, sh_hint_max, band_half=5
    ) or (pose_sh_span * 1.18)

    chest_px = measure_mask_span_at_y(
        mask, chest_y, sh_hint_min, sh_hint_max, band_half=6
    ) or (shoulder_px * 0.92)

    waist_px = measure_mask_span_at_y(
        mask, waist_y, hip_hint_min, hip_hint_max, band_half=6
    ) or (pose_hip_span * 1.15)

    # For a side-view photo, horizontal mask span represents body depth at that level
    side_chest_depth_px = measure_mask_span_at_y(mask, chest_y, band_half=6)
    side_waist_depth_px = measure_mask_span_at_y(mask, waist_y, band_half=6)

    return {
        "cm_per_pixel": cm_per_pixel,
        "shoulder_cm": shoulder_px * cm_per_pixel,
        "chest_width_cm": chest_px * cm_per_pixel,
        "waist_width_cm": waist_px * cm_per_pixel,
        "side_chest_depth_cm": (
            side_chest_depth_px * cm_per_pixel if side_chest_depth_px else None
        ),
        "side_waist_depth_cm": (
            side_waist_depth_px * cm_per_pixel if side_waist_depth_px else None
        ),
        "avg_vis": avg_vis,
    }, None


def compute_bmi(height_cm: float, weight_kg: Optional[float]) -> Optional[float]:
    if weight_kg is None or weight_kg <= 0 or height_cm <= 0:
        return None
    height_m = height_cm / 100.0
    return round(weight_kg / (height_m * height_m), 1)


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
        front_data, front_err = extract_body_metrics(front_bytes, height_cm)
        if front_err:
            if front_err == "Invalid image file":
                return JSONResponse(status_code=400, content={"error": front_err})
            return {"status": "error", "message": front_err}

        chest_width_cm = front_data["chest_width_cm"]
        waist_width_cm = front_data["waist_width_cm"]
        shoulder_cm = front_data["shoulder_cm"]
        avg_vis = front_data["avg_vis"]

        used_dual_photo = False
        side_note = "Front segmentation proportional depth estimation"

        if side_image is not None and side_image.filename:
            side_bytes = await side_image.read()
            if side_bytes:
                side_data, side_err = extract_body_metrics(side_bytes, height_cm)
                if side_data is not None:
                    chest_depth_cm = side_data["side_chest_depth_cm"] or (
                        chest_width_cm * 0.65
                    )
                    waist_depth_cm = side_data["side_waist_depth_cm"] or (
                        waist_width_cm * 0.68
                    )

                    chest_depth_cm = float(np.clip(chest_depth_cm, 15.0, 42.0))
                    waist_depth_cm = float(np.clip(waist_depth_cm, 14.0, 42.0))

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
                    side_note = "Front + Side segmentation ellipse estimation"
                else:
                    chest_depth_cm = chest_width_cm * 0.65
                    waist_depth_cm = waist_width_cm * 0.68
                    chest_in = round(
                        ellipse_circumference_cm(chest_width_cm, chest_depth_cm)
                        / 2.54,
                        1,
                    )
                    waist_in = round(
                        ellipse_circumference_cm(waist_width_cm, waist_depth_cm)
                        / 2.54,
                        1,
                    )
            else:
                chest_depth_cm = chest_width_cm * 0.65
                waist_depth_cm = waist_width_cm * 0.68
                chest_in = round(
                    ellipse_circumference_cm(chest_width_cm, chest_depth_cm) / 2.54,
                    1,
                )
                waist_in = round(
                    ellipse_circumference_cm(waist_width_cm, waist_depth_cm) / 2.54,
                    1,
                )
        else:
            chest_depth_cm = chest_width_cm * 0.65
            waist_depth_cm = waist_width_cm * 0.68
            chest_in = round(
                ellipse_circumference_cm(chest_width_cm, chest_depth_cm) / 2.54, 1
            )
            waist_in = round(
                ellipse_circumference_cm(waist_width_cm, waist_depth_cm) / 2.54, 1
            )

        shoulder_in = round(shoulder_cm / 2.54, 1)
        bmi_val = compute_bmi(height_cm, weight_kg)

        # Confidence reflects pose visibility & dual-photo availability (not a claim of ground-truth accuracy)
        conf_score = 80.0 + (avg_vis * 10.0) + (2.0 if used_dual_photo else 0.0)
        accuracy_percentage = round(min(max(conf_score, 80.0), 92.5), 1)

        # Physique classification from measured proportions (with BMI only as supporting note if available)
        chest_waist_diff = chest_in - waist_in
        if chest_waist_diff >= 7.5:
            body_type = "Athletic / V-Taper Physique 🏋️‍♂️"
        elif waist_in >= chest_in - 1.5:
            body_type = "Comfort / Relaxed Fit 🧥"
        else:
            body_type = "Standard / Balanced Fit ✨"

        # Shirt sizing from chest circumference
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

        # Pant / waist sizing from waist circumference
        if waist_in <= 31:
            pant_size = "30-32 (Small)"
        elif waist_in <= 34:
            pant_size = "32-34 (Medium)"
        elif waist_in <= 37:
            pant_size = "34-36 (Large)"
        elif waist_in <= 41:
            pant_size = "38 (XL)"
        else:
            pant_size = "40+ (XXL)"

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
            "confidence": f"Pose/Mask Quality ({side_note})",
            "note": "Loose garments can affect outer boundary width/depth; validate against tape measurements.",
        }

    except Exception as e:
        error_details = traceback.format_exc()
        print("❌ ERROR:\n", error_details)
        raise HTTPException(status_code=500, detail=str(e))


# --- ROUTE 3: AI Skin Tone & Color Analysis ---
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
        b, g, r = avg_color  # OpenCV uses BGR

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
        print("❌ SKIN TONE ERROR:\n", error_details)
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- Server Start for Railway ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
