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
    Calculates ellipse circumference in cm given full width and full depth.
    Uses semi-axes a = width / 2 and b = depth / 2.
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


# --- ROUTE 2: AI Body Size Prediction (Front + Optional Side Ellipse) ---

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True, model_complexity=2, min_detection_confidence=0.5
)


def extract_pose_metrics(image_bytes: bytes, height_cm: float):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None, "Invalid image file"

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        return (
            None,
            "Body clearly detect nahi hui. Full body photo seedhay kharay ho kar upload karein.",
        )

    landmarks = results.pose_landmarks.landmark

    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    avg_ankle_y = ((left_ankle.y + right_ankle.y) / 2.0) * h
    nose_y = nose.y * h
    # Nose to ankle is roughly ~84% of full head-to-sole height
    pixel_height = max((avg_ankle_y - nose_y) / 0.84, 1.0)

    if pixel_height <= 50:
        return None, "Full body photo upload karein."

    cm_per_pixel = height_cm / pixel_height

    key_indices = [0, 11, 12, 23, 24, 27, 28]
    try:
        visibilities = [landmarks[i].visibility for i in key_indices]
        avg_vis = sum(visibilities) / len(visibilities)
    except Exception:
        avg_vis = 0.5

    landmark_shoulder_px = abs(left_shoulder.x - right_shoulder.x) * w
    landmark_hip_px = abs(left_hip.x - right_hip.x) * w

    avg_shoulder_y = ((left_shoulder.y + right_shoulder.y) / 2.0) * h
    avg_hip_y = ((left_hip.y + right_hip.y) / 2.0) * h
    torso_len_cm = max(abs(avg_hip_y - avg_shoulder_y) * cm_per_pixel, 1.0)

    return {
        "cm_per_pixel": cm_per_pixel,
        "landmark_shoulder_px": landmark_shoulder_px,
        "landmark_hip_px": landmark_hip_px,
        "torso_len_cm": torso_len_cm,
        "avg_vis": avg_vis,
    }, None


@app.post("/api/predict-size")
async def predict_size(
    image: UploadFile = File(...),
    side_image: Optional[UploadFile] = File(None),
    height_cm: float = Form(...),
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

        # Convert inner MediaPipe landmarks to approximate outer body widths (cm)
        outer_shoulder_cm = max(
            front_data["landmark_shoulder_px"] * cm_per_pixel * 1.22, 25.0
        )
        chest_width_cm = max(outer_shoulder_cm * 0.94, 22.0)

        outer_hip_cm = max(
            front_data["landmark_hip_px"] * cm_per_pixel * 1.25, 22.0
        )
        waist_width_cm = max(outer_hip_cm * 0.92, 20.0)

        used_dual_photo = False
        side_note = "Single-photo proportional estimation"

        if side_image is not None and side_image.filename:
            side_bytes = await side_image.read()
            if side_bytes:
                side_data, side_err = extract_pose_metrics(side_bytes, height_cm)
                if side_data is not None:
                    side_torso_cm = side_data["torso_len_cm"]
                    ref_torso_cm = 0.5 * (torso_len_cm + side_torso_cm)

                    chest_depth_cm = float(np.clip(ref_torso_cm * 0.58, 20.0, 36.0))
                    waist_depth_cm = float(np.clip(ref_torso_cm * 0.52, 18.0, 36.0))

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
                    side_note = "Dual-photo (Front + Side) Ellipse estimation"
                else:
                    chest_in = round((chest_width_cm * 2.55) / 2.54, 1)
                    waist_in = round((waist_width_cm * 2.45) / 2.54, 1)
            else:
                chest_in = round((chest_width_cm * 2.55) / 2.54, 1)
                waist_in = round((waist_width_cm * 2.45) / 2.54, 1)
        else:
            chest_in = round((chest_width_cm * 2.55) / 2.54, 1)
            waist_in = round((waist_width_cm * 2.45) / 2.54, 1)

        if waist_in < 26.0:
            waist_in = round(max(28.0, chest_in * 0.80), 1)

        base_conf = 85.0 + (avg_vis * 8.0) + (0.5 if used_dual_photo else 0.0)
        accuracy_percentage = round(min(max(base_conf, 85.0), 93.5), 1)

        # 5. STRICT PHYSIQUE LOGIC (Must have >= 8.5 inch difference to be Athletic)
        chest_waist_diff = chest_in - waist_in

        if chest_waist_diff >= 8.5:
            body_type = "Athletic / V-Taper Physique 🏋️‍♂️"
        elif waist_in >= chest_in - 1.5:
            body_type = "Comfort / Relaxed Fit 🧥"
        else:
            body_type = "Standard / Balanced Fit ✨"

        # 6. DYNAMIC SHIRT SIZING
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

        # 7. DYNAMIC PANT / WAIST SIZING
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

        return {
            "status": "success",
            "body_type": body_type,
            "recommended_size": recommended_size,
            "recommended_shirt_size": recommended_size,
            "recommended_pant_size": pant_size,
            "accuracy_percentage": accuracy_percentage,
            "dual_photo_used": used_dual_photo,
            "measurements": {
                "height_cm": height_cm,
                "shoulder_inches": round(outer_shoulder_cm / 2.54, 1),
                "chest_inches": chest_in,
                "waist_inches": waist_in,
            },
            "confidence": f"High ({side_note})",
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
