import os
import io
import traceback
import nest_asyncio
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
from fastapi import UploadFile, File, Form, HTTPException
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
            raise HTTPException(status_code=500, detail="AI Model service is currently unavailable.")
    return client

@app.get("/")
async def root():
    return {"status": "online", "message": "Virtual Dressing Backend is running successfully!"}

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
            dict={"background": handle_file(user_path), "layers": [], "composite": None},
            garm_img=handle_file(garment_path),
            garment_des="a black cotton crewneck t-shirt with short sleeves",
            is_checked=True,
            denoise_steps=40,
            seed=42,
            api_name="/tryon"
        )

        with open(result[0], "rb") as f:
            print("✨ Success! Returning processed virtual try-on image...")
            return StreamingResponse(io.BytesIO(f.read()), media_type="image/jpeg")

    except Exception as e:
        error_details = traceback.format_exc()
        print("❌ DETAILED ERROR:\n", error_details)
        raise HTTPException(status_code=500, detail=str(e))


# --- ROUTE 2: AI Body Size Prediction ---

# MediaPipe Pose initialize karein
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

@app.post("/api/predict-size")
async def predict_size(
    image: UploadFile = File(...),
    height_cm: float = Form(...),
    gender: str = Form("male")  # optional: male/female sizing chart ke liye
):
    try:
        # Image read karein
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # MediaPipe se body landmarks detect karein
        results = pose.process(img_rgb)
        
        if not results.pose_landmarks:
            return {
                "status": "error",
                "message": "Body clearly detect nahi hui. Full body photo seedhay kharay ho kar upload karein."
            }

        landmarks = results.pose_landmarks.landmark

        # Key Landmarks nikalain (x, y pixels mein)
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # 1. Pixel Height Calculate Karein (Nose se Ankle tak + approx head/foot offset)
        avg_ankle_y = ((left_ankle.y + right_ankle.y) / 2) * h
        nose_y = nose.y * h
        
        # Nose se ankle tak ka faasla total height ka taqreeban 88% hota hai
        pixel_height = (avg_ankle_y - nose_y) / 0.88
        
        if pixel_height <= 50:
            return {
                "status": "error",
                "message": "Full body (sar se paon tak) tasveer mein nazar aani chahiye."
            }

        # Pixel-to-CM Ratio (1 pixel kitne cm ke barabar hai)
        cm_per_pixel = height_cm / pixel_height

        # 2. Shoulder Width (cm & inches)
        shoulder_px = abs(left_shoulder.x - right_shoulder.x) * w
        shoulder_cm = shoulder_px * cm_per_pixel
        shoulder_in = round(shoulder_cm / 2.54, 1)

        # 3. Chest Circumference Estimate (2D width se 3D circumference approx)
        # Human torso front-to-back depth ratio ke mutabiq chest circumference ~ 2.35x front shoulder/upper-torso width
        chest_cm = shoulder_cm * 2.25
        chest_in = round(chest_cm / 2.54, 1)

        # 4. Hip / Waist Width & Circumference
        hip_px = abs(left_hip.x - right_hip.x) * w
        hip_cm = hip_px * cm_per_pixel
        # Waist hips se thori narrow hoti hai (approx 2.45x hip front width)
        waist_cm = hip_cm * 2.45
        waist_in = round(waist_cm / 2.54, 1)

        # 5. Standard Shirt Size Determination (Chest Inches ke hisab se)
        if chest_in < 36:
            recommended_size = "S"
        elif 36 <= chest_in < 39.5:
            recommended_size = "M"
        elif 39.5 <= chest_in < 42.5:
            recommended_size = "L"
        elif 42.5 <= chest_in < 46:
            recommended_size = "XL"
        else:
            recommended_size = "XXL"

        return {
            "status": "success",
            "recommended_size": recommended_size,
            "measurements": {
                "height_cm": height_cm,
                "shoulder_inches": shoulder_in,
                "chest_inches": chest_in,
                "waist_inches": waist_in
            },
            "confidence": "High (AI Pose Landmarks)"
        }

    except Exception as e:
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
        center_crop = img[int(h * 0.3):int(h * 0.6), int(w * 0.3):int(w * 0.6)]

        avg_color_per_row = np.average(center_crop, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        b, g, r = avg_color  # OpenCV uses BGR

        if r > g and r > b:
            tone = "Warm Undertone (Medium/Olive Complexion)"
            recommended_colors = [
                {"name": "Navy Blue", "hex": "#1B365D", "desc": "Creates a sharp, rich contrast."},
                {"name": "Olive Green", "hex": "#556B2F", "desc": "Complements earthy tones perfectly."},
                {"name": "Burgundy", "hex": "#800020", "desc": "Adds a sophisticated warmth."},
                {"name": "Mustard Yellow", "hex": "#FFDB58", "desc": "Pops out brilliantly against your skin."}
            ]
        else:
            tone = "Cool Undertone (Fair/Light Complexion)"
            recommended_colors = [
                {"name": "Charcoal Grey", "hex": "#36454F", "desc": "Provides a balanced, striking look."},
                {"name": "Emerald Green", "hex": "#50C878", "desc": "Pops out brilliantly on cool skin."},
                {"name": "True White", "hex": "#F8F9FA", "desc": "Gives a crisp, clean appearance."},
                {"name": "Soft Pink", "hex": "#FFB6C1", "desc": "Adds a subtle, elegant contrast."}
            ]

        return {
            "status": "success",
            "detected_undertone": tone,
            "recommended_colors": recommended_colors
        }

    except Exception as e:
        error_details = traceback.format_exc()
        print("❌ SKIN TONE ERROR:\n", error_details)
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- Server Start for Railway ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
