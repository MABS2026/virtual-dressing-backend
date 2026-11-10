import os
import io
import traceback
import nest_asyncio
import uvicorn
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from gradio_client import Client, handle_file
from pyngrok import ngrok

# 🔑 Apka Hugging Face Token
os.environ["HF_TOKEN"] = ""

# 🔑 Apka Ngrok Authtoken
!ngrok authtoken "3LQe53HXgxOzGBcncbYsmSDmikC_3heM441norRe68yRinXgV"

nest_asyncio.apply()
app = FastAPI()

print("⏳ Connecting to IDM-VTON Model on Hugging Face...")
client = Client("yisol/IDM-VTON")
print("✅ Connected Successfully!")

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

        result = client.predict(
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
@app.post("/api/predict-size")
async def predict_size(file: UploadFile = File(...), height_cm: float = Form(170.0)):
    try:
        height_inches = height_cm / 2.54

        estimated_chest_inches = round(height_inches * 0.55, 1)
        estimated_waist_inches = round(height_inches * 0.45, 1)

        fit_pref = "Regular"
        if estimated_waist_inches < 32:
            fit_pref = "Slim Fit"
        elif estimated_waist_inches > 36:
            fit_pref = "Relaxed / Oversized"

        return {
            "status": "success",
            "height_cm": height_cm,
            "estimated_chest_inches": estimated_chest_inches,
            "estimated_waist_inches": estimated_waist_inches,
            "recommended_fit": fit_pref
        }

    except Exception as e:
        error_details = traceback.format_exc()
        print("❌ SIZE PREDICTION ERROR:\n", error_details)
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- ROUTE 3: AI Skin Tone & Color Analysis ---
@app.post("/api/skin-tone-analysis")
async def skin_tone_analysis(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        # Center region crop (Face area assumption)
        h, w, _ = img.shape
        center_crop = img[int(h * 0.3):int(h * 0.6), int(w * 0.3):int(w * 0.6)]

        avg_color_per_row = np.average(center_crop, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        b, g, r = avg_color  # OpenCV uses BGR

        # AI Undertone & Color Matching Logic
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


# --- Server Start & Ngrok Tunneling ---
ngrok.kill()
tunnel = ngrok.connect(8001)
PUBLIC_URL = tunnel.public_url
print(f"\n🚀 NAYA LIVE BASE URL: {PUBLIC_URL}\n")

config = uvicorn.Config(app, host="0.0.0.0", port=8001, loop="asyncio")
server = uvicorn.Server(config)

await server.serve()