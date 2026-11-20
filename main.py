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
# MEDIAPIPE POSE INITIALIZATION
# ==========================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
)


# ==========================================
# BODY SEGMENTATION + MEASUREMENTS
# ==========================================

def extract_pose_metrics(
    image_bytes: bytes,
    height_cm: float
):

    nparr = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    if img is None:

        return None, "Invalid image file"

    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    results = pose.process(img_rgb)

    if not results.pose_landmarks:

        return None, (
            "Body clearly detect nahi hui. "
            "Full body photo seedhay kharay ho kar upload karein."
        )

    landmarks = results.pose_landmarks.landmark

    # Important landmarks

    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    left_shoulder = landmarks[
        mp_pose.PoseLandmark.LEFT_SHOULDER
    ]

    right_shoulder = landmarks[
        mp_pose.PoseLandmark.RIGHT_SHOULDER
    ]

    left_hip = landmarks[
        mp_pose.PoseLandmark.LEFT_HIP
    ]

    right_hip = landmarks[
        mp_pose.PoseLandmark.RIGHT_HIP
    ]

    left_ankle = landmarks[
        mp_pose.PoseLandmark.LEFT_ANKLE
    ]

    right_ankle = landmarks[
        mp_pose.PoseLandmark.RIGHT_ANKLE
    ]

    # Visibility check

    key_points = [
        nose,
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip,
        left_ankle,
        right_ankle
    ]

    avg_vis = sum(
        p.visibility for p in key_points
    ) / len(key_points)

    if avg_vis < 0.45:

        return None, (
            "Body visibility low hai. "
            "Clear full-body photo upload karein."
        )

    # Landmark coordinates

    nose_y = int(nose.y * h)

    shoulder_y = int(
        ((left_shoulder.y + right_shoulder.y) / 2) * h
    )

    hip_y = int(
        ((left_hip.y + right_hip.y) / 2) * h
    )

    ankle_y = int(
        ((left_ankle.y + right_ankle.y) / 2) * h
    )

    shoulder_width = (
        abs(left_shoulder.x - right_shoulder.x) * w
    )

    hip_width = (
        abs(left_hip.x - right_hip.x) * w
    )

    if hip_y <= shoulder_y or ankle_y <= nose_y:

        return None, (
            "Invalid body position. "
            "Seedha kharay ho kar photo upload karein."
        )

    torso_pixels = hip_y - shoulder_y

    # ======================================
    # BODY BOUNDING RECTANGLE
    # ======================================

    body_width = max(
        shoulder_width,
        hip_width
    ) * 1.8

    center_x = int(
        (
            left_shoulder.x +
            right_shoulder.x +
            left_hip.x +
            right_hip.x
        ) / 4 * w
    )

    top_y = max(
        0,
        int(nose_y - (ankle_y - nose_y) * 0.12)
    )

    bottom_y = min(
        h - 1,
        int(ankle_y + (ankle_y - nose_y) * 0.04)
    )

    x1 = max(
        0,
        int(center_x - body_width / 2)
    )

    x2 = min(
        w - 1,
        int(center_x + body_width / 2)
    )

    if (
        x2 - x1 < 20 or
        bottom_y - top_y < 50
    ):

        return None, "Full body photo properly visible nahi hai."

    # ======================================
    # OPENCV GRABCUT SEGMENTATION
    # ======================================

    mask = np.zeros(
        (h, w),
        np.uint8
    )

    bgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    fgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    rect = (
        x1,
        top_y,
        x2 - x1,
        bottom_y - top_y
    )

    try:

        cv2.grabCut(
            img,
            mask,
            rect,
            bgd_model,
            fgd_model,
            3,
            cv2.GC_INIT_WITH_RECT
        )

        person_mask = np.where(
            (mask == cv2.GC_FGD) |
            (mask == cv2
