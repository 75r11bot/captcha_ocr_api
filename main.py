#main.py
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, Response  
import os
import cv2
import numpy as np
from utils.image_processing import preprocess_image, match_template, save_templates, crop_captcha, load_templates

app = FastAPI()

@app.head("/")
async def health_check():
    return Response(status_code=200)

@app.on_event("startup")
def startup_event():
    load_templates()

@app.post("/api/reload-templates")
def reload_templates():
    load_templates()
    return {"message": "Templates reloaded"}

@app.post("/api/add-template")
async def add_template(
    label: str = Query(..., min_length=4, max_length=4, regex="^[a-zA-Z0-9]{4}$"),
    file: UploadFile = File(...)
):
    image_bytes = await file.read()
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    char_images = crop_captcha(img, num_chars=4)

    if len(label) != len(char_images):
        return JSONResponse(status_code=400, content={"error": "Label length does not match cropped characters count."})

    saved_files = save_templates(label, char_images)
    load_templates()

    return {"message": "Templates saved.", "files": saved_files}

@app.post("/api/ocr")
async def ocr(file: UploadFile = File(...)):
    image_bytes = await file.read()
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    chars = crop_captcha(img, num_chars=4)

    result_text = ""
    confidences = []

    for i, char_img in enumerate(chars):
        print(f"\n--- Matching character #{i+1} ---")
        label, conf = match_template(char_img)
        result_text += label
        confidences.append(conf)

    avg_confidence = round(sum(confidences) / len(confidences), 0)  # <-- ปัดเป็นจำนวนเต็ม %

    return {
        "text": result_text,
        "confidence": int(avg_confidence)
    }


@app.get("/")
def read_root():
    return {"status": "ok"}