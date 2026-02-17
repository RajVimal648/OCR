import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from aadhar_service import process_aadhaar
from pan_service import process_pan_card

from datetime import datetime
app = FastAPI(title="Enhanced Aadhaar Card OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post("/extractaadhaar")
async def extract_aadhaar(image: UploadFile = File(...)):

    if not allowed_file(image.filename):
        raise HTTPException(400, "Invalid file type")

    contents = await image.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")

    path = os.path.join(UPLOAD_FOLDER, image.filename)

    with open(path, "wb") as f:
        f.write(contents)

    try:
        result = process_aadhaar(path)

        return {
            "success": True,
            "message": "Aadhaar details extracted successfully",
            "data": result
        }

    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post('/extractpan')
async def extract_pan_info(image: UploadFile = File(...)):
    """API endpoint to extract PAN card information"""
    try:
        if not image.filename:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": "No file selected",
                    "message": "Please select a file to upload"
                }
            )
        
        if not allowed_file(image.filename):
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": "Invalid file type",
                    "message": f"Allowed file types: {', '.join(ALLOWED_EXTENSIONS)}",
                    "uploaded_file": image.filename
                }
            )
        
        contents = await image.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail={
                    "success": False,
                    "error": "File too large",
                    "message": f"Maximum file size is {MAX_FILE_SIZE // (1024*1024)}MB",
                    "uploaded_size": f"{len(contents) // (1024*1024)}MB"
                }
            )
        
        filename = "".join(c for c in image.filename if c.isalnum() or c in ('_', '.', '-'))
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(filepath, 'wb') as f:
            f.write(contents)
        
        result = process_pan_card(filepath)
        
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {filepath}: {e}")
        
        return {
            "success": True,
            "data": result,
            "message": "PAN card processed successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "message": "An error occurred while processing the image. Please ensure the image is a valid PAN card."
            }
        )
