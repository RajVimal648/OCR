from csv import reader
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import easyocr
import re
import os
import traceback
import uvicorn
import base64
from pyzbar.pyzbar import decode
from PIL import Image
from ultralytics import YOLO

classes = ["NAME", "DOB", "GENDER", "AADHAAR_NO", "ADDRESS", "PHOTO", "QR"]

# ==================== INITIALIZATION ====================

# Initialize YOLO model (Ultralytics)
try:
    print("🔄 Initializing YOLO model...")
    model = YOLO("best.pt")  
    print("✅ YOLO model initialized")
except Exception as e:
    print(f"⚠️ YOLO model initialization failed: {e}")
    model = None

# Initialize EasyOCR reader - THIS WAS MISSING!
try:
    print("🔄 Initializing EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if GPU available
    print("✅ EasyOCR reader initialized")
except Exception as e:
    print(f"❌ Failed to initialize EasyOCR: {e}")
    reader = None

try:
    net = None
    output_layers = None
    print("⚠️ OpenCV DNN YOLO not initialized (using Ultralytics instead)")
except Exception as e:
    print(f"⚠️ YOLO DNN initialization failed: {e}")
    net = None
    output_layers = None


def extract_aadhaar_number(text):
    """Extract 12-digit Aadhaar number"""
    pattern1 = r'\b(\d{4})[\s\-](\d{4})[\s\-](\d{4})\b'
    match = re.search(pattern1, text)
    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3)}"
    
    digits_only = re.sub(r'\D', '', text)
    all_12_digit = re.finditer(r'\d{12}', digits_only)
    
    for match in all_12_digit:
        num = match.group(0)
        if num[0] not in ['0', '1']:
            return f"{num[:4]} {num[4:8]} {num[8:]}"
    return ""


def extract_dob(text):
    """Extract date of birth"""
    patterns = [
        r'\b(\d{2})[/\-\.](\d{2})[/\-\.](\d{4})\b',
        r'\b(\d{2})[\s](\d{2})[\s](\d{4})\b',
        r'\b(\d{2})[/\-\.](\d{2})[/\-\.](\d{2})\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return ""

def extract_yob(text: str) -> str:
    t = text.upper()

    patterns = [
        r'\bYOB[:\s\-]*([12]\d{3})\b',
        r'\bY0B[:\s\-]*([12]\d{3})\b',
        r'\bYEAR\s*OF\s*BIRTH[:\s\-]*([12]\d{3})\b',
        r'\bBIRTH\s*YEAR[:\s\-]*([12]\d{3})\b'
    ]

    for p in patterns:
        m = re.search(p, t)
        if m:
            return m.group(1)

    dob_match = re.search(r'\b(\d{2})[/-](\d{2})[/-]([12]\d{3})\b', t)
    if dob_match:
        return dob_match.group(3)

    return ""


def extract_gender(text):
    """Extract gender"""
    t = text.upper()
    if "FEMALE" in t:
        return "FEMALE"
    if "MALE" in t:
        return "MALE"
    return ""


def extract_vid(text):
    """Extract 16-digit VID (Virtual ID)"""
    pattern = r'\b(\d{4})[\s\-](\d{4})[\s\-](\d{4})[\s\-](\d{4})\b'
    match = re.search(pattern, text)
    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}"
    
    digits_only = re.sub(r'\D', '', text)
    match = re.search(r'\d{16}', digits_only)
    if match:
        num = match.group(0)
        return f"{num[:4]} {num[4:8]} {num[8:12]} {num[12:]}"
    return ""


def clean_name(text):
    """Clean and extract name"""
    t = text.upper()
    junk = ["GOVERNMENT", "INDIA", "DOB", "YOB", "MALE", "FEMALE", "AADHAAR", 
            "UNIQUE", "IDENTIFICATION", "AUTHORITY", "ENROLLMENT", "NUMBER",
            "FATHER", "MOTHER", "HUSBAND", "WIFE", "ADDRESS", "VID", "YEAR", "BIRTH"]
    for j in junk:
        t = t.replace(j, "")
    
    t = re.sub(r"[^A-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    
    return t.title() if len(t.split()) >= 2 else ""


def extract_father_name(text, full_ocr_text=""):
    """
    Extract father's name using robust S/O, D/O, C/O detection
    Handles OCR noise, spaces, dots, hyphens, etc.
    """

    search_text = (text + " " + full_ocr_text).upper()

    # Very flexible patterns for OCR variations
    patterns = [
        r'\bS\s*[/\\\-\|]?\s*O\b[:\.\-\s]*([A-Z\s]{3,})',
        r'\bD\s*[/\\\-\|]?\s*O\b[:\.\-\s]*([A-Z\s]{3,})',
        r'\bC\s*[/\\\-\|]?\s*O\b[:\.\-\s]*([A-Z\s]{3,})',

        r'\bSON\s+OF\b[:\.\-\s]*([A-Z\s]{3,})',
        r'\bDAUGHTER\s+OF\b[:\.\-\s]*([A-Z\s]{3,})',
        r'\bCARE\s+OF\b[:\.\-\s]*([A-Z\s]{3,})',
    ]

    for pattern in patterns:
        match = re.search(pattern, search_text)
        if match:
            fname = match.group(1)

            # Stop at next keyword if OCR captured too much
            fname = re.split(
                r'\b(ADDRESS|DOB|DATE|BIRTH|PIN|INDIA|MALE|FEMALE)\b',
                fname
            )[0]

            # Clean name
            fname = re.sub(r'[^A-Z\s]', ' ', fname)
            fname = re.sub(r'\s+', ' ', fname).strip()

            if len(fname.split()) >= 2:
                return fname.title()

    return ""



def extract_address(text):
    """
    Extract address without S/O, D/O, C/O line
    """

    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)

    # ❌ Remove father info at beginning
    text = re.sub(
        r'\b(S\s*[/\\\-]?\s*O|D\s*[/\\\-]?\s*O|C\s*[/\\\-]?\s*O)\b.*?(?=GRAM|VILL|PO|DIST|STATE|\d{6})',
        '',
        text,
        flags=re.IGNORECASE
    )

    # 🧠 Extract address using PIN code anchor
    pin_match = re.search(r'\b\d{6}\b', text)

    if not pin_match:
        return ""

    start = max(0, pin_match.start() - 200)
    end = pin_match.end() + 10

    addr = text[start:end]

    # Clean symbols
    addr = re.sub(r'[^A-Za-z0-9,\-:/\s]', ' ', addr)
    addr = re.sub(r'\s+', ' ', addr).strip()

    return addr

def extract_pincode(text):
    """Extract 6-digit pincode"""
    match = re.search(r'\b(\d{6})\b', text)
    if match:
        return match.group(1)
    return ""


# ==================== IMAGE PROCESSING ====================

def extract_photo_from_aadhaar(img):
    """Extract face/photo from Aadhaar card and return as base64"""
    try:
        # Resize for faster identification if image is huge
        if img.shape[1] > 1500:
            scale = 1500 / img.shape[1]
            detect_img = cv2.resize(img, None, fx=scale, fy=scale)
        else:
            scale = 1.0
            detect_img = img

        # Try using face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            
            # Map back to original coordinates
            if scale != 1.0:
                x = int(x / scale)
                y = int(y / scale)
                w = int(w / scale)
                h = int(h / scale)

            # Add some padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            face_img = img[y:y+h, x:x+w]
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', face_img)
            base64_photo = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_photo}"
        
        # Fallback: Extract from typical photo location (left side of card)
        h, w = img.shape[:2]
        photo_region = img[int(h*0.2):int(h*0.6), int(w*0.05):int(w*0.25)]
        
        _, buffer = cv2.imencode('.jpg', photo_region)
        base64_photo = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_photo}"
        
    except Exception as e:
        print(f"Photo extraction error: {e}")
        return ""


def extract_qr_code(img):
    """Extract and decode QR code from Aadhaar card"""
    try:
        # Resize for faster detection if image is huge
        if img.shape[1] > 2000:
             scale = 2000 / img.shape[1]
             detect_img = cv2.resize(img, None, fx=scale, fy=scale)
        else:
             detect_img = img

        # Try to detect and decode QR code
        decoded_objects = decode(detect_img)
        
        if decoded_objects:
            for obj in decoded_objects:
                if obj.type == 'QRCODE':
                    # Extract QR region
                    x, y, w, h = obj.rect
                    
                    # Map back (if we used detecting on resized img, we need to scale back rect if we want high res crop)
                    # Use scale from above (detect_img was used)
                    if img.shape[1] > 2000:
                        scale = 2000 / img.shape[1]
                        x = int(x / scale)
                        y = int(y / scale)
                        w = int(w / scale)
                        h = int(h / scale)

                    padding = 10
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(img.shape[1] - x, w + 2 * padding)
                    h = min(img.shape[0] - y, h + 2 * padding)
                    
                    qr_img = img[y:y+h, x:x+w]
                    
                    # Convert to base64
                    _, buffer = cv2.imencode('.png', qr_img)
                    base64_qr = base64.b64encode(buffer).decode('utf-8')
                    
                    return {
                        "qr_image_base64": f"data:image/png;base64,{base64_qr}",
                        "qr_data": obj.data.decode('utf-8') if obj.data else ""
                    }
        
        # Fallback: Extract from typical QR location (bottom right)
        h, w = img.shape[:2]
        qr_region = img[int(h*0.6):int(h*0.95), int(w*0.75):int(w*0.95)]
        
        # Try decoding from extracted region (small region, fast)
        decoded_region = decode(qr_region)
        if decoded_region:
            qr_data = decoded_region[0].data.decode('utf-8') if decoded_region[0].data else ""
        else:
            qr_data = ""
        
        _, buffer = cv2.imencode('.png', qr_region)
        base64_qr = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "qr_image_base64": f"data:image/png;base64,{base64_qr}",
            "qr_data": qr_data
        }
        
    except Exception as e:
        print(f"QR extraction error: {e}")
        return {"qr_image_base64": "", "qr_data": ""}


def fallback_ocr_extraction(img, reader):
    """Comprehensive OCR extraction when YOLO fails"""
    print("🔄 Using fallback full-image OCR...")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray, detail=1, paragraph=False)
    
    all_text = " ".join([text for (_, text, _) in results])
    print(f"📝 Full OCR text: {all_text[:300]}...")
    
    data = {
        "NAME": "",
        "FATHER_NAME": "",
        "DOB": extract_dob(all_text),
        "YOB": extract_yob(all_text),
        "GENDER": extract_gender(all_text),
        "AADHAAR_NO": extract_aadhaar_number(all_text),
        "VID": extract_vid(all_text),
        "ADDRESS": "",
        "PINCODE": extract_pincode(all_text)
    }
    
    # Extract name and address from high-confidence text
    name_candidates = []
    address_parts = []
    
    for (bbox, text, conf) in results:
        if conf > 0.5:
            cleaned_name = clean_name(text)
            if cleaned_name and len(cleaned_name.split()) >= 2:
                name_candidates.append(cleaned_name)
            
            # Collect address parts
            if len(text.strip()) > 5:
                address_parts.append(text)
    
    # Set name (first valid candidate)
    if name_candidates and not data["NAME"]:
        data["NAME"] = name_candidates[0]
    
    # Extract father's name
    data["FATHER_NAME"] = extract_father_name(all_text)
    
    # Construct address
    if address_parts:
        full_address = " ".join(address_parts)
        data["ADDRESS"] = extract_address(full_address)
    
    return data


# ==================== CORE PROCESSING ====================

def process_aadhaar(image_path):
    """Main processing function with complete field extraction"""
    if reader is None:
        raise Exception("OCR Reader not initialized")

    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Failed to read image")
    
    h, w = img.shape[:2]

    # Initialize result data structure
    data = {
        "NAME": "",
        "FATHER_NAME": "",
        "DOB": "",
        "YOB": "",
        "GENDER": "",
        "AADHAAR_NO": "",
        "VID": "",
        "ADDRESS": "",
        "PINCODE": "",
        "PHOTO_BASE64": "",
        "QR_BASE64": "",
        "QR_DATA": ""
    }

    # Extract photo and QR code
    print("📸 Extracting photo...")
    data["PHOTO_BASE64"] = extract_photo_from_aadhaar(img)
    
    print("📱 Extracting QR code...")
    qr_result = extract_qr_code(img)
    data["QR_BASE64"] = qr_result["qr_image_base64"]
    data["QR_DATA"] = qr_result["qr_data"]

    # YOLO-based detection (if available using OpenCV DNN)
    if net is not None and output_layers is not None:
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes, confs, class_ids = [], [], []

        for out in outs:
            for det in out:
                scores = det[5:]
                cid = np.argmax(scores)
                conf = scores[cid]
                if conf > 0.25:
                    cx, cy, bw, bh = (det[:4] * np.array([w, h, w, h])).astype(int)
                    x = cx - bw // 2
                    y = cy - bh // 2
                    boxes.append([x, y, bw, bh])
                    confs.append(float(conf))
                    class_ids.append(cid)

        idxs = cv2.dnn.NMSBoxes(boxes, confs, 0.4, 0.4)
        
        print(f"🎯 YOLO detected {len(idxs.flatten()) if len(idxs) > 0 else 0} regions")

        if len(idxs) > 0:
            for i in idxs.flatten():
                label = classes[class_ids[i]] if class_ids[i] < len(classes) else "UNKNOWN"
                conf = confs[i]
                
                print(f"  📍 {label} (confidence: {conf:.2f})")
                
                x, y, bw, bh = boxes[i]
                roi = img[max(0, y):y + bh, max(0, x):x + bw]

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # Enhance ROI for better OCR
                gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                txt = reader.readtext(gray, detail=0, paragraph=True)
                text = " ".join(txt)
                
                print(f"     OCR: {text[:50]}...")

                if label == "NAME" and not data["NAME"]:
                    data["NAME"] = clean_name(text)
                elif label == "DOB" and not data["DOB"]:
                    data["DOB"] = extract_dob(text)
                    if not data["YOB"]:
                        data["YOB"] = extract_yob(text)
                elif label == "GENDER" and not data["GENDER"]:
                    data["GENDER"] = extract_gender(text)
                elif label == "AADHAAR_NO" and not data["AADHAAR_NO"]:
                    data["AADHAAR_NO"] = extract_aadhaar_number(text)
                elif label == "ADDRESS" and not data["ADDRESS"]:
                    data["ADDRESS"] = extract_address(text)
                    if not data["PINCODE"]:
                        data["PINCODE"] = extract_pincode(text)
    
    # Always run fallback OCR if critical fields are missing
    if not data["AADHAAR_NO"] or not data["GENDER"] or not data["NAME"]:
        print("⚠️ Missing fields, using fallback OCR...")
        
        # Pass the original image, preprocessing happens inside
        try:
           # Preprocess full image for fallback: Smart Resize
           h, w = img.shape[:2]
           if w < 800:
               scale = 1000.0 / w
               processed_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
           elif w > 1000:
               scale = 1000.0 / w
               processed_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
           else:
               processed_img = img
           
           print(f"Fallback Image Shape: {processed_img.shape}")
           
           import time
           t0 = time.time()
           fallback_data = fallback_ocr_extraction(processed_img, reader)
           print(f"Fallback OCR took {time.time()-t0:.2f}s")
        except Exception as e:
           print(f"Fallback preprocessing error: {e}")
           fallback_data = fallback_ocr_extraction(img, reader)
        
        for key in fallback_data:
            if not data[key] and fallback_data[key]:
                data[key] = fallback_data[key]
                print(f"  ✓ Fallback found {key}: {fallback_data[key][:50] if len(str(fallback_data[key])) > 50 else fallback_data[key]}")

    is_valid = bool(data["AADHAAR_NO"] or data["VID"])
    
    print(f"\n✅ Final extraction complete")

    return {
        "name": data["NAME"],
        "father_name": data["FATHER_NAME"],
        "dob": data["DOB"],
        "yob": data["YOB"],
        "gender": data["GENDER"],
        "aadhaar_number": data["AADHAAR_NO"],
        "vid": data["VID"],
        "address": data["ADDRESS"],
        "pincode": data["PINCODE"],
        "photo_base64": data["PHOTO_BASE64"],
        "qr_code_base64": data["QR_BASE64"],
        "is_valid": is_valid
    }
