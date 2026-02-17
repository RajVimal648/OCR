import cv2
import numpy as np
import easyocr
import base64
import re
import os
from pathlib import Path
import traceback
import uvicorn
from datetime import datetime
from typing import Optional, Dict, Any, List
from pyzbar import pyzbar
from ultralytics import YOLO

classes = ["NAME", "FATHER_NAME", "DOB", "PAN_NUMBER"]

try:
    model = YOLO("best.pt")
    print("✓ YOLO (Ultralytics) model loaded successfully")
    
    # Check if model has expected classes
    model_classes = [c.lower() for c in model.names.values()]
    expected_classes = ["name", "father_name", "father-s name", "dob", "pan_number", "pan number"]
    
    if not any(c in model_classes for c in expected_classes):
        print(f"⚠️  WARNING: Loaded model 'best.pt' does not appear to have PAN card classes.")
        print(f"   Model classes: {list(model.names.values())[:5]}... (Total: {len(model.names)})")
        print(f"   Expected classes like: {expected_classes}")
except Exception as e:
    print(f"✗ Error loading YOLO model: {e}")
    model = None

try:
    reader = easyocr.Reader(['en'], gpu=False)
    print("✓ EasyOCR reader initialized successfully")
except Exception as e:
    print(f"✗ Error initializing EasyOCR: {e}")
    reader = None
    
def clean_text(text: str) -> str:
    """Clean and normalize extracted text"""
    if not text:
        return ""
    # Remove common OCR artifacts and non-printable chars
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'[|\[\]{}_]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_image_for_ocr(image: np.ndarray) -> List[np.ndarray]:
    """Generate multiple preprocessed versions of the image for better OCR"""
    processed_images = []
    
    # 1. Original
    processed_images.append(image)
    
    # 2. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_images.append(gray)
    
    # 3. CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    processed_images.append(enhanced)
    
    # 4. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    processed_images.append(adaptive)
    
    # 5. Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(otsu)
    
    # 6. Bilateral filter + CLAHE
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    bilateral_clahe = clahe.apply(bilateral)
    processed_images.append(bilateral_clahe)
    
    # 7. Denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    processed_images.append(denoised)
    
    # 8. Sharpening
    kernel_sharpen = np.array([[-1,-1,-1], 
                               [-1, 9,-1], 
                               [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
    processed_images.append(sharpened)
    
    return processed_images


def extract_text_robust(roi: np.ndarray, reader) -> str:
    """Extract text from ROI using multiple preprocessing methods"""
    if roi is None or roi.size == 0:
        return ""
    
    # Upscale if too small
    h, w = roi.shape[:2]
    if h < 60 or w < 60:
        scale_factor = max(2.0, 80.0 / min(h, w))
        roi = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, 
                        interpolation=cv2.INTER_CUBIC)
    
    results = []
    
    # Get multiple preprocessed versions
    if len(roi.shape) == 2:  # Already grayscale
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    else:
        roi_bgr = roi
    
    processed_images = preprocess_image_for_ocr(roi_bgr)
    
    # Try OCR on each preprocessed version
    for idx, img in enumerate(processed_images):
        try:
            # Read with detail to get confidence scores
            detections = reader.readtext(img, detail=1, paragraph=False)
            
            if detections:
                # Filter by confidence and combine
                texts = [text for bbox, text, conf in detections if conf > 0.3]
                if texts:
                    combined = " ".join(texts)
                    results.append((combined, len(combined), idx))
        except Exception as e:
            continue
    
    if results:
        # Sort by length (longer is often better) and return best
        results.sort(key=lambda x: x[1], reverse=True)
        return results[0][0]
    
    return ""


def extract_pan_number(text_dict: Dict, full_text: str) -> str:
    """Extract PAN number with multiple pattern matching strategies"""
    # Regex patterns without word boundaries for flexibility
    pan_patterns = [
        r'[A-Z]{5}[0-9]{4}[A-Z]{1}',
        r'[A-Z]{3}P[A-Z]{1}[0-9]{4}[A-Z]{1}',
        r'[A-Z]{3}[PCHATFBLJG]{1}[A-Z]{1}[0-9]{4}[A-Z]{1}',
    ]
    
    def clean_candidate(text: str) -> str:
        """Heuristic cleaning for potential PAN strings"""
        # Specific PAN corrections: 
        # First 5 chars (letters)
        # Next 4 chars (digits): O->0, I->1, B->8, S->5
        # Last char (letter): 0->O, 1->I, 8->B, 5->S
        text = text.upper().replace(" ", "")
        text = re.sub(r'[^A-Z0-9]', '', text)
        return text

    # Strategy 1: From detected PAN_NUMBER region
    if "PAN_NUMBER" in text_dict:
        candidates = text_dict["PAN_NUMBER"]
        if isinstance(candidates, str):
            candidates = [candidates]
            
        for pan_text in candidates:
            pan_text = clean_candidate(pan_text)
            for pattern in pan_patterns:
                match = re.search(pattern, pan_text)
                if match:
                    return match.group(0)
    
    # Strategy 2: From all detected texts combined
    all_texts = []
    for key, val in text_dict.items():
        if isinstance(val, list):
            all_texts.extend(val)
        else:
            all_texts.append(str(val))
    
    combined = clean_candidate(" ".join(all_texts))
    for pattern in pan_patterns:
        match = re.search(pattern, combined)
        if match:
            return match.group(0)
    
    # Strategy 3: From full text (cleaned)
    # Using more aggressive replacements for full text search
    full_clean = full_text.upper().replace(" ", "")
    full_clean = full_clean.replace("O", "0").replace("I", "1") # Basic corrections
    full_clean = re.sub(r'[^A-Z0-9]', '', full_clean)
    
    for pattern in pan_patterns:
        match = re.search(pattern, full_clean)
        if match:
            return match.group(0)
            
    # Strategy 4: Search for PAN-like words in original text with specific char fixups
    # This helps when the PAN is isolated by spaces but has OCR errors
    words = full_text.split()
    for word in words:
        word = re.sub(r'[^A-Za-z0-9]', '', word.upper())
        if len(word) == 10:
            # Try to fix mixture of digits/letters
            # First 5 chars should be letters
            part1 = word[:5].replace('0', 'O').replace('1', 'I').replace('8', 'B').replace('5', 'S')
            # Next 4 chars should be digits
            part2 = word[5:9].replace('O', '0').replace('I', '1').replace('B', '8').replace('S', '5').replace('Z', '2')
            # Last char is a letter
            part3 = word[9].replace('0', 'O').replace('1', 'I').replace('8', 'B').replace('5', 'S')
            
            candidate = part1 + part2 + part3
            for pattern in pan_patterns:
                if re.match(pattern, candidate):
                    return candidate

    return ""

def extract_name(text: str, detected_data: Dict) -> str:
    BLACKLIST = {
        'INCOME', 'TAX', 'DEPARTMENT', 'GOVT', 'GOVERNMENT', 'INDIA', 
        'PERMANENT', 'ACCOUNT', 'NUMBER', 'CARD', 'MAHOTSAV', 'AMRIT','NAME'
    }

    def clean(name: str) -> str:
        name = re.sub(r'[^A-Za-z.\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip().upper()

        words = [w for w in name.split() if len(w.replace('.', '')) >= 2]
        
        if any(w in BLACKLIST for w in words):
            return ""

        if 1 <= len(words) <= 5:
            return " ".join(words)
        return ""

    if "NAME" in detected_data:
        candidates = detected_data["NAME"]
        if isinstance(candidates, str):
            candidates = [candidates]
        for c in candidates:
            result = clean(c)
            if result:
                return result

    pattern = r"NAME\s*[:\-]?\s*\n?\s*([A-Z][A-Za-z.\s]{2,})"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        result = clean(m.group(1))
        if result:
            return result

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    income_tax_found = False
    
    for i, line in enumerate(lines):
        line_upper = line.upper()
        
        if "INCOME" in line_upper and "TAX" in line_upper:
            income_tax_found = True
            continue 
        
        if income_tax_found:
            words = set(re.sub(r'[^A-Z]', ' ', line_upper).split())
            if len(words.intersection(BLACKLIST)) >= 1:
                continue

            result = clean(line)
            if result:
                return result

    return ""

def extract_father_name(text: str, detected_data: Dict, cardholder_name: str = "") -> str:
    
    # 1. Invalid Keywords (Standalone words to remove)
    INVALID_KEYWORDS = {
        'INCOME', 'TAX', 'DEPARTMENT', 'GOVERNMENT', 'INDIA',
        'PERMANENT', 'ACCOUNT', 'NUMBER', 'CARD', 'PAN',
        'SIGNATURE', 'PHOTO', 'DOB', 'DATE', 'BIRTH',
        'MALE', 'FEMALE', 'SEX', 'AGE', 'OF', 'AT', 'AND'
    }

    TRAILING_NOISE = {'DOB', 'DATE', 'BIRTH', 'BIRT', 'DAT', 'YY', 'YYYY'}

    def clean(name: str) -> str:
        # STEP 1: Fix specific OCR merger errors (The "VIRENDRAATKUMAR" fix)
        # Logic: If 'AT' appears with at least 3 letters on both sides, it's likely noise.
        # e.g., "VIRENDRAATKUMAR" -> "VIRENDRA KUMAR"
        # e.g., "CHATURVEDI" -> Stays "CHATURVEDI" (only 2 letters 'CH' before AT)
        name = re.sub(r'(?<=[A-Za-z]{3})AT(?=[A-Za-z]{3})', ' ', str(name))

        # STEP 2: Basic cleanup
        name = re.sub(r'[^A-Za-z.\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip().upper()

        words = name.split()

        # STEP 3: Remove trailing noise (e.g. "VIRENDRA KUMAR DOB")
        if words and words[-1] in TRAILING_NOISE:
            words.pop()

        cleaned_words = []
        for w in words:
            if w in INVALID_KEYWORDS: continue
            
            # Fix Stuttering (e.g. "VVIRENDRA" -> "VIRENDRA")
            if len(w) > 3 and w[0] == w[1]:
                w = w[1:]

            if len(w.replace('.', '')) >= 1: 
                cleaned_words.append(w)

        if 1 <= len(cleaned_words) <= 5:
            result = " ".join(cleaned_words)
            
            if cardholder_name and result == cardholder_name.upper():
                return ""
                
            if len(re.sub(r'[^A-Z]', '', result)) >= 2:
                return result

        return ""

    # --- Search Priority 1: Detected Data ---
    if "FATHER_NAME" in detected_data:
        candidates = detected_data["FATHER_NAME"]
        if isinstance(candidates, str): candidates = [candidates]
        for c in candidates:
            r = clean(c)
            if r: return r

    # --- Search Priority 2: Regex (Labeled) ---
    m = re.search(
        r"FATHER'?S?\s*NAME\s*[:\-]?\s*\n?\s*([A-Z][A-Za-z.\s]{2,})",
        text, re.IGNORECASE
    )
    if m:
        r = clean(m.group(1))
        if r: return r

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # --- Search Priority 3: Positional (Line after Cardholder Name) ---
    if cardholder_name:
        for i, line in enumerate(lines):
            # Aggressive normalize for comparison
            clean_line = re.sub(r'[^A-Z\s]', '', line.upper()).strip()
            clean_cardholder = re.sub(r'[^A-Z\s]', '', cardholder_name.upper()).strip()
            
            # Match even if the line has the "AT" glitch
            if clean_line == clean_cardholder or clean_line.replace('AT', '') == clean_cardholder:
                if i + 1 < len(lines):
                    candidate = lines[i + 1]
                    if not re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", candidate) and \
                       "FATHER" not in candidate.upper():
                        r = clean(candidate)
                        if r: return r

    # --- Search Priority 4: Keyword (Line after "FATHER") ---
    for i, line in enumerate(lines):
        if "FATHER" in line.upper():
            after = re.split(r"FATHER'?S?\s*NAME\s*[:\-]?", line, flags=re.I)
            if len(after) > 1 and len(after[1].strip()) > 2:
                r = clean(after[1])
                if r: return r
            
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if not re.search(r"\d", next_line):
                    r = clean(next_line)
                    if r: return r

    return ""

def extract_date_of_birth(text: str, detected_data: Dict) -> str:
    """Extract date of birth with improved pattern matching"""
    
    # Strategy 1: From detected DOB region
    if "DOB" in detected_data:
        candidates = detected_data["DOB"]
        if isinstance(candidates, str):
            candidates = [candidates]
            
        date_patterns = [
            r'(\d{2}[/-]\d{2}[/-]\d{4})',
            r'(\d{2}\s+\d{2}\s+\d{4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ]
        
        for dob_text in candidates:
            dob_text = str(dob_text).replace('O', '0').replace('o', '0')
            for pattern in date_patterns:
                match = re.search(pattern, dob_text)
                if match:
                    date_str = match.group(1)
                    if validate_date(date_str):
                        return normalize_date(date_str)
    
    # Strategy 2: Pattern matching from full text
    patterns = [
        # After DOB label
        r"(?:DATE\s+OF\s+BIRTH|DOB|Date\s+of\s+Birth)\s*[:/-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        r"(?:DATE\s+OF\s+BIRTH|DOB|Date\s+of\s+Birth)\s*[:/-]?\s*(\d{2}\s+\d{2}\s+\d{4})",
        # Any date pattern (fallback)
        r'(\d{2}[/-]\d{2}[/-]\d{4})',
        # With spaces
        r'(\d{2}\s+\d{2}\s+\d{4})',
    ]
    
    text_clean = text.replace('O', '0').replace('o', '0')
    
    for pattern in patterns:
        matches = re.finditer(pattern, text_clean, re.IGNORECASE)
        for match in matches:
            date_str = match.group(1)
            if validate_date(date_str):
                return normalize_date(date_str)
    
    return ""


def validate_date(date_str: str) -> bool:
    """Validate if string is a valid date"""
    try:
        date_str = date_str.replace(" ", "").replace("-", "/")
        parts = date_str.split("/")
        
        if len(parts) != 3:
            return False
        
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Handle 2-digit years
        if year < 100:
            year = 1900 + year if year > 50 else 2000 + year
        
        # Basic validation
        if not (1 <= month <= 12):
            return False
        if not (1 <= day <= 31):
            return False
        if not (1900 <= year <= datetime.now().year):
            return False
        
        # Validate actual date
        datetime(year, month, day)
        return True
    except:
        return False


def normalize_date(date_str: str) -> str:
    """Normalize date to DD/MM/YYYY format"""
    try:
        date_str = date_str.replace(" ", "").replace("-", "/")
        parts = date_str.split("/")
        
        if len(parts) == 3:
            day, month, year = parts[0].zfill(2), parts[1].zfill(2), parts[2]
            
            # Handle 2-digit years
            if len(year) == 2:
                year = "19" + year if int(year) > 50 else "20" + year
            
            return f"{day}/{month}/{year}"
    except:
        pass
    return date_str


def extract_card_type(pan_number: str) -> str:
    """Determine card type from PAN number"""
    if len(pan_number) >= 4:
        fourth_char = pan_number[3]
        card_types = {
            'P': 'Individual/Person',
            'C': 'Company',
            'H': 'Hindu Undivided Family (HUF)',
            'F': 'Firm/Partnership',
            'A': 'Association of Persons (AOP)',
            'T': 'Trust',
            'B': 'Body of Individuals (BOI)',
            'L': 'Local Authority',
            'J': 'Artificial Juridical Person',
            'G': 'Government',
        }
        return card_types.get(fourth_char, 'Unknown')
    return ""


def extract_profile_image(image: np.ndarray) -> Optional[str]:
    """Extract profile photo from PAN card"""
    try:
        h, w = image.shape[:2]
        
        # Try face detection first
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                x, y, w_face, h_face = max(faces, key=lambda f: f[2] * f[3])
                
                padding_x = int(w_face * 0.35)
                padding_y = int(h_face * 0.40)
                
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(w, x + w_face + padding_x)
                y2 = min(h, y + h_face + padding_y)
                
                photo = image[y1:y2, x1:x2]
                
                if photo.size > 0 and photo.shape[0] > 40 and photo.shape[1] > 40:
                    _, buffer = cv2.imencode('.png', photo)
                    photo_base64 = base64.b64encode(buffer).decode('utf-8')
                    return f"data:image/png;base64,{photo_base64}"
        except Exception as e:
            print(f"Face detection failed: {e}")
        
        # Fallback: extract from expected region
        x_start = int(w * 0.02)
        x_end = int(w * 0.35)
        y_start = int(h * 0.10)
        y_end = int(h * 0.75)
        
        photo_region = image[y_start:y_end, x_start:x_end]
        
        if photo_region.size > 0:
            _, buffer = cv2.imencode('.png', photo_region)
            photo_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/png;base64,{photo_base64}"
        
        return None
        
    except Exception as e:
        print(f"Error extracting profile image: {e}")
        return None


def extract_qr_code(image: np.ndarray) -> Optional[Dict[str, Any]]:
    """Extract QR code from PAN card with enhanced detection"""
    try:
        h, w = image.shape[:2]
        
        # Expanded search regions - QR can be in various positions
        search_regions = [
            # Right side regions (most common)
            (int(w * 0.60), w, 0, h),                    # Full right 40%
            (int(w * 0.65), w, 0, h),                    # Right 35%
            (int(w * 0.70), w, 0, int(h * 0.60)),       # Top-right
            (int(w * 0.70), w, int(h * 0.40), h),       # Bottom-right
            (int(w * 0.55), w, 0, h),                    # Wider right region
            # Center-right regions
            (int(w * 0.50), w, int(h * 0.20), int(h * 0.80)),  # Center-right vertical strip
        ]
        
        for x_start, x_end, y_start, y_end in search_regions:
            qr_region = image[y_start:y_end, x_start:x_end].copy()
            
            if qr_region.size == 0:
                continue
            
            # Method 1: Direct decode on original region
            try:
                decoded_objects = pyzbar.decode(qr_region)
                
                if decoded_objects:
                    qr = decoded_objects[0]
                    x, y, w_qr, h_qr = qr.rect
                    
                    # Extract with padding
                    padding = 15
                    y1 = max(0, y - padding)
                    x1 = max(0, x - padding)
                    y2 = min(qr_region.shape[0], y + h_qr + padding)
                    x2 = min(qr_region.shape[1], x + w_qr + padding)
                    
                    qr_image = qr_region[y1:y2, x1:x2]
                    
                    if qr_image.size > 0 and qr_image.shape[0] > 20 and qr_image.shape[1] > 20:
                        _, buffer = cv2.imencode('.png', qr_image)
                        qr_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        return {
                            "qr_code_image": f"data:image/png;base64,{qr_base64}",
                            "qr_data": qr.data.decode('utf-8') if qr.data else None,
                            "qr_type": qr.type,
                            "qr_found": True,
                            "qr_dimensions": {
                                "width": qr_image.shape[1],
                                "height": qr_image.shape[0]
                            }
                        }
            except Exception as e:
                pass
            
            # Method 2: Try with preprocessing
            try:
                gray = cv2.cvtColor(qr_region, cv2.COLOR_BGR2GRAY)
                
                # Multiple preprocessing approaches
                preprocessed_versions = []
                
                # 1. Original grayscale
                preprocessed_versions.append(gray)
                
                # 2. Gaussian blur
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                preprocessed_versions.append(blurred)
                
                # 3. Median blur
                median = cv2.medianBlur(gray, 5)
                preprocessed_versions.append(median)
                
                # 4. Binary threshold
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                preprocessed_versions.append(binary)
                
                # 5. Otsu threshold
                _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                preprocessed_versions.append(otsu)
                
                # 6. Adaptive threshold
                adaptive = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                preprocessed_versions.append(adaptive)
                
                # 7. Inverted binary
                _, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                preprocessed_versions.append(binary_inv)
                
                # 8. CLAHE enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                preprocessed_versions.append(enhanced)
                
                # 9. Sharpened
                kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
                preprocessed_versions.append(sharpened)
                
                # Try decoding each preprocessed version
                for processed in preprocessed_versions:
                    decoded_objects = pyzbar.decode(processed)
                    
                    if decoded_objects:
                        qr = decoded_objects[0]
                        x, y, w_qr, h_qr = qr.rect
                        
                        padding = 15
                        y1 = max(0, y - padding)
                        x1 = max(0, x - padding)
                        y2 = min(qr_region.shape[0], y + h_qr + padding)
                        x2 = min(qr_region.shape[1], x + w_qr + padding)
                        
                        qr_image = qr_region[y1:y2, x1:x2]
                        
                        if qr_image.size > 0 and qr_image.shape[0] > 20 and qr_image.shape[1] > 20:
                            _, buffer = cv2.imencode('.png', qr_image)
                            qr_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            return {
                                "qr_code_image": f"data:image/png;base64,{qr_base64}",
                                "qr_data": qr.data.decode('utf-8') if qr.data else None,
                                "qr_type": qr.type,
                                "qr_found": True,
                                "qr_dimensions": {
                                    "width": qr_image.shape[1],
                                    "height": qr_image.shape[0]
                                }
                            }
            except Exception as e:
                pass
        
        # Method 3: Contour-based fallback (if QR not decoded but visually present)
        try:
            # Focus on right side
            fallback_region = image[0:h, int(w * 0.60):w].copy()
            
            if fallback_region.size == 0:
                return {"qr_code_image": None, "qr_found": False}
            
            gray = cv2.cvtColor(fallback_region, cv2.COLOR_BGR2GRAY)
            
            # Edge detection for QR patterns
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)
            
            # Dilate to connect edges
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_qr_candidates = []
            region_area = fallback_region.shape[0] * fallback_region.shape[1]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                min_area = region_area * 0.03  # At least 3% of region
                max_area = region_area * 0.70  # At most 
                
                if min_area < area < max_area:
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    aspect_ratio = float(w_box) / h_box if h_box > 0 else 0
                    
                    # QR codes are square (aspect ratio close to 1.0)
                    if 0.7 <= aspect_ratio <= 1.4 and w_box > 40 and h_box > 40:
                        # Calculate squareness score
                        squareness = 1.0 - abs(1.0 - aspect_ratio)
                        
                        # Check if it has QR-like features (count corners)
                        peri = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                        
                        valid_qr_candidates.append({
                            'contour': contour,
                            'area': area,
                            'bbox': (x, y, w_box, h_box),
                            'squareness': squareness,
                            'corners': len(approx)
                        })
            
            if valid_qr_candidates:
                valid_qr_candidates.sort(
                    key=lambda c: (c['squareness'], c['area']), 
                    reverse=True
                )
                
                # Take the best candidate
                best = valid_qr_candidates[0]
                x, y, w_box, h_box = best['bbox']
                
                # Extract with padding
                padding = 15
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(fallback_region.shape[1], x + w_box + padding)
                y2 = min(fallback_region.shape[0], y + h_box + padding)
                
                qr_image = fallback_region[y1:y2, x1:x2]
                
                if qr_image.size > 0 and qr_image.shape[0] > 30 and qr_image.shape[1] > 30:
                    _, buffer = cv2.imencode('.png', qr_image)
                    qr_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    return {
                        "qr_code_image": f"data:image/png;base64,{qr_base64}",
                    }
        except Exception as e:
            print(f"Contour-based QR detection failed: {e}")
        
        return {"qr_code_image": None, "qr_found": False}
    
    except Exception as e:
        print(f"Error extracting QR code: {e}")
        traceback.print_exc()
        return {"qr_code_image": None, "qr_found": False, "error": str(e)}

def extract_signature(image: np.ndarray) -> Optional[str]:
    """Extract signature from PAN card"""
    try:
        h, w = image.shape[:2]
        
        x_start = int(w * 0.35)
        x_end = int(w * 0.65)
        y_start = int(h * 0.70)
        y_end = h
        
        sig_region = image[y_start:y_end, x_start:x_end].copy()
        
        if sig_region.size > 0:
            _, buffer = cv2.imencode('.png', sig_region)
            sig_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/png;base64,{sig_base64}"
        
        return None
        
    except Exception as e:
        print(f"Error extracting signature: {e}")
        return None


def process_pan_card(image_path: str) -> Dict[str, Any]:
    """Main function to process PAN card and extract all information"""
    if model is None or reader is None:
        raise Exception("YOLO model or EasyOCR reader not initialized")
    
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Failed to read image")
    
    h, w, _ = img.shape
    
    # Extract visual elements
    profile_image_base64 = extract_profile_image(img.copy())
    qr_code_info = extract_qr_code(img.copy())
    signature_base64 = extract_signature(img.copy())
    
    # Run YOLO detection
    results = model(img)
    
    detected_regions = {}
    
    # Class name mapping
    class_mapping = {
        'dob': 'DOB',
        'father-s name': 'FATHER_NAME',
        'father_name': 'FATHER_NAME',
        "father's name": 'FATHER_NAME',
        'name': 'NAME',
        'pan number': 'PAN_NUMBER',
        'pan_number': 'PAN_NUMBER',
    }
    
    # Extract text from detected regions
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label_name = result.names[cls_id].lower()
            
            if conf > 0.25:  # Lower threshold to catch more detections
                # Map the label to standard keys
                mapped_label = class_mapping.get(label_name, label_name.upper())
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Add padding to ROI for better OCR
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                roi = img[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                
                # Extract text using robust method
                extracted_text = extract_text_robust(roi, reader)
                
                if extracted_text:
                    if mapped_label not in detected_regions:
                        detected_regions[mapped_label] = []
                    detected_regions[mapped_label].append(extracted_text)
    
    # Full text extraction with multiple methods
    print("Extracting full text from image...")
    full_text_list = reader.readtext(img, detail=0, paragraph=False)
    full_text = " ".join(full_text_list)
    
    # Try enhanced version
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_text_list = reader.readtext(enhanced, detail=0, paragraph=False)
        enhanced_text = " ".join(enhanced_text_list)
        
        # Use the longer text
        if len(enhanced_text) > len(full_text):
            full_text = enhanced_text
    except Exception as e:
        print(f"Enhanced text extraction failed: {e}")
    
    print(f"Full text extracted: {full_text[:200]}...")
    
    # Extract fields
    pan_number = clean_text(extract_pan_number(detected_regions, full_text))
    name = clean_text(extract_name(full_text, detected_regions))
    father_name = clean_text(extract_father_name(full_text, detected_regions, name))
    dob = extract_date_of_birth(full_text, detected_regions)
    card_type = extract_card_type(pan_number)
    
    is_valid = bool(pan_number) and len(pan_number) == 10
    
    # Prepare clean output
    clean_detected_regions = {}
    for key, val in detected_regions.items():
        if isinstance(val, list) and val:
            # Combine all detections for this field
            clean_detected_regions[key] = " | ".join(val)
        elif isinstance(val, str):
            clean_detected_regions[key] = val
        else:
            clean_detected_regions[key] = ""
    
    # Calculate completeness score
    fields_found = sum([
        bool(pan_number),
        bool(name),
        bool(father_name),
        bool(dob),
    ])
    completeness_score = (fields_found / 4) * 100
    
    result = {
        "pan_number": pan_number,
        "name": name,
        "father_name": father_name,
        "date_of_birth": dob,
        "card_type": card_type,
        "profile_image": profile_image_base64,
        "qr_code_info": qr_code_info,
        "signature": signature_base64,
        "is_valid_pan": is_valid,
        "completeness_score": round(completeness_score, 2),
        "confidence_level": "high" if completeness_score >= 75 else "medium" if completeness_score >= 50 else "low",
        "detected_regions": clean_detected_regions,
        "full_extracted_text": full_text[:500],  # First 500 chars for debugging
        "image_dimensions": {
            "width": w,
            "height": h
        },
        "total_detections": sum(len(r.boxes) for r in results) if results else 0,
    }
    
    return result
