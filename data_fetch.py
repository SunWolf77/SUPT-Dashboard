import cv2
import pytesseract
from PIL import Image
from io import BytesIO
import numpy as np

def get_tomsk_schumann_power_ocr():
    """Download Tomsk live chart, crop, OCR amplitude of mode 1."""
    url = "https://sosrff.tsu.ru/new/sch.png"  # Live amplitude chart
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Crop to approximate recent amplitude region (adjust these coords if layout changes)
        # Typical: bottom-right shows latest values
        height, width = gray.shape
        crop = gray[int(height * 0.7):int(height * 0.95), int(width * 0.6):int(width * 0.95)]

        # Enhance contrast for better OCR
        crop = cv2.convertScaleAbs(crop, alpha=1.5, beta=0)

        # OCR with digit config
        text = pytesseract.image_to_string(crop, config='--psm 6 digits')
        # Clean and take first reasonable number (usually mode 1 power)
        numbers = [int(s) for s in text.split() if s.isdigit() and 5 < int(s) < 200]
        power = numbers[0] if numbers else 20.0  # Default fallback

        return float(power)
    except Exception as e:
        print(f"Tomsk OCR failed: {e}")
        return 20.0  # Safe default
