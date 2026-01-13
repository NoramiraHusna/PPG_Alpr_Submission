import cv2
import numpy as np
import os
import glob
import re
import csv
import time
from ultralytics import YOLO
import easyocr

# --- CONFIGURATION ---
INPUT_FOLDER = 'testpic_combine'
CROP_SAVE_FOLDER = os.path.join('model_smart_final', 'crops') 
OUTPUT_CSV = os.path.join('model_smart_final', 'data_smart_v2.csv')

# *** USING BASE MODEL ***
MODEL_PATH = os.path.join('model_base', 'best.pt') 

EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
DISPLAY_WIDTH = 1024        

YOLO_IMG_SIZE = 1280 
CONFIDENCE_THRESHOLD = 0.25 
PADDING_PCT = 0.10 

# --- BLOCKLIST ---
BLOCKLIST = [
    "TOYOTA", "HONDA", "NISSAN", "MAZDA", "PROTON", "PERODUA", 
    "ISUZU", "HINO", "VOLVO", "SCANIA", "MITSUBISHI", "SUZUKI", 
    "DAIHATSU", "BMW", "MERCEDES", "BENZ", "AUDI", "FORD", 
    "LEXUS", "KIA", "HYUNDAI", "CHERY", "BYD", "TESLA", 
    "HILUX", "TRITON", "RANGER", "DMAX", "NAVARA", 
    "ALPHARD", "VELLFIRE", "MYVI", "SAGA", "BEZZA", "AXIA",
    "POS", "LAJU", "EXPRESS", "LOGISTICS", "TURBO", "INTERCOOLER",
    "V6", "4X4", "AWD", "ABS", "SRS", "NLR", "NMR", "ELF"
]

# --- SETUP ---
os.makedirs(CROP_SAVE_FOLDER, exist_ok=True)
if not os.path.exists('model_smart_final'):
    os.makedirs('model_smart_final')

print(f"Initializing SMART SYSTEM V2 (Result-Based Trigger)...")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Could not find model at: {MODEL_PATH}")
    exit()

model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=True) 
ALLOWED_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# --- METRICS ---
total_images = 0
total_plates_detected = 0 
total_plates_read = 0     
total_processing_time = 0

# --- HELPER FUNCTIONS ---

def apply_night_vision(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def preprocess_plate_for_ocr(plate_img):
    if plate_img.size == 0: return plate_img
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

def expand_box(x1, y1, x2, y2, img_w, img_h):
    width = x2 - x1
    height = y2 - y1
    pad_w = int(width * PADDING_PCT)
    pad_h = int(height * PADDING_PCT)
    return (max(0, x1 - pad_w), max(0, y1 - pad_h), min(img_w, x2 + pad_w), min(img_h, y2 + pad_h))

def malaysian_clean_and_sort(ocr_results):
    if not ocr_results: return ""
    sorted_res = sorted(ocr_results, key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))
    final_parts = []
    for res in sorted_res:
        text = res[1].upper()
        conf = res[2]
        clean_text = re.sub(r'[^A-Z0-9]', '', text)
        if conf > 0.2 and len(clean_text) > 0:
            is_brand = False
            for blocked_word in BLOCKLIST:
                if blocked_word in clean_text:
                    is_brand = True
                    break
            if not is_brand:
                final_parts.append(clean_text)
    return " ".join(final_parts)

def draw_hud(img, detected_items, time_sec, mode_label=""):
    h, w = img.shape[:2]
    scale = max(0.6, h / 1080.0)

    THUMB_W = int(450 * scale)
    THUMB_H = int(100 * scale)
    ROW_HEIGHT = int(140 * scale)
    FONT_SIZE = 2.0 * scale
    FONT_THICK = max(2, int(4 * scale))
    MARGIN = int(30 * scale)
    
    # TOP RIGHT: TIME
    time_text = f"{time_sec:.3f} s"
    box_w = int(450 * scale)
    box_h = int(80 * scale)
    cv2.rectangle(img, (w - box_w, 0), (w, box_h), (0, 0, 0), -1)
    text_y = int(60 * scale)
    cv2.putText(img, time_text, (w - int(box_w * 0.9), text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE * 0.8, (0, 0, 255), FONT_THICK)

    # TOP LEFT: LIST
    list_len = len(detected_items)
    bg_height = max(ROW_HEIGHT, list_len * ROW_HEIGHT)
    bg_width = int(THUMB_W * 2.5)
    cv2.rectangle(img, (0, 0), (bg_width, bg_height), (0, 0, 0), -1)

    # Mode Label
    if mode_label:
        color = (0, 255, 0) if "Standard" in mode_label else (0, 0, 255)
        cv2.putText(img, mode_label, (MARGIN, h - MARGIN), 
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, color, FONT_THICK)

    if not detected_items:
        cv2.putText(img, "SEARCHING...", (MARGIN, int(80 * scale)), 
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (200, 200, 200), FONT_THICK)
    else:
        for i, (text, crop_img) in enumerate(detected_items):
            y_pos = (i * ROW_HEIGHT)
            try:
                thumb = cv2.resize(crop_img, (THUMB_W, THUMB_H))
                y_offset = int((ROW_HEIGHT - THUMB_H) / 2)
                draw_y = y_pos + y_offset
                img[draw_y : draw_y + THUMB_H, MARGIN : MARGIN + THUMB_W] = thumb
                cv2.rectangle(img, (MARGIN, draw_y), (MARGIN + THUMB_W, draw_y + THUMB_H), (255, 255, 255), max(1, int(2*scale)))
            except: pass

            text_x = MARGIN + THUMB_W + int(30 * scale)
            text_y = y_pos + int(ROW_HEIGHT * 0.65)
            display_text = f"{i+1}. {text}"
            cv2.putText(img, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 255, 255), FONT_THICK)

def process_detection_pass(img, results):
    """ Helper to process YOLO results and return list of (text, crop) """
    detections = []
    if len(results[0].boxes) > 0:
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                px1, py1, px2, py2 = map(int, box)
                x1, y1, x2, y2 = expand_box(px1, py1, px2, py2, img.shape[1], img.shape[0])
                
                plate_crop = img[y1:y2, x1:x2]
                processed_plate = preprocess_plate_for_ocr(plate_crop)
                ocr_raw = reader.readtext(processed_plate, allowlist=ALLOWED_CHARS, mag_ratio=1.5, paragraph=False)
                full_text = malaysian_clean_and_sort(ocr_raw)

                if len(full_text) > 1:
                    detections.append((full_text, plate_crop, x1, y1, x2, y2))
    return detections

# --- START PROCESSING ---
image_files = []
for ext in EXTENSIONS:
    image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

total_images = len(image_files)
print(f"Starting processing on {total_images} images...")

csv_file = open(OUTPUT_CSV, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['No.', 'Original Filename', 'Detections Count', 'Detected Text(s)', 'Processing Time (ms)', 'Method Used'])

for idx, img_path in enumerate(image_files, start=1):
    img = cv2.imread(img_path)
    if img is None: continue
    h, w = img.shape[:2]
    filename = os.path.basename(img_path)

    frame_detections = [] 
    detection_method = "Standard"
    start_time = time.perf_counter()

    # --- PASS 1: STANDARD DETECTION ---
    results = model(img, imgsz=YOLO_IMG_SIZE, verbose=False, conf=CONFIDENCE_THRESHOLD)
    raw_detections = process_detection_pass(img, results)
    
    # --- LOGIC FIX: Check if we actually READ anything ---
    if len(raw_detections) > 0:
        # Success on Pass 1
        for (text, crop, x1, y1, x2, y2) in raw_detections:
            frame_detections.append((text, crop))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3) # Green
            
            crop_filename = f"crop_{idx}_{len(frame_detections)}_{filename}"
            cv2.imwrite(os.path.join(CROP_SAVE_FOLDER, crop_filename), crop)

    else:
        # --- PASS 2: NIGHT RECOVERY (Triggered because Pass 1 found NO TEXT) ---
        detection_method = "Night Recovery"
        img_night = apply_night_vision(img) 
        results_night = model(img_night, imgsz=YOLO_IMG_SIZE, verbose=False, conf=CONFIDENCE_THRESHOLD)
        raw_detections_night = process_detection_pass(img, results_night) # Note: Passing original img for cropping is safer
        
        if len(raw_detections_night) > 0:
            for (text, crop, x1, y1, x2, y2) in raw_detections_night:
                frame_detections.append((text, crop))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3) # Red
                
                crop_filename = f"crop_{idx}_{len(frame_detections)}_{filename}"
                cv2.imwrite(os.path.join(CROP_SAVE_FOLDER, crop_filename), crop)

    end_time = time.perf_counter()
    duration_sec = end_time - start_time
    duration_ms = duration_sec * 1000
    total_processing_time += duration_sec

    draw_hud(img, frame_detections, duration_sec, detection_method if len(frame_detections) > 0 else "")

    if len(frame_detections) > 0:
        total_plates_detected += 1
        total_plates_read += 1

    all_texts = " | ".join([item[0] for item in frame_detections])
    csv_writer.writerow([idx, filename, len(frame_detections), all_texts, f"{duration_ms:.2f}", detection_method])
    
    print(f"[{idx}/{total_images}] {filename} -> {detection_method}: {len(frame_detections)} plates | {duration_sec:.3f}s")

    scale = DISPLAY_WIDTH / w
    new_h = int(h * scale)
    display_img = cv2.resize(img, (DISPLAY_WIDTH, new_h))
    
    cv2.imshow("OPTIMISED2.0", display_img)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

avg_fps = total_images / total_processing_time if total_processing_time > 0 else 0
det_rate = (total_plates_detected / total_images) * 100 if total_images > 0 else 0
read_rate = (total_plates_read / total_images) * 100 if total_images > 0 else 0

print("\n" + "=" * 40)
print(f" OPTIMISED2.0 PERFORMANCE REPORT")
print("=" * 40)
print(f" CSV Saved to    : {OUTPUT_CSV}")
print(f" Total Images    : {total_images}")
print(f" Average Speed   : {avg_fps:.2f} FPS")
print("-" * 40)
print(f" Plate Detection Rate  : {det_rate:.2f} %")
print(f" Plate Reading Rate    : {read_rate:.2f} %")
print("=" * 40)

csv_file.close()
cv2.destroyAllWindows()