import cv2
import numpy as np
import os
import glob
import re
import csv
import time
import shutil  # Added for deleting old folders
from ultralytics import YOLO
import easyocr
import tkinter as tk 

# --- CONFIGURATION ---
INPUT_FOLDER = 'testpic_combine'
MAIN_OUTPUT_FOLDER = 'base_vs_optimised'

# SUB-FOLDERS
VISUALS_FOLDER = os.path.join(MAIN_OUTPUT_FOLDER, 'comparison_visuals')
CROPS_BASE = os.path.join(MAIN_OUTPUT_FOLDER, 'crops_base')
CROPS_SMART = os.path.join(MAIN_OUTPUT_FOLDER, 'crops_smart')
OUTPUT_CSV = os.path.join(MAIN_OUTPUT_FOLDER, 'comparison_report.csv')

MODEL_PATH = os.path.join('model_base', 'best.pt') 
EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

# --- AUTO-DETECT SCREEN SIZE ---
try:
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    root.destroy()
    DISPLAY_WIDTH = int((screen_width / 2) - 40) 
    print(f"Detected Screen Width: {screen_width}px. Split View Width: {DISPLAY_WIDTH}px")
except:
    DISPLAY_WIDTH = 640 
    print("Could not detect screen size. Using default 640px.")

# YOLO SETTINGS
YOLO_IMG_SIZE = 1280 
CONFIDENCE_THRESHOLD = 0.25 
PADDING_PCT = 0.10 

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

# --- CLEANUP & SETUP ---
print(f"Cleaning up previous results in '{MAIN_OUTPUT_FOLDER}'...")
if os.path.exists(MAIN_OUTPUT_FOLDER):
    shutil.rmtree(MAIN_OUTPUT_FOLDER) # Deletes the whole folder tree

# Re-create fresh folders
os.makedirs(MAIN_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(VISUALS_FOLDER, exist_ok=True)
os.makedirs(CROPS_BASE, exist_ok=True)
os.makedirs(CROPS_SMART, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Could not find model at: {MODEL_PATH}")
    exit()

model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=True) 
ALLOWED_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

stats = {
    "total_images": 0, "base_detected": 0, "smart_detected": 0,
    "base_time": 0, "smart_time": 0
}

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
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)) 
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
                    is_brand = True; break
            if not is_brand:
                final_parts.append(clean_text)
    return " ".join(final_parts)

def draw_hud_on_resized(img, detected_items, system_name, color_theme):
    h, w = img.shape[:2]
    
    # FIXED SIZES for Laptop Screen
    THUMB_W = 120
    THUMB_H = 40
    ROW_HEIGHT = 50
    FONT_SIZE = 0.6 
    MARGIN = 10
    HEADER_H = 40
    
    # 1. Header Background 
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, HEADER_H), (0,0,0), -1)
    
    # 2. List Background
    list_len = len(detected_items)
    if list_len > 0:
        bg_height = HEADER_H + (list_len * ROW_HEIGHT) + 10
        bg_width = MARGIN + THUMB_W + 150 
        cv2.rectangle(overlay, (0, HEADER_H), (bg_width, bg_height), (0, 0, 0), -1)

    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # 3. Text
    cv2.putText(img, system_name, (MARGIN, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_theme, 2)

    if list_len == 0:
        cv2.putText(img, "NO PLATE", (MARGIN, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
    else:
        for i, (text, crop_img) in enumerate(detected_items):
            y_pos = HEADER_H + (i * ROW_HEIGHT) + 10
            try:
                thumb_resized = cv2.resize(crop_img, (THUMB_W, THUMB_H))
                img[y_pos : y_pos + THUMB_H, MARGIN : MARGIN + THUMB_W] = thumb_resized
                cv2.rectangle(img, (MARGIN, y_pos), (MARGIN + THUMB_W, y_pos + THUMB_H), (200, 200, 200), 1)
            except: pass

            text_x = MARGIN + THUMB_W + 10
            text_y = y_pos + 28
            cv2.putText(img, f"{i+1}. {text}", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 255, 255), 2)
            cv2.putText(img, f"{i+1}. {text}", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, color_theme, 1)

def process_image_logic(img, mode="Base"):
    h, w = img.shape[:2]
    start_t = time.perf_counter()
    
    # PASS 1: STANDARD
    results = model(img, imgsz=YOLO_IMG_SIZE, verbose=False, conf=CONFIDENCE_THRESHOLD)
    found_detections = []
    
    if len(results[0].boxes) > 0:
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                px1, py1, px2, py2 = map(int, box)
                x1, y1, x2, y2 = expand_box(px1, py1, px2, py2, w, h)
                plate_crop = img[y1:y2, x1:x2]
                
                processed_plate = preprocess_plate_for_ocr(plate_crop)
                ocr_raw = reader.readtext(processed_plate, allowlist=ALLOWED_CHARS, mag_ratio=1.5, paragraph=False)
                full_text = malaysian_clean_and_sort(ocr_raw)

                if len(full_text) > 1:
                    found_detections.append((full_text, plate_crop, x1, y1, x2, y2))

    # PASS 2: SMART LOGIC
    if mode == "Smart" and len(found_detections) == 0:
        img_night = apply_night_vision(img)
        results_night = model(img_night, imgsz=YOLO_IMG_SIZE, verbose=False, conf=CONFIDENCE_THRESHOLD)
        
        if len(results_night[0].boxes) > 0:
            for result in results_night:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    px1, py1, px2, py2 = map(int, box)
                    x1, y1, x2, y2 = expand_box(px1, py1, px2, py2, w, h)
                    plate_crop = img[y1:y2, x1:x2] 
                    
                    processed_plate = preprocess_plate_for_ocr(plate_crop)
                    ocr_raw = reader.readtext(processed_plate, allowlist=ALLOWED_CHARS, mag_ratio=1.5, paragraph=False)
                    full_text = malaysian_clean_and_sort(ocr_raw)

                    if len(full_text) > 1:
                        found_detections.append((full_text, plate_crop, x1, y1, x2, y2))
    
    end_t = time.perf_counter()
    duration = end_t - start_t
    return found_detections, duration

# --- MAIN LOOP ---
image_files = []
for ext in EXTENSIONS:
    image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

stats["total_images"] = len(image_files)

csv_file = open(OUTPUT_CSV, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Filename', 'Base Found?', 'Base Text', 'Smart Found?', 'Smart Text', 'Smart Improvement?'])

print(f"Processing {stats['total_images']} images...")

cv2.namedWindow("Comparison", cv2.WINDOW_NORMAL)

for idx, img_path in enumerate(image_files, start=1):
    img_orig = cv2.imread(img_path)
    if img_orig is None: continue
    
    filename = os.path.basename(img_path)
    
    # 1. RUN BASE
    img_base = img_orig.copy()
    base_res, base_dur = process_image_logic(img_base, mode="Base")
    stats["base_time"] += base_dur
    if len(base_res) > 0: stats["base_detected"] += 1
    
    base_texts = []
    for i, (text, crop, x1, y1, x2, y2) in enumerate(base_res):
        cv2.rectangle(img_base, (x1, y1), (x2, y2), (0, 255, 255), 4) 
        base_texts.append(text)
        cv2.imwrite(os.path.join(CROPS_BASE, f"base_{filename}_{i}.jpg"), crop)
    
    # 2. RUN SMART
    img_smart = img_orig.copy()
    smart_res, smart_dur = process_image_logic(img_smart, mode="Smart")
    stats["smart_time"] += smart_dur
    if len(smart_res) > 0: stats["smart_detected"] += 1
    
    smart_texts = []
    for i, (text, crop, x1, y1, x2, y2) in enumerate(smart_res):
        cv2.rectangle(img_smart, (x1, y1), (x2, y2), (0, 255, 0), 4) 
        smart_texts.append(text)
        cv2.imwrite(os.path.join(CROPS_SMART, f"smart_{filename}_{i}.jpg"), crop)

    # 3. RESIZE
    h, w = img_orig.shape[:2]
    scale = DISPLAY_WIDTH / w
    new_h = int(h * scale)
    
    display_base = cv2.resize(img_base, (DISPLAY_WIDTH, new_h))
    display_smart = cv2.resize(img_smart, (DISPLAY_WIDTH, new_h))

    # 4. DRAW HUD
    draw_hud_on_resized(display_base, [(t, c) for t,c,_,_,_,_ in base_res], "BASE MODEL", (0, 255, 255))
    draw_hud_on_resized(display_smart, [(t, c) for t,c,_,_,_,_ in smart_res], "OPTIMISED 2.0", (0, 255, 0))
    
    # 5. STACK & SAVE
    combined_display = np.hstack((display_base, display_smart))
    visual_path = os.path.join(VISUALS_FOLDER, f"comp_{filename}")
    cv2.imwrite(visual_path, combined_display)
    
    # 6. LOG
    b_found = "Y" if base_res else "N"
    b_txt = " | ".join(base_texts)
    s_found = "Y" if smart_res else "N"
    s_txt = " | ".join(smart_texts)
    improvement = "YES" if (not base_res and smart_res) else ""
    
    csv_writer.writerow([filename, b_found, b_txt, s_found, s_txt, improvement])
    print(f"[{idx}] {filename} -> Base: {b_found} | Smart: {s_found} {improvement}")

    cv2.imshow("Comparison", combined_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- REPORT ---
print("\n" + "="*60)
print("              FINAL PERFORMANCE COMPARISON")
print("="*60)
print(f" Results saved to folder: {MAIN_OUTPUT_FOLDER}")
print("-" * 60)
print(f" {'METRIC':<20} | {'BASE MODEL':<15} | {'OPTIMISED 2.0':<15}")
print("-" * 60)

base_rate = (stats['base_detected'] / stats['total_images']) * 100
smart_rate = (stats['smart_detected'] / stats['total_images']) * 100
base_fps = stats['total_images'] / stats['base_time']
smart_fps = stats['total_images'] / stats['smart_time']

print(f" {'Success Rate':<20} | {base_rate:<14.2f}% | {smart_rate:<14.2f}%")
print(f" {'Total Plates':<20} | {stats['base_detected']:<15} | {stats['smart_detected']:<15}")
print(f" {'Avg Speed (FPS)':<20} | {base_fps:<15.2f} | {smart_fps:<15.2f}")
print("="*60)

if smart_rate > base_rate:
    print(f" 🏆 WINNER: OPTIMISED 2.0 (Recovered {stats['smart_detected'] - stats['base_detected']} extra plates)")
elif smart_rate < base_rate:
    print(f" 🏆 WINNER: BASE MODEL (Faster and more accurate)")
else:
    print(f" 🤝 DRAW (Both systems performed equally)")
print("="*60)

csv_file.close()
cv2.destroyAllWindows()