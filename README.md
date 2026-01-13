GROUP: PPG
GROUPMEMBERS: 1. NORAMIRA HUSNA BINTI NORKHAIRANI 2214496
2. NUR SHADATUL BALQISH BINTI SAHRUNIZAM 2212064
3. KHALISAH AMANI BINTI YAHYA AZMI 2218184

# Smart Recovery ALPR System (Malaysian License Plates)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Project Overview
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
This project implements an enhanced Automatic License Plate Recognition (ALPR) system designed to handle challenging environmental conditions such as low light, glare, and noise. 

While standard ALPR systems often fail in dark conditions, this project introduces an **Adaptive Multi-Pass Inference Logic** (Smart Recovery). It acts as a "Smart Double-Check" system:
1.  **Pass 1:** Attempts detection on the raw image (preserving daytime quality).
2.  **Smart Trigger:** Checks if a license plate was successfully read.
3.  **Pass 2 (Recovery):** If Pass 1 fails, the system applies a **"Night Vision" filter (CLAHE)** and re-scans the image to recover the missed plate.

---
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Setup Instructions
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 1. Environment Setup
The system requires **Python 3.9**. You can set up the environment using `conda` or `pip`.

**Option A: Using Conda (Recommended)**
```
conda create -n alpr_project python=3.9
conda activate alpr_project
```
**Option B: Installing Dependencies Install the required packages matching the developed environment:**
```
pip install opencv-python==4.12.0.88
pip install numpy==2.0.2
pip install ultralytics==8.3.246
pip install easyocr==1.7.2
pip install torch torchvision
pip install pandas==2.3.3
```
Note: tkinter is usually included with Python. If you encounter an error regarding tk, ensure your Python installation includes Tcl/Tk support.

How to Run the System

First run the Individual Systems
- If you wish to run the systems separately:
- Ensure your testpic_combine folder has images.

1. For Baseline Only: Navigate to model_base/ and run python main_base.py.
2. For Smart System Only: Navigate to model_optimised2/ and run python main_optimised2.py.

Running the Live Comparison (Recommended)
This script runs the Baseline System and the Smart Recovery System side-by-side to demonstrate the improvements in real-time.

1.Open the folder in VS Code.
2.Ensure your testpic_combine folder has images.
3.Run the comparison script:
```
python main_comparison.py
````
What you will see:
- Left Window: The Baseline model running standard detection.
- Right Window: The Optimised model running the Smart Recovery logic.
- Terminal Output: A final "Winner Report" showing the accuracy score and execution speed of both systems.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Methodology: The Innovation
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The core innovation is the "Smart Recovery" Logic.

1. The Problem (Baseline)
Standard detectors (YOLO) treats every image the same.

- Daytime: Works perfectly.
- Nighttime/Dark: Fails to detect the plate features; output is zero.

2. The Solution (Adaptive Logic)
We implemented a dynamic feedback loop that acts like a human turning on a flashlight when it's too dark to read.

3. The Algorithm Flow
    1. Input Image arrives.
    2. Pass 1 (Standard): System attempts detection.
      - If text is found: STOP & SAVE (Keeps image clean).
      - If NO text is found: TRIGGER RECOVERY.
    3.Night Recovery Mode:
      - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L-channel (Lightness).
      - This "brightens" the details without washing out the image like simple brightness adjustments.
    4. Pass 2 (Retry): System scans the enhanced image.
    5. Result: Plates that were invisible to the Base model are successfully detected and read.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Performance Summary
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

| Metric | Baseline System | Smart Recovery System (Ours) |
| :--- | :--- | :--- |
| **Logic** | Static (Single Pass) | Adaptive (Multi-Pass) |
| **Detection Accuracy** | ~80% | **~84%** |
| **Dark Condition Handling** | Fails | **Succeeds (Auto-Recovery)** |
| **Trade-off** | Faster (~0.73 FPS) | Slightly Slower (~0.65 FPS) |

*The Smart System sacrifices a small amount of speed (milliseconds) to gain significantly higher reliability in security-critical scenarios.*
