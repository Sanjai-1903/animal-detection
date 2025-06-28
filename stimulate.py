import cv2
import numpy as np
import os
from glob import glob

# === STEP 1: Gamma Correction ===
def gamma_correction(img, gamma=1.1):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

# === STEP 2: Simulate Custom Thermal Image ===
def simulate_custom_thermal(img, brightness_weight=1.2, red_weight=1.5, green_weight=0.7, blue_weight=0.3):
    img = gamma_correction(img, gamma=1.1)
    img = img.astype(np.float32) / 255.0
    b, g, r = cv2.split(img)

    thermal_float = (r * red_weight + g * green_weight + b * blue_weight) * brightness_weight
    thermal_norm = cv2.normalize(thermal_float, None, 0, 1, cv2.NORM_MINMAX)
    thermal_gray = (thermal_norm * 255).astype(np.uint8)

    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    thermal_gray = clahe.apply(thermal_gray)

    # Use inferno colormap for better visual balance
    thermal_colored = cv2.applyColorMap(thermal_gray, cv2.COLORMAP_INFERNO)

    return thermal_colored, thermal_gray

# === STEP 3: Process All Images in Folder ===
def process_folder(input_folder="images", output_folder="thermal_outputs"):
    os.makedirs(os.path.join(output_folder, "colored"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "grayscale"), exist_ok=True)

    supported_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    for ext in supported_extensions:
        found = glob(os.path.join(input_folder, ext))
        print(f"📁 Found {len(found)} images with extension {ext}")
        image_paths.extend(found)

    print(f"\n🔍 Total images found: {len(image_paths)}\n")

    if len(image_paths) == 0:
        print("❌ No images found. Check folder name or add supported formats.")
        return

    for idx, path in enumerate(image_paths):
        filename = os.path.basename(path)
        print(f"🛠️ [{idx + 1}/{len(image_paths)}] Processing: {filename}")

        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Failed to load {filename}, skipping.")
            continue

        img = cv2.resize(img, (512, 512))
        thermal_colored, thermal_gray = simulate_custom_thermal(img)

        out_colored = os.path.join(output_folder, "colored", filename)
        out_gray = os.path.join(output_folder, "grayscale", filename)

        cv2.imwrite(out_colored, thermal_colored)
        cv2.imwrite(out_gray, thermal_gray)
        print(f"✅ Saved: {out_colored} and {out_gray}\n")

    print("🎉 Done! All thermal images saved.")

# === MAIN ===
if __name__ == "__main__":
    process_folder(input_folder="images", output_folder="thermal_outputs")
