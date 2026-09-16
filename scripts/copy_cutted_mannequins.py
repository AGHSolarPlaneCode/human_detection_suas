import os
import random
from PIL import Image, ImageEnhance, ImageFilter

# ==========================================
# 1. CONFIGURATION PATHS & VARIABLES
# ==========================================
BG_FOLDER = r"C:\Users\Bartek\Desktop\SUAV\cuts_v3\bg\train"  
FG_FOLDER = r"C:\Users\Bartek\Desktop\SUAV\cuts_v3\people\train1"
OUT_IMG_FOLDER = r"c:\Users\Bartek\Desktop\SUAV\cuts_v3\train1\images"
OUT_LABEL_FOLDER = r"c:\Users\Bartek\Desktop\SUAV\cuts_v3\train1\labels"

# --- NEW CONFIGURATION SETTINGS ---

TOTAL_IMAGES = 4500           # Total number of generated output images
PURE_BG_PROBABILITY = 0.15     # 0.1 = 10% of generated images will be pure backgrounds (no cuts)

MIN_CUTS_PER_IMAGE = 1        # Minimum number of foregrounds pasted on one background (if not pure)
MAX_CUTS_PER_IMAGE = 5        # Maximum number of foregrounds pasted on one background (if not pure)

BG_BRIGHTNESS_LEVEL = [0.8, 1.2] # Range for random background brightness (1.0 is original)

# ----------------------------------

SLICE_SIZE = 640          # 640x640 crops
CLASS_ID = 0              # YOLO class ID for your mannequin

# Create output directories if they don't exist
os.makedirs(OUT_IMG_FOLDER, exist_ok=True)
os.makedirs(OUT_LABEL_FOLDER, exist_ok=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def augment_foreground(fg_image):
    """Applies rotation, flipping, brightness, and optional blur, then crops tightly."""
    # Random horizontal flip
    if random.choice([True, False]):
        fg_image = fg_image.transpose(Image.FLIP_LEFT_RIGHT)
        
    # Random rotation (-90 to +90 degrees)
    angle = random.uniform(-90, 90)
    fg_image = fg_image.rotate(angle, resample=Image.BICUBIC, expand=True)
    
    # Random brightness (0.6x to 1.4x)
    enhancer = ImageEnhance.Brightness(fg_image)
    fg_image = enhancer.enhance(random.uniform(0.6, 1.4))
    
    # Apply random Gaussian Blur to soften artificial edges
    if random.choice([True, False]):
        blur_radius = random.uniform(0.1, 2.5)
        fg_image = fg_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # CRITICAL: Cropping to the actual non-transparent pixels after rotation/blur
    bbox = fg_image.getbbox()
    if bbox:
        fg_image = fg_image.crop(bbox)
        
    return fg_image

def adjust_background(bg_image):
    """Applies brightness adjustments to the background."""
    # Random Brightness
    enhancer = ImageEnhance.Brightness(bg_image)
    brightness_factor = random.uniform(BG_BRIGHTNESS_LEVEL[0], BG_BRIGHTNESS_LEVEL[1])
    bg_image = enhancer.enhance(brightness_factor)
    return bg_image

# ==========================================
# 3. MAIN GENERATION LOOP
# ==========================================
def generate_dataset():
    # Get all valid image files (Full paths)
    bg_paths = [os.path.join(BG_FOLDER, f) for f in os.listdir(BG_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fg_paths = [os.path.join(FG_FOLDER, f) for f in os.listdir(FG_FOLDER) if f.lower().endswith('.png')]
    
    if not bg_paths or not fg_paths:
        print("Error: Background or Foreground folder is empty or paths are wrong.")
        return

    generated_count = 0
    pure_bg_count = 0

    print(f"Starting generation of {TOTAL_IMAGES} images...")

    while generated_count < TOTAL_IMAGES:
        # 1. Randomly select and prepare a background
        bg_path = random.choice(bg_paths)
        try:
            bg_full = Image.open(bg_path).convert("RGBA")
            bg_full = adjust_background(bg_full)
        except Exception as e:
            print(f"Could not open/process {bg_path}: {e}")
            continue
            
        bg_w, bg_h = bg_full.size
        
        # Ensure the background is large enough for the slice
        if bg_w < SLICE_SIZE or bg_h < SLICE_SIZE:
            continue

        # Crop a random 640x640 slice from the background
        crop_x = random.randint(0, bg_w - SLICE_SIZE)
        crop_y = random.randint(0, bg_h - SLICE_SIZE)
        bg_slice = bg_full.crop((crop_x, crop_y, crop_x + SLICE_SIZE, crop_y + SLICE_SIZE))

        # 2. Decide if this will be a pure background (negative sample)
        is_pure_bg = random.random() < PURE_BG_PROBABILITY
        yolo_labels = []

        if not is_pure_bg:
            # 3. Paste the foreground cuts
            num_cuts = random.randint(MIN_CUTS_PER_IMAGE, MAX_CUTS_PER_IMAGE)
            for _ in range(num_cuts):
                fg_path = random.choice(fg_paths)
                try:
                    fg_original = Image.open(fg_path).convert("RGBA")
                except Exception as e:
                    print(f"Could not open {fg_path}: {e}")
                    continue

                # Augment foreground
                fg_aug = augment_foreground(fg_original)
                fg_aug_w, fg_aug_h = fg_aug.size
                
                # Skip if the cut is larger than the 640x640 background slice
                if fg_aug_w >= SLICE_SIZE or fg_aug_h >= SLICE_SIZE:
                    continue
                
                # Determine random paste coordinates
                paste_x = random.randint(0, SLICE_SIZE - fg_aug_w)
                paste_y = random.randint(0, SLICE_SIZE - fg_aug_h)
                
                # Paste the cut onto the background slice
                bg_slice.alpha_composite(fg_aug, dest=(paste_x, paste_y))
                
                # Calculate YOLO format coordinates
                center_x = (paste_x + (fg_aug_w / 2.0)) / SLICE_SIZE
                center_y = (paste_y + (fg_aug_h / 2.0)) / SLICE_SIZE
                norm_w = fg_aug_w / SLICE_SIZE
                norm_h = fg_aug_h / SLICE_SIZE
                
                yolo_labels.append(f"{CLASS_ID} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

        # 4. Save the final image and label file
        # Save if it's explicitly a pure background, OR if cuts were successfully pasted
        if is_pure_bg or yolo_labels:
            base_name = f"synth_{generated_count:05d}"
            
            img_out_path = os.path.join(OUT_IMG_FOLDER, f"{base_name}.jpg")
            label_out_path = os.path.join(OUT_LABEL_FOLDER, f"{base_name}.txt")
            
            # Save Image
            final_img = bg_slice.convert("RGB")
            final_img.save(img_out_path, format="JPEG", quality=95)
            
            # Save Labels (will be empty for pure backgrounds)
            with open(label_out_path, "w") as f:
                if yolo_labels:
                    f.write("\n".join(yolo_labels) + "\n")
            
            generated_count += 1
            if is_pure_bg:
                pure_bg_count += 1
            
            # Print progress every 100 images
            if generated_count % 100 == 0:
                print(f"Generated {generated_count}/{TOTAL_IMAGES} images...")
                
    print(f"\nDone! Generated {TOTAL_IMAGES} total images ({pure_bg_count} pure backgrounds).")

if __name__ == "__main__":
    generate_dataset()