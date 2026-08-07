
import os
import random
from PIL import Image, ImageEnhance, ImageFilter

# ==========================================
# 1. CONFIGURATION PATHS & VARIABLES
# ==========================================
BG_FOLDER = r"C:\Users\Bartek\Desktop\SUAV\whole_nomad\TN_a50"
FG_FOLDER = r"C:\Users\Bartek\Desktop\SUAV\cuts"
OUT_IMG_FOLDER = r"C:\Users\Bartek\Desktop\SUAV\cuts\output\images"
OUT_LABEL_FOLDER = r"C:\Users\Bartek\Desktop\SUAV\cuts\output\labels"

SLICE_SIZE = 640          # 640x640 crops
SLICES_PER_PAIR = 5       # How many slices to generate per bg + fg combination
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
        
    # Random rotation (-15 to +15 degrees)
    angle = random.uniform(-15, 15)
    fg_image = fg_image.rotate(angle, resample=Image.BICUBIC, expand=True)
    
    # Random brightness (0.8x to 1.2x)
    enhancer = ImageEnhance.Brightness(fg_image)
    fg_image = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Apply random Gaussian Blur to soften artificial edges
    # We apply it randomly so the model sees both sharp and slightly blurry examples
    if random.choice([True, False]):
        # Radius between 0.5 and 1.5 pixels (keeps it subtle but effective)
        blur_radius = random.uniform(0.5, 1.5)
        fg_image = fg_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # CRITICAL: Cropping to the actual non-transparent pixels after rotation/blur
    bbox = fg_image.getbbox()
    if bbox:
        fg_image = fg_image.crop(bbox)
        
    return fg_image

# ==========================================
# 3. MAIN GENERATION LOOP
# ==========================================
def generate_dataset():
    # Get all valid image files
    bg_files = [f for f in os.listdir(BG_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fg_files = [f for f in os.listdir(FG_FOLDER) if f.lower().endswith('.png')]
    
    if not bg_files or not fg_files:
        print("Error: Background or Foreground folder is empty or paths are wrong.")
        return

    generated_count = 0

    for bg_name in bg_files:
        bg_path = os.path.join(BG_FOLDER, bg_name)
        
        try:
            bg_full = Image.open(bg_path).convert("RGBA")
        except Exception as e:
            print(f"Could not open {bg_name}: {e}")
            continue
            
        bg_w, bg_h = bg_full.size
        
        if bg_w < SLICE_SIZE or bg_h < SLICE_SIZE:
            continue

        for fg_name in fg_files:
            fg_path = os.path.join(FG_FOLDER, fg_name)
            
            try:
                fg_original = Image.open(fg_path).convert("RGBA")
            except Exception as e:
                print(f"Could not open {fg_name}: {e}")
                continue

            for i in range(SLICES_PER_PAIR):
                crop_x = random.randint(0, bg_w - SLICE_SIZE)
                crop_y = random.randint(0, bg_h - SLICE_SIZE)
                bg_slice = bg_full.crop((crop_x, crop_y, crop_x + SLICE_SIZE, crop_y + SLICE_SIZE))
                
                fg_aug = augment_foreground(fg_original)
                fg_aug_w, fg_aug_h = fg_aug.size
                
                if fg_aug_w >= SLICE_SIZE or fg_aug_h >= SLICE_SIZE:
                    continue
                
                paste_x = random.randint(0, SLICE_SIZE - fg_aug_w)
                paste_y = random.randint(0, SLICE_SIZE - fg_aug_h)
                
                bg_slice.alpha_composite(fg_aug, dest=(paste_x, paste_y))
                
                base_name = f"{os.path.splitext(bg_name)[0]}_{os.path.splitext(fg_name)[0]}_slice{i}"
                img_out_path = os.path.join(OUT_IMG_FOLDER, f"{base_name}.jpg")
                
                final_img = bg_slice.convert("RGB")
                final_img.save(img_out_path, format="JPEG", quality=95)
                
                center_x = (paste_x + (fg_aug_w / 2.0)) / SLICE_SIZE
                center_y = (paste_y + (fg_aug_h / 2.0)) / SLICE_SIZE
                norm_w = fg_aug_w / SLICE_SIZE
                norm_h = fg_aug_h / SLICE_SIZE
                
                label_out_path = os.path.join(OUT_LABEL_FOLDER, f"{base_name}.txt")
                with open(label_out_path, "w") as f:
                    f.write(f"{CLASS_ID} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                
                generated_count += 1
                
    print(f"\nDone! Successfully generated {generated_count} synthetic images and labels.")

if __name__ == "__main__":
    generate_dataset()