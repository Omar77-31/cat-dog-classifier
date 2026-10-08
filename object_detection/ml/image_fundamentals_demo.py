import os
# WHY WE NEED IT: images become grids of numbers, and this is the number library.
# WHY THIS WAY: NumPy makes whole-grid maths easy (e.g. "divide every pixel").
# HOW IT WORKS: we rename it "np" (the normal nickname) to type less.
import numpy as np

# WHY WE NEED IT: to open image files and make small edits (flip, rotate...).
# WHY THIS WAY: Pillow (PIL) is the simplest, most common image library.
# HOW IT WORKS: we pull in "Image" (open/save) and "ImageEnhance" (brightness).
from PIL import Image, ImageEnhance

# WHY WE NEED IT: to tell the script where your cat/dog photos live.
# WHY THIS WAY: keeping it at the top means students change ONE line, not many.
# HOW IT WORKS: os.walk will later search inside this folder (and sub-folders).
# ملاحظة: قم بتغيير هذا المسار إلى مسار مجلد الصور الخاص بك إن وجد
IMAGE_FOLDER = "/home/tala/Downloads/data/training_set/training_set"

# WHY WE NEED IT: a place to save the example pictures we create, to look at them.
# WHY THIS WAY: a separate output folder keeps our results tidy and easy to find.
# HOW IT WORKS: we create this folder later, just before we save into it.
OUTPUT_FOLDER = "demo_output"

# WHY WE NEED IT: how many photos of EACH animal to preprocess in the batch step.
# HOW IT WORKS: used by preprocess_training_data() near the bottom of the file.
IMAGES_PER_CLASS = 1


# =============================================================================
# HELPER: find one photo by name, or make a fake one so the lesson never stalls
# =============================================================================
def find_image(folder, keyword):
    # WHY WE NEED IT: only search if the data folder actually exists on disk.
    # WHY THIS WAY: avoids a crash when students haven't downloaded data yet.
    # HOW IT WORKS: os.path.isdir returns True only if the folder is real.
    if os.path.isdir(folder):
        # WHY WE NEED IT: to look through every sub-folder for image files.
        # WHY THIS WAY: Kaggle data is often nested (train/cats/, etc.).
        # HOW IT WORKS: os.walk hands us each folder + the files inside it.
        for root, _dirs, files in os.walk(folder):
            # WHY WE NEED IT: to check the files in a fixed, predictable order.
            # WHY THIS WAY: sorted() means everyone in class gets the same photo.
            # HOW IT WORKS: loops over the filenames one at a time.
            for name in sorted(files):
                # WHY WE NEED IT: filename checks should ignore capital letters.
                # WHY THIS WAY: "Cat.JPG" and "cat.jpg" should both match "cat".
                # HOW IT WORKS: .lower() makes a lowercase copy of the name.
                lower = name.lower()
                # WHY WE NEED IT: keep only files that are the animal AND an image.
                # WHY THIS WAY: skips stray files like notes.txt or thumbs.db.
                # HOW IT WORKS: both conditions must be True to accept the file.
                if keyword in lower and lower.endswith((".jpg", ".jpeg", ".png")):
                    # WHY WE NEED IT: hand back the full path so we can open it.
                    # WHY THIS WAY: os.path.join builds a correct path on any OS.
                    # HOW IT WORKS: "return" ends the search at the first match.
                    return os.path.join(root, name)
    # WHY WE NEED IT: signal "found nothing" to the caller.
    # WHY THIS WAY: None is Python's clear way of saying "no result".
    # HOW IT WORKS: the caller checks for None and makes a fake image instead.
    return None


# =============================================================================
# 1. LOADING AN IMAGE
# =============================================================================
def load_image(keyword):
    # WHY WE NEED IT: try to get a real photo of the requested animal.
    # WHY THIS WAY: real photos make the lesson concrete and interesting.
    # HOW IT WORKS: find_image returns a path, or None if there isn't one.
    path = find_image(IMAGE_FOLDER, keyword)

    # WHY WE NEED IT: only open a file if we actually found one.
    # WHY THIS WAY: "if path" is True when path is a real string, False if None.
    # HOW IT WORKS: runs this block only when a photo exists.
    if path:
        # WHY WE NEED IT: tell the student which file we are using.
        # WHY THIS WAY: printing progress makes the script easy to follow.
        # HOW IT WORKS: an f-string drops the path variable into the message.
        print(f" Using real photo: {path}")
        # WHY WE NEED IT: load the file and guarantee it is 3-channel colour.
        # WHY THIS WAY: some photos are grayscale/RGBA; .convert("RGB") unifies them.
        # HOW IT WORKS: Image.open reads the file; .convert("RGB") forces 3 channels.
        return Image.open(path).convert("RGB")

    # --- Fallback: build a tiny coloured image so class always works ---
    # WHY WE NEED IT: let the class continue even with no dataset downloaded.
    # WHY THIS WAY: proves the point that "an image is just numbers" from scratch.
    # HOW IT WORKS: we will fill a small number grid and turn it into an image.
    print(f" No '{keyword}' photo found -> using a small coloured test image.")

    # WHY WE NEED IT: an empty 8x8 colour canvas to paint on.
    # WHY THIS WAY: 8x8 is tiny, so students can imagine every pixel.
    # HOW IT WORKS: np.zeros makes an all-0 grid; uint8 = whole numbers 0..255.
    fake = np.zeros(shape=(8, 8, 3), dtype=np.uint8)

    # WHY WE NEED IT / HOW: set the top-left quarter to a red-ish colour (R,G,B).
    fake[:4, :4] = [210, 60, 60]
    # WHY WE NEED IT / HOW: set the top-right quarter to a green-ish colour.
    fake[:4, 4:] = [60, 160, 90]
    # WHY WE NEED IT / HOW: set the bottom-left quarter to a blue-ish colour.
    fake[4:, :4] = [70, 90, 200]
    # WHY WE NEED IT / HOW: set the bottom-right quarter to a yellow-ish colour.
    fake[4:, 4:] = [230, 220, 60]

    # WHY WE NEED IT: turn our number grid back into an image object.
    # WHY THIS WAY: the rest of the code expects a Pillow image, not raw numbers.
    # HOW IT WORKS: Image.fromarray wraps the NumPy grid as an image.
    return Image.fromarray(fake)


# =============================================================================
# 2. AN IMAGE IS JUST NUMBERS, WITH A SHAPE
# =============================================================================
def show_image_is_numbers(pil_image):
    # WHY WE NEED IT / HOW: print a clear banner so students know the section.
    print("\n" + "=" * 60)
    print("1 & 2. AN IMAGE IS JUST NUMBERS (with a shape)")
    print("=" * 60)

    # WHY WE NEED IT: turn the picture into the grid of numbers behind it.
    # WHY THIS WAY: this is THE key idea of the session, made real.
    # HOW IT WORKS: np.array copies every pixel value into a NumPy grid.
    pixels = np.array(pil_image)

    # WHY WE NEED IT: read the three size numbers that describe any image.
    # WHY THIS WAY: .shape is how we always inspect an image's dimensions.
    # HOW IT WORKS: it returns (rows, columns, channels), unpacked into 3 names.
    height, width, channels = pixels.shape

    # WHY WE NEED IT / HOW: show the shape and explain what each number means.
    print(f" Shape (Height, Width, Channels) = {pixels.shape}")
    print(f"   Height   = {height} (rows of pixels, top to bottom)")
    print(f"   Width    = {width} (columns of pixels, left to right)")
    print(f"   Channels = {channels} (3 means Red, Green, Blue)")

    # WHY WE NEED IT: show how many raw numbers an image really is.
    # WHY THIS WAY: multiplying the three sizes gives the total value count.
    # HOW IT WORKS: H x W x C; the ":" formatting adds thousands separators.
    total = height * width * channels
    print(f" Total numbers stored = {height} x {width} x {channels} = {total:,}")

    # WHY WE NEED IT: prove a single pixel is just three numbers.
    # WHY THIS WAY: the top-left corner (row 0, column 0) is easy to point to.
    # HOW IT WORKS: pixels[0, 0] grabs that pixel; we unpack its R, G, B.
    r, g, b = pixels[0, 0]
    print(f" The top-left pixel is stored as: Red={r}, Green={g}, Blue={b}")
    print("  -> every value is between 0 (dark) and 255 (bright).")

    # WHY WE NEED IT: hand the number grid back so other steps can reuse it.
    # WHY THIS WAY: computing it once and returning it avoids repeating work.
    # HOW IT WORKS: "return" sends "pixels" to whoever called this function.
    return pixels


# =============================================================================
# 3. RGB CHANNELS
# =============================================================================
def show_rgb_channels(pixels):
    print("\n" + "=" * 60)
    print("3. RGB: three separate colour layers")
    print("=" * 60)

    # WHY WE NEED IT: pull out just the RED layer of the image.
    # WHY THIS WAY: a colour image is really 3 stacked grids; this takes grid 0.
    # HOW IT WORKS: [:,:, 0] means "all rows, all columns, channel 0 (Red)".
    red = pixels[:, :, 0]
    # WHY WE NEED IT / HOW: same idea for GREEN, which is channel 1.
    green = pixels[:, :, 1]
    # WHY WE NEED IT / HOW: same idea for BLUE, which is channel 2.
    blue = pixels[:, :, 2]

    # WHY WE NEED IT: give a simple, single number that describes each layer.
    # WHY THIS WAY: the average brightness is easy for beginners to compare.
    # HOW IT WORKS: .mean() adds up all values in that grid and divides by count.
    print(f" Average brightness of the RED   layer = {red.mean():.0f}")
    print(f" Average brightness of the GREEN layer = {green.mean():.0f}")
    print(f" Average brightness of the BLUE  layer = {blue.mean():.0f}")
    print(" Each layer is the same size as the image; stacking all three makes colour.")


# =============================================================================
# 4. GRAYSCALE CONVERSION
# =============================================================================
def show_grayscale(pil_image, pixels):
    print("\n" + "=" * 60)
    print("4. GRAYSCALE: squeeze 3 colour numbers into 1 brightness number")
    print("=" * 60)

    # WHY WE NEED IT / HOW: grab each colour layer again to combine them.
    r = pixels[:, :, 0]
    g = pixels[:, :, 1]
    b = pixels[:, :, 2]

    # WHY WE NEED IT: blend the 3 colours into 1 brightness value per pixel.
    # WHY THIS WAY: this is the standard luminance formula; green weighs most
    #               because the human eye is most sensitive to green light.
    # HOW IT WORKS: NumPy multiplies and adds the whole grids at once.
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # WHY WE NEED IT: convert the result back to whole numbers 0..255.
    # WHY THIS WAY: pixels are stored as whole bytes, not decimals.
    # HOW IT WORKS: .astype(np.uint8) rounds/casts to the 0..255 integer type.
    gray = gray.astype(np.uint8)

    # WHY WE NEED IT / HOW: show that colour has 3 channels but gray has 1.
    print(f" Colour shape   = {pixels.shape}  (3 channels)")
    print(f" Grayscale shape = {gray.shape}  (just 1 brightness value per pixel)")
    print("  -> grayscale uses about 3x LESS memory (colour is dropped).")

    # WHY WE NEED IT: make sure the output folder exists before saving into it.
    # WHY THIS WAY: saving to a missing folder would crash the script.
    # HOW IT WORKS: makedirs creates it; exist_ok=True means "fine if it's there".
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # WHY WE NEED IT: save the colour version so students can view it.
    # WHY THIS WAY: comparing files side by side makes the idea click.
    # HOW IT WORKS: .save writes the image to the given path.
    pil_image.save(os.path.join(OUTPUT_FOLDER, "1_original_colour.png"))

    # WHY WE NEED IT: save the grayscale version next to it.
    # WHY THIS WAY: Pillow needs an image object, not a raw number grid.
    # HOW IT WORKS: Image.fromarray wraps our gray grid, then .save writes it.
    Image.fromarray(gray).save(os.path.join(OUTPUT_FOLDER, "2_grayscale.png"))


# =============================================================================
# 5 & 6. NORMALIZATION + THE FULL PREPROCESSING PIPELINE
# =============================================================================
def preprocess(pil_image, size=(128, 128)):
    print("\n" + "=" * 60)
    print("5 & 6. THE PREPROCESSING PIPELINE")
    print("=" * 60)

    # WHY WE NEED IT: STEP "Resize" -- make every image one fixed size.
    # WHY THIS WAY: a model has ONE input size; all images must match to batch.
    # HOW IT WORKS: .resize squashes/stretches the photo to (128, 128).
    resized = pil_image.resize(size)
    print(f" Resized to {size} so all images match.")

    # WHY WE NEED IT: STEP "Convert" -- guarantee the colour format is RGB.
    # WHY THIS WAY: the model expects 3 channels; this removes surprises.
    # HOW IT WORKS: .convert("RGB") forces exactly 3 channels.
    resized = resized.convert("RGB")

    # WHY WE NEED IT: STEP "Load into numbers", as decimals we can divide.
    # WHY THIS WAY: normalizing needs decimals, so we use float, not int.
    # HOW IT WORKS: np.array reads the pixels; .astype("float32") makes them decimal.
    pixels = np.array(resized).astype("float32")
    print(f" Before normalizing: min pixel = {pixels.min():.0f}, max = {pixels.max():.0f}")

    # WHY WE NEED IT: STEP "Normalize" -- shrink 0..255 down to 0..1.
    # WHY THIS WAY: small tidy numbers make a network train faster and steadier.
    # HOW IT WORKS: dividing the whole grid by 255 scales every value at once.
    normalized = pixels / 255.0
    print(f" After normalizing: min pixel = {normalized.min():.2f}, "
          f"max = {normalized.max():.2f}  (small, tidy numbers)")
    print(f" Final shape ready for a model = {normalized.shape}")

    # WHY WE NEED IT / HOW: give the cleaned image back for the batching demo.
    return normalized


def show_batching(one_image):
    print("\n BATCHING: models look at many images at once.")
    # WHY WE NEED IT: show how single images stack into a group (a batch).
    # WHY THIS WAY: training many at once is faster and learns more steadily.
    # HOW IT WORKS: np.stack piles 4 copies, adding a new size at the front.
    batch = np.stack([one_image, one_image, one_image, one_image])

    # WHY WE NEED IT / HOW: compare one-image shape vs the 4D batch shape.
    print(f"   One image shape = {one_image.shape}")
    print(f"   Batch of 4 shape = {batch.shape}  (the leading 4 = batch size)")


# =============================================================================
# 7. DATA AUGMENTATION
# =============================================================================
def show_augmentation(pil_image):
    print("\n" + "=" * 60)
    print("7. DATA AUGMENTATION: make MORE training images from one")
    print("=" * 60)
    print(" Small random changes = new training examples (label stays the same).")

    # WHY WE NEED IT / HOW: make sure the save folder exists (see note earlier).
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # WHY WE NEED IT: start from one clean, fixed-size colour image.
    # WHY THIS WAY: all our augmented versions should share the same base.
    # HOW IT WORKS: resize to 128x128 and force RGB, then reuse "base" below.
    base = pil_image.resize((128, 128)).convert("RGB")

    # WHY WE NEED IT: create a mirrored copy (a flipped cat is still a cat).
    # WHY THIS WAY: teaches the model that left/right doesn't change the label.
    # HOW IT WORKS: transpose(FLIP_LEFT_RIGHT) swaps the image left-to-right.
    flipped = base.transpose(Image.FLIP_LEFT_RIGHT)
    flipped.save(os.path.join(OUTPUT_FOLDER, "3_aug_flip.png"))

    # WHY WE NEED IT: create a slightly rotated copy.
    # WHY THIS WAY: real objects appear at small angles; the model should cope.
    # HOW IT WORKS: .rotate(15) turns the image 15 degrees.
    rotated = base.rotate(15)
    rotated.save(os.path.join(OUTPUT_FOLDER, "4_aug_rotate.png"))

    # WHY WE NEED IT: create a brighter copy.
    # WHY THIS WAY: photos vary with lighting; brightness changes help the model.
    # HOW IT WORKS: Brightness(base).enhance(1.4) makes it 40% brighter.
    brighter = ImageEnhance.Brightness(base).enhance(1.4)
    brighter.save(os.path.join(OUTPUT_FOLDER, "5_aug_bright.png"))

    print(" Saved 3 new versions from ONE image:")
    print("   3_aug_flip.png    (mirrored left-right)")
    print("   4_aug_rotate.png  (rotated 15 degrees)")
    print("   5_aug_bright.png  (brightened)")
    print(" REMEMBER: only augment TRAINING images -- never test images.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # 1. Load an image (or fallback to a fake one if the dataset is missing)
    img = load_image("cat")

    # 2 & 3. Show numbers and RGB channels
    pixels = show_image_is_numbers(img)
    show_rgb_channels(pixels)

    # 4. Grayscale
    show_grayscale(img, pixels)

    # 5 & 6. Preprocess (Resize, Normalize)
    processed_img = preprocess(img)

    # Batching demonstration
    show_batching(processed_img)

    # 7. Data Augmentation
    show_augmentation(img)