import os
import numpy as np
# NOTE: TensorFlow is imported *inside* the functions below, not here.
# That keeps this file light so `migrate` / `runserver` start instantly;
# the neural-network library only loads the moment we make a prediction.

# We load the network ONCE and keep it in memory (loading is slow, predicting
# is fast). These module-level variables remember it between requests.
_model = None
_model_kind = None

# Where a student-trained network would be saved.
CUSTOM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "cat_dog_model.h5")


def _load_model():
    """Load the neural network the first time we need it."""
    global _model, _model_kind
    if _model is not None:
        return  # already loaded

    if os.path.exists(CUSTOM_MODEL_PATH):
        # Option A: the network YOU built and trained.
        from tensorflow.keras.models import load_model
        _model = load_model(CUSTOM_MODEL_PATH)
        _model_kind = "custom"
    else:
        # Option B: a ready-made network someone already trained for us.
        # We are still only doing forward propagation through it.
        from tensorflow.keras.applications import MobileNetV2
        _model = MobileNetV2(weights="imagenet")
        _model_kind = "pretrained"


def classify_image(image_path):
    """
    Take an image file path and return (label, confidence).
        label      -> "Cat", "Dog", or "Unknown"
        confidence -> how sure the network is, from 0.0 to 1.0
    """
    _load_model()
    if _model_kind == "custom":
        return _predict_custom(image_path)
    return _predict_pretrained(image_path)


# --------------------------------------------------------------------------
# Path A: forward pass through YOUR small CNN (from network.py)
# --------------------------------------------------------------------------
def _predict_custom(image_path):
    from tensorflow.keras.preprocessing import image as keras_image

    # 1. INPUT LAYER: load the image at the size the network expects
    img = keras_image.load_img(image_path, target_size=(128, 128))
    x = keras_image.img_to_array(img) / 255.0     # scale pixels to 0..1
    x = np.expand_dims(x, axis=0)                  # shape becomes (1, 128, 128, 3)

    # 2. FORWARD PROPAGATION: push the pixels through every layer.
    #    The single output neuron uses SIGMOID, so its value is already a
    #    probability between 0 and 1 -- here it means P(dog).
    prob_dog = float(_model.predict(x, verbose=0)[0][0])

    # 3. READ THE OUTPUT NEURON: Sigmoid gave us P(dog).
    #    >= 0.5 means "more likely dog"; below 0.5 means "more likely cat".
    #    (For a cat, confidence is 1 - P(dog), i.e. P(cat).)
    if prob_dog >= 0.5:
        return "Dog", prob_dog
    return "Cat", 1.0 - prob_dog


# --------------------------------------------------------------------------
# Path B: forward pass through the ready-made MobileNetV2 network
# --------------------------------------------------------------------------
# MobileNetV2 knows 1000 different things. These are the positions in its
# output layer (its "output neurons") that mean cats and dogs.
_DOG_CLASSES = range(151, 269)   # 151..268 are dog breeds
_CAT_CLASSES = range(281, 286)   # 281..285 are domestic cats


def _predict_pretrained(image_path):
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    # 1. INPUT LAYER: MobileNetV2 expects 224x224 images
    img = keras_image.load_img(image_path, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)   # scale the pixels the way this network expects

    # 2. FORWARD PROPAGATION -> 1000 probabilities from a SOFTMAX layer.
    #    Softmax makes all 1000 numbers add up to 1 (100% of the confidence
    #    shared out), so each one is "how likely is it THIS object?".
    preds = _model.predict(x, verbose=0)[0]

    # 3. This network knows many things, not just cat/dog.
    #    IMPORTANT: there are ~118 dog breeds but only ~5 cat types in the list.
    #    If we ADDED UP each group's probabilities, the many dog classes would
    #    almost always win -- even for a clear cat! So instead we take the
    #    SINGLE STRONGEST cat guess and the SINGLE STRONGEST dog guess and
    #    compare those two. That is a fair, size-independent comparison.
    best_cat = max(_CAT_CLASSES, key=lambda i: preds[i])
    best_dog = max(_DOG_CLASSES, key=lambda i: preds[i])
    cat_score = float(preds[best_cat])
    dog_score = float(preds[best_dog])

    # Normalise so the confidence we report is "cat vs dog only".
    total = cat_score + dog_score
    if total == 0:
        return "Unknown", 0.0   # it recognised neither a cat nor a dog

    if dog_score > cat_score:
        return "Dog", dog_score / total
    return "Cat", cat_score / total