from tensorflow.keras import layers, models


def build_cnn(input_shape=(128, 128, 3)):
    """Create and return an (untrained) cat-vs-dog network."""

    model = models.Sequential([
        # ---------- INPUT LAYER ----------
        # An image arrives as height x width x 3 colour channels (R, G, B).
        layers.Input(shape=input_shape),

        # ---------- HIDDEN LAYER 1: find simple patterns (edges) ----------
        # Conv2D slides small filters over the image looking for patterns.
        # activation="relu"  ->  ReLU = max(0, x): negatives become 0,
        # positives pass through unchanged. (See the note at the top.)
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(),   # shrink the picture, keep the strongest signals

        # ---------- HIDDEN LAYER 2: combine edges into shapes ----------
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        # ---------- HIDDEN LAYER 3: combine shapes into parts ----------
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        # Flatten the 2D feature maps into one long list of numbers
        # so we can feed them into ordinary (dense) neurons.
        layers.Flatten(),

        # ---------- A NORMAL HIDDEN LAYER of 64 neurons ----------
        layers.Dense(64, activation="relu"),

        # ---------- OUTPUT LAYER ----------
        # ONE neuron with Sigmoid. Sigmoid squashes any number into 0..1,
        # so we read the result as a probability that the image is a DOG:
        #   ~1 = Dog,  ~0 = Cat,  ~0.5 = unsure. (See the note at the top.)
        #
        # LABEL ORDER (so cats never get read as dogs): when you train with
        # train.py, the folders are read in ALPHABETICAL order, which makes
        #   cats -> 0   and   dogs -> 1.
        # That matches how classifier.py reads this neuron (>= 0.5 means Dog).
        # Keep the folders named "cats" and "dogs" and the labels stay correct.
        layers.Dense(1, activation="sigmoid"),
    ])

    return model


# Quick manual check: run "python network.py" to print the layer summary.
if __name__ == "__main__":
    build_cnn().summary()