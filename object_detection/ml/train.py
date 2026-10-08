import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from network import build_cnn   # the brain we designed in network.py

IMG_SIZE = (128, 128)
BATCH = 32
HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")
SAVE_PATH = os.path.join(HERE, "cat_dog_model.h5")


def main():
    # Load images from the folders and scale every pixel to the 0..1 range.
    gen = ImageDataGenerator(rescale=1.0 / 255)

    train = gen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE, batch_size=BATCH, class_mode="binary",
    )
    val = gen.flow_from_directory(
        os.path.join(DATA_DIR, "validation"),
        target_size=IMG_SIZE, batch_size=BATCH, class_mode="binary",
    )

    # Build the (untrained) network.
    model = build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Tell it HOW to learn:
    #   loss      = how wrong it is (binary_crossentropy suits yes/no answers)
    #   optimizer = the rule that nudges the weights (Adam, a smart version
    #               of the learning rule you saw)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # LEARN: look at all the images a few times (each pass = one "epoch")
    # and adjust the weights to make fewer mistakes.
    model.fit(train, validation_data=val, epochs=5)

    # Save the trained brain so the web app can load it.
    model.save(SAVE_PATH)
    print(f"\nDone! Saved your trained network to:\n  {SAVE_PATH}")


if __name__ == "__main__":
    main()