import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize + add channel dim: (N, 28, 28, 1)
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test  = (x_test.astype("float32") / 255.0)[..., None]

    # Model (use Input() to avoid the warning)
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        x_train, y_train,
        epochs=3,
        validation_split=0.1,
        batch_size=32,
        verbose=1
    )

    # Save formats
    model.save("mnist_model.h5")            # optional legacy
    model.save("mnist_model.keras")         # recommended Keras format

    # THIS creates the folder ./saved_mnist/ with saved_model.pb + variables/
    model.export("saved_mnist")

    print("Saved: mnist_model.h5, mnist_model.keras, and exported SavedModel to ./saved_mnist")

if __name__ == "__main__":
    main()
