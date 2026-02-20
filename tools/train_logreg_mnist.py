import json
import sys
import traceback
import numpy as np

print("Script started.", flush=True)

try:
    print("Importing sklearn...", flush=True)
    from sklearn.linear_model import LogisticRegression

    print("Importing MNIST...", flush=True)
    from tensorflow.keras.datasets import mnist

    print("Loading MNIST...", flush=True)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("Normalizing...", flush=True)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    subset = 10000
    x_train = x_train[:subset]
    y_train = y_train[:subset]

    print("Training logistic regression...", flush=True)

    model = LogisticRegression(
        max_iter=200,
        solver="lbfgs",
        verbose=1,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    acc = model.score(x_test, y_test)
    print("Accuracy:", acc, flush=True)

    print("Exporting weights...", flush=True)

    W = model.coef_.T.flatten().tolist()
    b = model.intercept_.tolist()

    output = {
        "inDim": 784,
        "outDim": 10,
        "W": W,
        "b": b,
    }

    output_path = "../public/model/logreg/logreg.json"

    print("Saving to:", output_path, flush=True)

    with open(output_path, "w") as f:
        json.dump(output, f)

    print("Done.", flush=True)

except Exception:
    print("ERROR OCCURRED:", flush=True)
    traceback.print_exc()
    sys.exit(1)
