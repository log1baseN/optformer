import sys
import os

import sys
import os

dataset_name = "gas"  

# Create output directory if needed
os.makedirs("outputs", exist_ok=True)
log_file_path = f"outputs/{dataset_name}_output.txt"

# Redirect stdout to file
log_file = open(log_file_path, "w")
sys.stdout = log_file

# Print the dataset name at the top
print(f"Dataset: {dataset_name}\n{'='*40}\n")

current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level
one_level_up = os.path.dirname(current_dir)
# Go up another level
two_level_up = os.path.dirname(one_level_up)
project_root = os.path.dirname(two_level_up)
os.chdir(project_root)
sys.path.append(os.getcwd()) # Fix Python Path

import functools
import numpy as np
from optformer.decoding_regression import models
from optformer.decoding_regression import vocabs
from optformer.decoding_regression import generate_data
import tensorflow as tf
from absl.testing import absltest
from absl.testing import parameterized

keras = tf.keras


class ModelTest(parameterized.TestCase):

  @parameterized.parameters((None, None), (5, None), (None, 0.5), (3, 0.1))
  def test_e2e(self, top_k, top_p):
    # pylint: disable=invalid-name
    encoder = tf.keras.Sequential([
    # tf.keras.layers.InputLayer(input_shape=(40,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    # tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512),  # final embedding, linear activation
    ])
    vocab = vocabs.UnnormalizedVocab()
    decoder = models.AttentionDecoder(
        encoder=encoder,      # a Keras model (e.g., your MLP encoder)
        vocab=vocab,          # an instance of FloatVocab (like HammingDistanceVocab)
        units=256,            # hidden dimension for attention layers (you can tune this)
        num_layers=2,         # number of transformer decoder layers
        num_heads=4,          # number of attention heads
        dropout=0.1,          # dropout rate
    )

    X_train , Y_train, X_test, Y_test = generate_data.load_data(dataset_name)
    y_min = min(np.min(Y_train),np.min(Y_test))
    y_max = max(np.max(Y_train),np.max(Y_test))
    # Y_train = (Y_train - y_min) / (y_max - y_min)
    # Y_test = (Y_test - y_min) / (y_max - y_min)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    Y_token_ids = np.array([vocab.to_int(y[0]) for y in Y_train], dtype=np.int32)
    decoder.compile(
        keras.optimizers.Adam(learning_rate=1e-4),
        loss=functools.partial(
            models.weighted_sparse_categorical_crossentropy,
            # weights=np.array([0.3, 0.3, 0.09, 0.01, 0.01, 0.3, 0.5]),
        ),
    )
    decoder.fit(
        [X_train, Y_token_ids[:, :-1]],
        Y_token_ids,
        batch_size=64,
        epochs=50,
        validation_split=0.2,
    )

    floats = decoder.decode(X_test, top_k=top_k, top_p=top_p)
    Y_test_token_ids = np.array([vocab.to_int(y[0]) for y in Y_test], dtype=np.int32)
    # Y_test = (y_max - y_min) * Y_test + y_min
    # floats = (y_max - y_min) * np.array(floats) + y_min
    print("True value")
    print(Y_test[:10])
    print("Predicted")
    print(floats[:10])
    mse = np.mean((Y_test - np.array(floats))**2)
    print("Mean Square Error - " + str(mse))
    var = np.var(Y_test)
    if var > 0:
        print("Relative Mean Square Error - " + str(mse/var))

    
    # Continuous Negative Log‑Likelihood (Gaussian)
    sigma2 = np.var(Y_train)  # use variance of training targets as σ²
    nll = 0.5 * np.log(2 * np.pi * sigma2) + ((Y_test - np.array(floats))**2) / (2 * sigma2)
    nll_mean = np.mean(nll)
    print("Gaussian Negative Log-Likelihood - " + str(nll_mean))

    score = decoder.evaluate(
        [X_test, Y_test_token_ids[:, :-1]],
        Y_test_token_ids,
        batch_size=32,
    )

    print("Test loss: ", score)


if __name__ == "__main__":
    absltest.main()
    log_file.close()
