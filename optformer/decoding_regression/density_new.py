import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level
one_level_up = os.path.dirname(current_dir)
# Go up another level
project_root = os.path.dirname(one_level_up)
os.chdir(project_root)
sys.path.append(os.getcwd()) # Fix Python Path

import numpy as np
import pandas as pd
import tensorflow as tf
import math
from optformer.decoding_regression import models
from optformer.decoding_regression import vocabs
from absl.testing import absltest
from absl.testing import parameterized
import matplotlib.pyplot as plt
import functools

keras = tf.keras

def make_half_moons(n=400, noise=0.01):
    theta = np.random.rand(n) * np.pi
    x1 = np.cos(theta)
    y1 = np.sin(theta)
    x2 = 1 - np.cos(theta)
    y2 = 1 - np.sin(theta) - 0.5
    pts = np.vstack([np.hstack([x1, x2]), np.hstack([y1, y2])]).T
    pts += np.random.randn(*pts.shape) * noise
    return pts

def make_zigzag(n=400, noise=0.01, segments=6):
    xs = np.random.rand(n)
    ys = ((np.floor(xs * segments) % 2) * 2 - 1) * (xs * segments % 1)
    pts = np.c_[xs, ys] + np.random.randn(n, 2) * noise
    return pts

def make_spiral(n=400, noise=0.01, revolutions=3):
    r = np.linspace(0.1, 1, n)
    theta = revolutions * 2 * np.pi * r
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.c_[x, y] + np.random.randn(n, 2) * noise
    return pts

def make_hollow_square(n=400, noise=0.01):
    side = n // 4
    xs = np.linspace(-1, 1, side)
    top = np.c_[xs, np.ones_like(xs)]
    bottom = np.c_[xs, -np.ones_like(xs)]
    left = np.c_[np.ones_like(xs), xs]
    right = np.c_[ -np.ones_like(xs), xs]
    pts = np.vstack([top, bottom, left, right]) + np.random.randn(n, 2) * noise
    return pts

SHAPES = dict(half_moons=make_half_moons,
              zigzag=make_zigzag,
              spiral=make_spiral,
              hollow_square=make_hollow_square)

class ShapeDataset(tf.data.Dataset):
    def __new__(cls, pts, tokenizer, pad=0):
        x = tf.convert_to_tensor(pts[:, 0:1], dtype=tf.float32)
        y = pts[:, 1].astype(float)
        y_tok = [tokenizer.encode(v) for v in y]

        # Convert token sequences to RaggedTensor first
        y_tok_rt = tf.ragged.constant(y_tok, dtype=tf.int64)

        # Zip x and tokenized y together
        dataset = tf.data.Dataset.from_tensor_slices((x, y_tok_rt))

        def pad_fn(x, y):
            return x, tf.pad(y, [[0, max_len - tf.shape(y)[0]]], constant_values=pad)

        max_len = max(len(seq) for seq in y_tok)
        dataset = dataset.map(lambda x, y: (x, tf.pad(y, [[0, max_len - tf.shape(y)[0]]], constant_values=pad)))

        return dataset
    
# class DensityEstimationTest(absltest.TestCase):

#     def test_density_estimation(self):
#         # Test the density estimation function
#         train(shape='zigzag')
    
def train(shape = 'zigzag',top_k=None, top_p=None):
    pts = SHAPES[shape](n=3000, noise=0.01)
    vocab = vocabs.UnnormalizedVocab()
    encoder = tf.keras.Sequential([
        keras.layers.Dense(256, activation='relu'),
        # keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256),
    ])
    decoder = models.AttentionDecoder(
        encoder=encoder,      # a Keras model (e.g., your MLP encoder)
        vocab=vocab,          # an instance of FloatVocab (like HammingDistanceVocab)
        units=256,            # hidden dimension for attention layers (you can tune this)
        num_layers=2,         # number of transformer decoder layers
        num_heads=4,          # number of attention heads
        dropout=0.1,          # dropout rate
    )

    X = pts[:, 0:1].astype(float)
    Y = pts[:, 1].astype(float)

    Y_token_ids = np.array([vocab.to_int(y) for y in Y], dtype=np.int32)
    print(X[0])
    print(Y_token_ids[0])
    decoder.compile(
        keras.optimizers.Adam(learning_rate=1e-4),
        loss=functools.partial(
            models.weighted_sparse_categorical_crossentropy,
            weights=np.array([0.3, 0.3, 0.09, 0.01, 0.01, 0.3, 0.5]),
        ),
    )
    decoder.fit(
        [X, Y_token_ids[:, :-1]],
        Y_token_ids,
        batch_size=32,
        epochs=100,
        validation_split=0.2,
    )

    # Sampling from the learned density: generate multiple y samples for given x
    x_test = np.linspace(X.min(), X.max(), 3000).astype(float)
    sampled_y = decoder.decode(x_test, top_k=top_k, top_p=top_p)  # returns float predictions

    points = []
    refused_pts = 0
    for i in range(len(sampled_y)):
        if sampled_y[i] > 1.5 or sampled_y[i] < -1:
            refused_pts += 1
            continue
        points.append([x_test[i], sampled_y[i]])
    points = np.array(points)
    print(f"Percentage of rejected points is {refused_pts*100.0/(refused_pts+len(pts))} and number is {refused_pts}")

    ground_truth_df = pd.DataFrame({
        "x": X.ravel(),
        "y": Y
    })
    ground_truth_df.to_csv("/home/sankalp/optformer/optformer/decoding_regression/data/ground_truth.csv", index=False)

    # Sampled points
    predicted_df = pd.DataFrame({
        "x": points[:,0],
        "y": points[:,1]
    })
    predicted_df.to_csv("/home/sankalp/optformer/optformer/decoding_regression/data/predicted.csv", index=False)



# class ModelTest(parameterized.TestCase):

#   @parameterized.parameters((None, None), (5, None), (None, 0.5), (3, 0.1))
#   def test_e2e(self, top_k, top_p):
#     train(shape='zigzag',top_k = top_k,top_p = top_p)


# if __name__ == "__main__":
#     absltest.main()

train(shape='spiral',top_k = None,top_p = None)

