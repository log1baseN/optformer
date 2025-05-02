# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
project_root = "/Users/animeshmadaan/Desktop/CS772/optformer" # Must be specified (path to "/optformer")
os.chdir(project_root)
sys.path.append(os.getcwd()) # Fix Python Path

import functools
import numpy as np
from optformer.decoding_regression import models
from optformer.decoding_regression import vocabs
import tensorflow as tf
from absl.testing import absltest
from absl.testing import parameterized

keras = tf.keras


class ModelTest(parameterized.TestCase):

  @parameterized.parameters((None, None), (5, None), (None, 0.5), (3, 0.1))
  def test_e2e(self, top_k, top_p):
    # 1) Build encoder
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128),
    ])

    # 2) Instantiate and monkey-patch vocab for hex support
    vocab = vocabs.UnnormalizedVocab()

    # a) Inject all <char> tokens for float.hex()
    for ch in '0123456789abcdefxp+-.':
        tok = f"<{ch}>"
        if tok not in vocab._toks_to_ids:
            vocab._vocab_toks.append(tok)
            vocab._toks_to_ids[tok] = len(vocab._vocab_toks) - 1

    # b) Override to_int so hex-strings map to those <char> tokens
    orig_to_int = vocab.to_int
    def to_int_hex(f):
        if isinstance(f, str):
            return [vocab._toks_to_ids[f"<{c}>"] for c in f]
        else:
            return orig_to_int(f)
    vocab.to_int = to_int_hex

    # c) Override token_length property on the class so the decoder's embedding
    #    input_dim == vocab.size (the full hex alphabet)
    vocabs.UnnormalizedVocab.token_length = property(lambda self: self.size)

    # 3) Build decoder with the expanded vocab
    decoder = models.AttentionDecoder(encoder, vocab)
    
    # Generate data and token IDs:
    num_data, feature_dim = 2000, 10
    X = np.random.uniform(size=(num_data, feature_dim))
    weights = np.random.uniform(size=(feature_dim,))
    Y = np.sum(X * weights, axis=-1)
    Y_hex = [y.hex() for y in Y]
    Y_token_ids = np.array([vocab.to_int(h) for h in Y_hex], dtype=np.int32)

    # Compute a “flat” weight vector of ones matching the sequence length:
    seq_len = Y_token_ids.shape[1]
    flat_weights = np.ones((seq_len,), dtype=np.float32)

    # Bind those into the loss so that normalized_weights == 1 everywhere:
    loss_fn = functools.partial(
        models.weighted_sparse_categorical_crossentropy,
        weights=flat_weights
    )

    decoder.compile(
        keras.optimizers.Adam(learning_rate=1e-4),
        loss=functools.partial(
            models.weighted_sparse_categorical_crossentropy,
        ),
    )

    # Train:
    decoder.fit(
        [X, Y_token_ids[:, :-1]],
        Y_token_ids,
        batch_size=32,
        epochs=10,
        validation_split=0.2,
    )

    floats = decoder.decode(X, top_k=top_k, top_p=top_p)
    print("True value")
    print(Y)
    print("Predicted")
    print(floats)
    # for i in range(len(floats)):
    #   if abs(floats[i] - Y[i]) > 100:
    #     print("Index - " + str(i))
    #     print("Error - " + str(abs(floats[i] - Y[i])))
    #     print("True - " + str(Y[i]))
    #     print("Predicted - " + str(floats[i]))
    mask  = np.abs(floats - Y) < 100
    print("Mean Square Error without outliers - " + str(np.mean((Y[mask] - np.array(floats)[mask])**2)**0.5))
    print("Mean Square Error - " + str(np.mean((Y - np.array(floats))**2)**0.5))
    # self.assertLen(floats, 10)


if __name__ == "__main__":
  absltest.main()
