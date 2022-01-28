import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from models.vit import Patches, PatchEncoder, transformer_block

def conv_block(x, filters):
    x = layers.Conv3D(filters, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool3D(2)(x)
    x = layers.Dropout(.5)(x)
    return x


def pdnet(input_shape,
          patch_size=16,
          projection_dim=16,
          transformer_layers=1,
          num_heads=2,
          classes=2):
    grid_nums = [dim // patch_size for dim in input_shape[:-1]]
    inp = layers.Input(input_shape)
    
    x1 = Patches(patch_size)(inp)
    x1 = PatchEncoder(np.prod(grid_nums), projection_dim)(x1)
    x1 = transformer_block(x1, transformer_layers, num_heads, projection_dim)
    x2 = conv_block(inp, 8)
    x3 = layers.MaxPool3D(8)(x2)
    x3 = tf.reshape(x3, (-1, np.prod(x3.shape[1:-1]), x3.shape[-1]))
    
    x1 = layers.Concatenate()([x1, x3])
    x1 = layers.Dense(projection_dim, activation=tf.nn.gelu)(x1)
    x1 = layers.Dropout(.5)(x1)
    x1 = transformer_block(x1, transformer_layers, num_heads, projection_dim)
    x2 = conv_block(x2, 16)
    x3 = layers.MaxPool3D(4)(x2)
    x3 = tf.reshape(x3, (-1, np.prod(x3.shape[1:-1]), x3.shape[-1]))

    x1 = layers.Concatenate()([x1, x3])
    x1 = layers.Dense(projection_dim, activation=tf.nn.gelu)(x1)
    x1 = layers.Dropout(.5)(x1)
    x1 = transformer_block(x1, transformer_layers, num_heads, projection_dim)
    x2 = conv_block(x2, 32)

    x1 = layers.LayerNormalization(epsilon=1e-6)(x1)
    attn_weights = tf.nn.softmax(layers.Dense(1)(x1), axis=1)
    x1 = tf.matmul(attn_weights, x1, transpose_a=True)
    x1 = tf.squeeze(x1, -2)
    x2 = layers.GlobalAveragePooling3D()(x2)
    x2 = layers.Dense(projection_dim, activation='relu')(x2)
    x2 = layers.LayerNormalization(epsilon=1e-6)(x2)

    out = layers.Concatenate()([x1, x2])
    out = layers.Dropout(.5)(out)
    out = layers.Dense(classes, activation='softmax')(out)
    return tf.keras.Model(inp, out, name='PDNet')
