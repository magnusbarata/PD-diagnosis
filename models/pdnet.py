import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from models.vit import mlp, Patches, PatchEncoder

def transformer_encoder(x, num_heads=2, projection_dim=32):
    mlp_units = [projection_dim * 2, projection_dim]
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    x1 = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0.1
    )(x1, x1)
    x += x1
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    x1 = mlp(x1, hidden_units=mlp_units, dropout_rate=0.1)
    return x + x1


def pdnet(input_shape,
          patch_size=16,
          projection_dim=16,
          num_heads=4,
          classes=2):
    grid_nums = [dim // patch_size for dim in input_shape[:-1]]
    inp = layers.Input(input_shape)
    
    x1= layers.Conv3D(16, 3, activation='relu', padding='same')(inp)
    x1 = layers.MaxPool3D(2)(x1)
    x2 = Patches(patch_size)(inp)
    x2 = PatchEncoder(np.prod(grid_nums), projection_dim)(x2)
    x2 = transformer_encoder(x2, num_heads, projection_dim)
    x2 = transformer_encoder(x2, num_heads, projection_dim)
    x = tf.reshape(x2, (-1, *grid_nums, x2.shape[-1]))
    x = layers.Conv3DTranspose(16, 8, 8, activation='relu')(x)
    x = layers.Concatenate()([x, x1])
    x = layers.Dropout(.5)(x)

    x1 = layers.Conv3D(32, 3, activation='relu', padding='same')(x)
    x1 = layers.MaxPool3D(2)(x1)
    x2 = transformer_encoder(x2, num_heads, projection_dim)
    x2 = transformer_encoder(x2, num_heads, projection_dim)
    x = tf.reshape(x2, (-1, *grid_nums, x2.shape[-1]))
    x = layers.Conv3DTranspose(32, 4, 4, activation='relu')(x)
    x = layers.Concatenate()([x, x1])
    x = layers.Dropout(.5)(x)
    
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(.5)(x)
    out = layers.Dense(classes, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='PDNet')
