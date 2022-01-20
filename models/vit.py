import tensorflow as tf
from tensorflow.keras import layers

def mlp(x, hidden_units, dropout_rate):
    """Multi Layer Perceptron"""
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    """Layer to divide input image into patches"""
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1] # flattened patch dim (patch_size * patch_size * depth)
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        cfg = super().get_config().copy()
        cfg.update({'patch_size': self.patch_size})
        return cfg


class PatchEncoder(layers.Layer):
    """Patch Encoding + Position Embedding"""
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim = num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        cfg = super().get_config().copy()
        cfg.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim
        })
        return cfg


def vit(input_shape,
        patch_size=16,
        projection_dim=128,
        transformer_layers=8,
        num_heads=4,
        mlp_head_units=[512, 128],
        classes=2,
        augmentation_layer=None,
        **kwargs):
    """Vision Transformer model from https://arxiv.org/pdf/2010.11929.pdf"""
    assert input_shape[0] == input_shape[1], 'Input image must be square'
    transformer_units = [projection_dim * 2, projection_dim]
    
    inp = layers.Input(input_shape)
    x = augmentation_layer(inp) if augmentation_layer else inp
    num_patches = (x.shape[1] // patch_size) ** 2
    x = Patches(patch_size)(x)
    x = PatchEncoder(num_patches, projection_dim)(x)    
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        x1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x = layers.Add()([x, x1])
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        x1 = mlp(x1, hidden_units=transformer_units, dropout_rate=0.1)
        x = layers.Add()([x, x1])

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = mlp(x, hidden_units=mlp_head_units, dropout_rate=0.5)
    out = layers.Dense(classes, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='ViT')
