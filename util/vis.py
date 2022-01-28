import tensorflow as tf
import numpy as np

from models.vit import Patches

def attention_map(inputs, model):
    """Attention map visualizer.

    Args:
      inputs: Inputs to be passed to the model.
      model: Vision Transformer model.

    Returns:
      Attention map mask.
    """
    # Get all the outputs from MultiHeadAttention layers
    patch_layer_found = False
    grid_nums = [0] * (inputs.ndim - 2)
    new_outputs = []
    for layer in model.layers:
        if isinstance(layer, Patches) and not patch_layer_found:
            grid_nums = tuple(dim // layer.patch_size for dim in layer.input_shape[1:-1])
            patch_layer_found = True

        if isinstance(layer, tf.keras.layers.MultiHeadAttention):
            _, weights = layer(layer.input, layer.input, return_attention_scores=True)
            new_outputs.append(weights)
    attn_map_model = tf.keras.Model(model.inputs, new_outputs)

    weights = np.array(attn_map_model(inputs))
    weights = np.squeeze(weights).mean(axis=1)
    weights += np.eye(weights.shape[1])
    weights /= weights.sum(axis=(1,2), keepdims=True)
    
    v = weights[-1]
    for n in range(1, len(weights)):
        v = np.matmul(v, weights[-1-n])

    mask = v[0].reshape(grid_nums)
    return mask / mask.max()
