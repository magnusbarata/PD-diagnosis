import tensorflow as tf

def gradcam(inputs, model, layer_name):
    """Grad-CAM algorithm implementation.

    Args:
      inputs: Inputs to be passed to the model.
      model: `keras.Model` to be inspected using Grad-CAM.
      layer_name: The name of layer to be visualized.

    Returns:
      Grad-CAM heatmap.
    """
    grad_model = tf.keras.Model(
        model.inputs, [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inputs)
        class_channel = preds[:, tf.argmax(preds[0])]
    
    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2,3))
    
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    return (tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)).numpy()
