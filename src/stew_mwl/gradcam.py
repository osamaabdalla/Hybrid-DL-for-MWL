
from __future__ import annotations
import numpy as np
import tensorflow as tf

def make_gradcam_heatmap(model, sequence_batch, class_index=None, conv_layer_name=None):
    if conv_layer_name is None:
        # choose the last conv layer from the frame encoder
        conv_candidates = [l.name for l in model.layers if "conv" in l.name.lower()]
        if not conv_candidates:
            raise ValueError("Could not infer a convolutional layer for Grad-CAM.")
        conv_layer_name = conv_candidates[-1]

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(sequence_batch)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def activation_frequency_summary(heatmaps, threshold=0.6):
    heatmaps = np.asarray(heatmaps)
    return (heatmaps >= threshold).mean(axis=0)
