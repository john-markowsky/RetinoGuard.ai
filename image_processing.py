import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os
import datetime
from keras.models import Model

def generate_gradcam_heatmap(image_array, model_inception):
    # Create a separate model for Grad-CAM
    grad_model = Model([model_inception.inputs], 
                    [model_inception.get_layer('mixed10').output, 
                    model_inception.output])

    # Generate Grad-CAM heatmap
    with tf.GradientTape(persistent=True) as tape:
        image_tensor = tf.convert_to_tensor(image_array)  # Convert to TensorFlow tensor
        tape.watch(image_tensor)
        last_conv_layer_output, preds = grad_model(image_tensor)  # Pass the input through the grad_model
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the predicted class with regard to
    # the output feature map of 'mixed10'
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    # Convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)

    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.4 + image_array

    return superimposed_img

def cleanup_old_files(directory):
    """Remove files older than 1 day from the specified directory."""
    now = datetime.datetime.now()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        file_time = datetime.datetime.fromtimestamp(os.path.getctime(filepath))
        if (now - file_time).days > 1:
            os.remove(filepath)
