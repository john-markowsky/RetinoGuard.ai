import tensorflow as tf
import numpy as np
import cv2
import tempfile
from keras.models import Model

def generate_gradcam_heatmap(image_array, image_file, model_inception):
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

    # Create a temporary file in memory
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_image:
        image_file.save(temp_image.name)

        # Use cv2 to resize the heatmap to the original image size
        img = cv2.imread(temp_image.name)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Save the original image to a file
        original_img_path = 'static/original_img.jpg'
        cv2.imwrite(original_img_path, img)

    # Convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)

    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.4 + img

    # Save the image to a file
    superimposed_img_path = 'static/superimposed_img.jpg'
    cv2.imwrite(superimposed_img_path, superimposed_img)

    return original_img_path, superimposed_img_path