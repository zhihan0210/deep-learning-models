import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from unet3d import build_3d_unet


def SSIM_3D(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=0.05))  # Adjust for 3D data if needed


import numpy as np


def extract_patches(volume, label, patch_size=(64, 128, 128), stride=(64, 64, 64)):
    """
    Extract overlapping 3D patches from a volume and corresponding label.

    Parameters:
        volume (numpy array): Input image of shape (D, H, W)
        label (numpy array): Corresponding target of shape (D, H, W)
        patch_size (tuple): Size of the patch (depth, height, width)
        stride (tuple): Stride for patch extraction (depth, height, width)

    Returns:
        np.array: Extracted patches from the volume
        np.array: Corresponding extracted patches from the label
    """
    D, H, W = volume.shape
    d_s, h_s, w_s = stride
    d_p, h_p, w_p = patch_size

    image_patches, label_patches = [], []

    for z in range(0, D - d_p + 1, d_s):  # Move along depth
        for y in range(0, H - h_p + 1, h_s):  # Move along height
            for x in range(0, W - w_p + 1, w_s):  # Move along width
                img_patch = volume[z:z + d_p, y:y + h_p, x:x + w_p]
                lbl_patch = label[z:z + d_p, y:y + h_p, x:x + w_p]

                image_patches.append(img_patch)
                label_patches.append(lbl_patch)

    return np.array(image_patches), np.array(label_patches)

# Assume original input and target are stored as NumPy arrays
input_data = np.load('/home2/zwang/3dGE/data/bin0.npy')  # Simulated 6 phantoms
target_data = np.load('/home2/zwang/3dGE/data/bin5.npy')  # Corresponding target

# Define patch size and stride (e.g., 50% overlap in height & width)
patch_size = (64, 128, 128)
stride = (64, 64, 64)  # No overlap in depth, 50% overlap in height/width

# Extract patches
image_patches, label_patches = extract_patches(input_data, target_data, patch_size, stride)

print(f"Extracted patches shape: {image_patches.shape}")  # (N_patches, 64, 128, 128)


def create_tf_dataset(image_patches, label_patches, batch_size=16, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_patches, label_patches))

    # Normalize images and convert to TensorFlow tensors
    def preprocess(img, lbl):
        img = tf.cast(img, tf.float32)
        lbl = tf.cast(lbl, tf.float32)
        return img, lbl

    dataset = dataset.map(preprocess)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_patches))

    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Create TensorFlow dataset
train_dataset = create_tf_dataset(image_patches, label_patches, batch_size=5)

input_shape = (64, 128, 128, 1)  # Depth, Height, Width, Channels
model = build_3d_unet(input_shape)

learning_rate = 2e-4
epoch = 1500

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='mean_squared_error', metrics=[SSIM_3D])

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") + \
          f"bin0_3d_unet_patch_e{epoch}_b5_lr{learning_rate}"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_dataset, epochs=epoch, callbacks=[tensorboard_callback])

saved_model_dir = 'saved_models_3d_unet'
saved_model_name = f'bin0_3d_unet_patch_e{epoch}_b5_lr{learning_rate}_ct_ssim' + \
                   datetime.now().strftime("%Y%m%d-%H%M%S")
saved_model_path = f'{saved_model_dir}/{saved_model_name}'
model.save(saved_model_path)

# Clear session
keras.backend.clear_session()
