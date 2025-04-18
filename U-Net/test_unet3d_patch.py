import numpy as np
import tensorflow as tf
from tensorflow import keras
import os


print(os.getcwd())

saved_model_path = 'saved_models_3d_unet/bin5_3d_unet_e2500_b5_lr0.0002_ct_ssim20250325-072712'
model = keras.models.load_model(saved_model_path, compile=False)


def SSIM_3D(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=0.05))  # Adjust for 3D data if needed


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

target_data = np.load('/home2/zwang/3dGE/data/recon/noisefree_bin5.npy')
input_data = np.load('/home2/zwang/3dGE/data/recon/noisefree_bin0.npy')

# Define patch size and stride (e.g., 50% overlap in height & width)
patch_size = (64, 128, 128)
stride = (64, 64, 64)  # No overlap in depth, 50% overlap in height/width

# Extract patches
image_patches, label_patches = extract_patches(input_data, target_data, patch_size, stride)

print(f"Extracted patches shape: {image_patches.shape}")  # (N_patches, 64, 128, 128)

def create_test_patches(image_patches, label_patches, batch_size=16):
    """
    创建用于测试的 patch 数据集，分别返回输入 patch 和目标 patch（ground truth）。
    """
    input_dataset = tf.data.Dataset.from_tensor_slices(image_patches)
    target_dataset = tf.data.Dataset.from_tensor_slices(label_patches)

    # 预处理：转换为 float32
    def preprocess(img):
        return tf.cast(img, tf.float32)

    input_dataset = input_dataset.map(preprocess).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    target_dataset = target_dataset.map(preprocess).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return input_dataset, target_dataset

input_dataset, target_dataset = create_test_patches(image_patches, label_patches, batch_size=5)

for batch in input_dataset.take(1):  # 取一个 batch
    print("Batch shape:", batch.shape)

for batch in target_dataset.take(1):  # 取一个 batch
    print("Batch shape:", batch.shape)

loss_fn = tf.keras.losses.MeanSquaredError(reduction="auto")
losses = []
ssims = []

preds = model.predict(np.concatenate(list(input_dataset.as_numpy_iterator()), axis=0))

print(np.concatenate(list(target_dataset.as_numpy_iterator()), axis=0).shape)
print(type(np.concatenate(list(target_dataset.as_numpy_iterator()), axis=0)))
print(type(preds))
print(tf.squeeze(preds).shape)
loss = loss_fn(np.concatenate(list(target_dataset.as_numpy_iterator()), axis=0), tf.squeeze(preds))

print(loss)

