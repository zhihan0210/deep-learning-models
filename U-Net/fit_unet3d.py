import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from unet3d import build_3d_unet

def SSIM_3D(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=0.05))  # Adjust for 3D data if needed

# target energy (keV)
energy_tgt = 40 # 60 or 80 or 100 or 120 or 140
# reference energy: lowest energy (keV)
energy_ref = 40

# load datasets for training from img_ref to img_tgt
path_tgt = '/home/zhihan/data/xcat/512/male1/atten_' + str(energy_tgt) + 'KeV.npy'
path_ref = '/home/zhihan/data/xcat/512/male1/atten_' + str(energy_ref) + 'KeV.npy'
img_tgt = np.load(path_tgt)
img_ref = np.load(path_ref)

# organize the datasets and add a scale to each slide of target image: because the attenuation value for high energies is much smaller than the reference
imgs_tgt = []
imgs_ref = []
for i_slide in range(np.shape(img_tgt)[0]):
    x_tgt = img_tgt[i_slide,:,:]
    x_ref = img_ref[i_slide,:,:]
    s = np.sum(x_ref) / np.sum(x_tgt)
    imgs_tgt.append(x_tgt*s)
    imgs_ref.append(x_ref)
imgs_tgt = np.expand_dims(imgs_tgt, axis=3)
imgs_ref = np.expand_dims(imgs_ref, axis=3)[:256,:,:,:]

# Reshape data into 3D volumes
depth = 8
x_train = np.reshape(imgs_ref[:256,:,:,:], (-1, depth, 512, 512, 1))  # Shape: (7, 40, 512, 512, 1)
y_train = np.reshape(imgs_tgt[:256,:,:,:], (-1, depth, 512, 512, 1))  # Shape: (7, 40, 512, 512, 1)

x_val = x_train  # Example: you can split this properly if you have a separate validation set
y_val = y_train

learning_rate = 2e-4
epoch = 700
train_batch_size = 1  # Lower batch size for 3D due to higher memory usage
val_batch_size = 1

# Get 3D U-Net model
input_shape = (8, 512, 512, 1)  # Depth, Height, Width, Channels
model = build_3d_unet(input_shape)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='mean_squared_error', metrics=[SSIM_3D])

# TensorBoard logging
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") + \
          f"3d_unet_e{epoch}_t{train_batch_size}_v{val_batch_size}_lr{learning_rate}"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train model
history = model.fit(x=x_train, y=y_train, epochs=epoch, batch_size=train_batch_size,
                    validation_data=(x_val, y_val), shuffle=True,
                    callbacks=[tensorboard_callback])

# Save model
saved_model_dir = 'saved_models_3d_unet'
saved_model_name = f'3d_unet_e{epoch}_t{train_batch_size}_v{val_batch_size}_lr{learning_rate}_512x512x8_ct_ssim' + \
                   datetime.now().strftime("%Y%m%d-%H%M%S")
saved_model_path = f'{saved_model_dir}/{saved_model_name}'
model.save(saved_model_path)

# Clear session
keras.backend.clear_session()

