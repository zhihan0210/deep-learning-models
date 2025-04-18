import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from unet import build_unet_with_multiple_heads


def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=0.05)

imgs40 = np.load('/home/zhihan/data/xcat/128/male1/atten_40KeV.npy')
imgs60 = np.load('/home/zhihan/data/xcat/128/male1/atten_60KeV.npy')
imgs80 = np.load('/home/zhihan/data/xcat/128/male1/atten_80KeV.npy')
imgs100 = np.load('/home/zhihan/data/xcat/128/male1/atten_100KeV.npy')
imgs120 = np.load('/home/zhihan/data/xcat/128/male1/atten_120KeV.npy')
imgs140 = np.load('/home/zhihan/data/xcat/128/male1/atten_140KeV.npy')

pixel_size = 0.3 # unit: cm/pixel
mu_cm_water = [0.2683,0.2059,0.1837,0.1707,0.1614,0.1538] # unit: /cm
mu_cm_air = [0.0786,0.0609,0.0545,0.0507,0.0479,0.0457] # unit: /cm
mu_cm_max = []
mu_pixel_max = []
for i_e in range(6):
    mu_max = 3071 * (mu_cm_water[i_e] - mu_cm_air[i_e]) / 1000 + mu_cm_water[i_e]
    mu_cm_max.append(mu_max) # unit: /cm
    mu_pixel_max.append(mu_max * pixel_size)


def normalize_data(data, mu_max):
    nor = np.zeros_like(data)  # Create an array to store normalized values
    nor[data / mu_max > 1] = 1  # Set values greater than 1 to 1
    nor[data / mu_max < 0] = 0  # Set values less than 0 to 0
    nor[(data / mu_max >= 0) & (data / mu_max <= 1)] = data[(data / mu_max >= 0) & (data / mu_max <= 1)] / mu_max  # Set other values to data/mu_max
    return nor

data = np.zeros([6,140,128,128,1])
for i_slide in range(np.shape(imgs40)[0]):
    data[0, i_slide, :, :, 0] = normalize_data(imgs40[i_slide, :, :], mu_pixel_max[0])
    data[1, i_slide, :, :, 0] = normalize_data(imgs60[i_slide, :, :], mu_pixel_max[1])
    data[2, i_slide, :, :, 0] = normalize_data(imgs80[i_slide, :, :], mu_pixel_max[2])
    data[3, i_slide, :, :, 0] = normalize_data(imgs100[i_slide, :, :], mu_pixel_max[3])
    data[4, i_slide, :, :, 0] = normalize_data(imgs120[i_slide, :, :], mu_pixel_max[4])
    data[5, i_slide, :, :, 0] = normalize_data(imgs140[i_slide, :, :], mu_pixel_max[5])


imgs_tgt = [data[i,:,:,:,:] for i in range(6)]
imgs_ref = data[0, :, :, :, :]


x_train = imgs_ref
y_train = imgs_tgt

x_val = imgs_ref
y_val = imgs_tgt


# set hyperparameters
learning_rate = 2e-4
epoch = 1000
train_batch_size = 5
val_batch_size = 5

# get model
input_shape = (128, 128, 1)
num_heads = 6
model = build_unet_with_multiple_heads(input_shape, num_heads)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='mean_squared_error', metrics=[SSIM])

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") + \
          f"unet_e{epoch}_t{train_batch_size}_v{val_batch_size}_lr{learning_rate}"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(x=x_train, y=y_train, epochs=epoch, batch_size=train_batch_size, validation_data=(x_val, y_val), shuffle=True,
                    callbacks=[tensorboard_callback])

saved_model_dir = 'saved_models_unet'
saved_model_name = f'unet_e{epoch}_t{train_batch_size}_v{val_batch_size}_lr{learning_rate}_128x128_ct_ssim' + \
                   datetime.now().strftime("%Y%m%d-%H%M%S")
saved_model_path = f'{saved_model_dir}/{saved_model_name}'
model.save(saved_model_path)
keras.backend.clear_session()
