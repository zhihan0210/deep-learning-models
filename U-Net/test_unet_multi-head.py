# -- coding: utf-8 --**

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os


print(os.getcwd())


saved_model_path = 'saved_models_unet/xcat/128/multihead/normalized/unet_e2500_t10_v10_lr0.0002_128x128_ct_ssim20240312-173817'

model = keras.models.load_model(saved_model_path, compile=False)



def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=0.05)


img_shape = [128, 128]


imgs40 = np.load('/home/zhihan/data/xcat/128/female1/atten_40KeV.npy')
imgs60 = np.load('/home/zhihan/data/xcat/128/female1/atten_60KeV.npy')
imgs80 = np.load('/home/zhihan/data/xcat/128/female1/atten_80KeV.npy')
imgs100 = np.load('/home/zhihan/data/xcat/128/female1/atten_100KeV.npy')
imgs120 = np.load('/home/zhihan/data/xcat/128/female1/atten_120KeV.npy')
imgs140 = np.load('/home/zhihan/data/xcat/128/female1/atten_140KeV.npy')

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


imgs_ref = data[0, :, :, :, :]
imgs_tgt = data

x_test = imgs_ref
y_test = imgs_tgt

i_slide_test = 50

ct_input = x_test[i_slide_test:i_slide_test+1]
ct_tgt = y_test[:,i_slide_test:i_slide_test+1,:,:,:]




# get prediction image
ct_output = model.predict(ct_input)
loss_fn = tf.keras.losses.MeanSquaredError(reduction="auto")
# compute loss
for i_e in range(6):
    loss_value1 = loss_fn(ct_output[i_e], ct_tgt[i_e,:,:,:])
    print(loss_value1)
    ssim1 = SSIM(ct_tgt[i_e,:,:,:].astype('double'), ct_output[i_e].astype('double'))
    print(ssim1)


# create subplots
fig = plt.figure(figsize=(80, 30))
rows = 2
cols = 6


for i in range(6):
    fig.add_subplot(rows, cols, i+1)
    plt.imshow(ct_tgt[:,:,:,i:i+1].reshape(img_shape), cmap='gray')
    plt.clim(0, 0.04)
    plt.axis('off')
    plt.title('GT')

    fig.add_subplot(rows, cols, i+7)
    plt.imshow(ct_output[i].reshape(img_shape), cmap='gray')
    plt.clim(0, 0.04)
    plt.axis('off')
    plt.title('Obtained')


plt.show()
