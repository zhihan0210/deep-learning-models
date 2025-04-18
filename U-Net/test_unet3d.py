import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os


print(os.getcwd())

saved_model_path = '/home/zhihan/cluster/zwang/multi_head/saved_models_3d_unet/3d_unet_depth64_ref4_tgt8_e10000_t5_v5_lr0.0002_128x128x64_ct_ssim20250208-070755'  # 120KeV
model = keras.models.load_model(saved_model_path, compile=False)


def SSIM_3D(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=0.05))  # Adjust for 3D data if needed


energy_tgt = 8
energy_ref = 4
path_tgt = '/home/zhihan/data/GE data 2/LAC/ReconData/resize/bin' + str(energy_tgt) + '.npy'
path_ref = '/home/zhihan/data/GE data 2/LAC/ReconData/resize/bin' + str(energy_ref) + '.npy'

img_tgt = np.load(path_tgt)
img_ref = np.load(path_ref)

imgs_tgt = []
imgs_ref = []
s_all = []

for i_slide in range(np.shape(img_tgt)[0]):
    x_tgt = img_tgt[i_slide,:,:]
    x_ref = img_ref[i_slide,:,:]
    s = np.sum(x_ref) / np.sum(x_tgt)
    s_all.append(s)
    imgs_tgt.append(x_tgt*s)
    imgs_ref.append(x_ref)
imgs_tgt = np.expand_dims(imgs_tgt, axis=3)
imgs_ref = np.expand_dims(imgs_ref, axis=3)

depth = 64
x_test = np.reshape(imgs_ref, (-1, depth, 128, 128, 1))  # Shape: (7, 40, 512, 512, 1)
y_test = np.reshape(imgs_tgt, (-1, depth, 128, 128, 1))  # Shape: (7, 40, 512, 512, 1)


i_slide_test = 0

ct_input = x_test[i_slide_test:i_slide_test+1]
ct_tgt = y_test[i_slide_test:i_slide_test+1]


# get prediction image
ct_output = model.predict(ct_input)

# compute loss
loss_fn = tf.keras.losses.MeanSquaredError(reduction="auto")
loss_value1 = loss_fn(ct_output, ct_tgt)

# compute ssim
ssim1 = SSIM_3D(ct_tgt.astype('double'), ct_output.astype('double'))

# display results
print(f'ct loss: {loss_value1}')
print(f'ct ssim: {ssim1}')

np.save('/home/zhihan/data/GE data 2/LAC/ReconData/resize/output_bin' + str(energy_tgt) + '.npy', ct_output)
np.save('/home/zhihan/data/GE data 2/LAC/ReconData/resize/s_bin' + str(energy_tgt) + '.npy', s_all)
exit()
i_slide_plt = 0
fig = plt.figure(figsize=(80, 30))
rows = 1
cols = 3

fig.add_subplot(rows, cols, 1)
plt.imshow(ct_input[:,i_slide_plt,:,:,:].reshape([512,512]), cmap='gray')
plt.clim(0,0.05)
plt.axis('off')
plt.title('Ref')

fig.add_subplot(rows, cols, 2)
plt.imshow(ct_output[:,i_slide_plt,:,:,:].reshape([512,512])/s_all[depth*i_slide_test+i_slide_plt], cmap='gray')
plt.clim(0,0.05)
plt.axis('off')
plt.title('Obtained')

fig.add_subplot(rows, cols, 3)
plt.imshow(ct_tgt[:,i_slide_plt,:,:,:].reshape([512,512])/s_all[depth*i_slide_test+i_slide_plt], cmap='gray')
plt.clim(0,0.05)
plt.axis('off')
plt.title('GT')

plt.show()
