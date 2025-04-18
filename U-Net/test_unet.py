import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import os
from utils import attenuation


print(os.getcwd())

#saved_model_dir = '/home/zhihan/recon_results/real_data/models_unet'
saved_model_dir = 'saved_models_unet'
saved_model_name = 'unet_e2500_t10_v10_lr0.002_512x512_ct_ssim20220706-114509' #


saved_model_path = f'{saved_model_dir}/{saved_model_name}'

model = keras.models.load_model(saved_model_path, compile=False)



def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=0.05)


img_shape = [512, 512]


energy_tgt = 80
energy_ref = 40


#mul = 1.0 # for 40KeV
#mul = 1.3034588890645624 # for 60KeV
#mul = 1.4600299617587578 # for 80KeV
#mul = 1.571715730661157 # for 100KeV
#mul = 1.662940058219701 # for 120KeV
#mul = 1.7446877621129084 # for 140KeV

#img_tgt = attenuation(energy_tgt,2).transpose(1,0,2)
#img_ref = attenuation(energy_ref,2).transpose(1,0,2)

path_tgt = '/home/zhihan/data/patient2/RME_therassemartine/DICOM/atten_' + str(energy_tgt) + 'KeV.npy'
path_ref = '/home/zhihan/data/patient2/RME_therassemartine/DICOM/atten_' + str(energy_ref) + 'KeV.npy'
#path_tgt = '/home/zhihan/data/patient1/RME_baguesse_philippe/DICOM/atten_' + str(energy_tgt) + 'KeV.npy'
#path_ref = '/home/zhihan/data/patient1/RME_baguesse_philippe/DICOM/atten_' + str(energy_ref) + 'KeV.npy'
img_tgt = np.load(path_tgt)
img_ref = np.load(path_ref)

imgs_tgt = []
imgs_ref = []
s_all = []

for i_slide in range(100):
    x_tgt = img_tgt[:,:,i_slide]
    #x_tgt = cv2.resize(x_tgt, (128, 128), interpolation=cv2.INTER_LINEAR)
    #x_tgt = x_tgt * 4
    x_ref = img_ref[:,:,i_slide]
    #x_ref = cv2.resize(x_ref, (128, 128), interpolation=cv2.INTER_LINEAR)
    #x_ref = x_ref * 4
    s = np.sum(x_ref) / np.sum(x_tgt)
    s_all.append(s)
    imgs_tgt.append(x_tgt*s)
    imgs_ref.append(x_ref)

imgs_tgt = np.expand_dims(imgs_tgt, axis=3)
imgs_ref = np.expand_dims(imgs_ref, axis=3)

x_test = imgs_ref
y_test = imgs_tgt

x_val = imgs_ref
y_val = imgs_tgt



#ct_input = x_test[3:4]
#ct_tgt = y_test[3:4]


ct_input = []
ct_tgt = []
ct_output = []
loss_value1 = []
ssim1 = []

for i in range(5):
    x_temp = x_test[10*i:10*i+1]
    y_temp = y_test[10*i:10*i+1]
    output_temp = model.predict(x_temp)
    ct_input.append(x_temp)
    ct_tgt.append(y_temp)
    ct_output.append(output_temp)
    loss_fn = tf.keras.losses.MeanSquaredError(reduction="auto")
    loss_value1.append(loss_fn(output_temp, y_temp))
    ssim1.append(SSIM(y_temp.astype('double'), output_temp.astype('double')))



'''

# get prediction image
ct_output = model.predict(ct_input)

# compute loss
loss_fn = tf.keras.losses.MeanSquaredError(reduction="auto")
loss_value1 = loss_fn(ct_output, ct_tgt)

# compute ssim
ssim1 = SSIM(ct_tgt.astype('double'), ct_output.astype('double'))

# display results
print(f'ct loss: {loss_value1}')
print(f'ct ssim: {ssim1[0]}')
'''
#ct_output = ct_output/mul
#ct_tgt = ct_tgt/mul

print(f'loss: {loss_value1}')
print(f'ssim: {ssim1}')

output = [ct_output[i]/s_all[i] for i in range(5)]
tgt = [ct_tgt[i]/s_all[i] for i in range(5)]
ct_output = output
ct_tgt = tgt

# create subplots
fig = plt.figure(figsize=(20, 30))
rows = 5
cols = 3


for i in range(5):
    fig.add_subplot(rows, cols, 3*i+1)
    plt.imshow(ct_input[i].reshape(img_shape), cmap='gray')
    plt.clim(0, 0.05)
    plt.axis('off')
    plt.title('CT ref-low energy')

    fig.add_subplot(rows, cols, 3*i+2)
    plt.imshow(ct_output[i].reshape(img_shape), cmap='gray')
    plt.clim(0, 0.05)
    plt.axis('off')
    plt.title('Obtained CT high energy')

    fig.add_subplot(rows, cols, 3*i+3)
    plt.imshow(ct_tgt[i].reshape(img_shape), cmap='gray')
    plt.clim(0, 0.05)
    plt.axis('off')
    plt.title('CT tgt-high energy')



'''
fig = plt.figure(figsize=(20, 30))
rows = 1
cols = 3

fig.add_subplot(rows, cols, 1)
plt.imshow(ct_input.reshape(img_shape), cmap='gray')
plt.clim(0,0.05)
plt.axis('off')
plt.title('CT ref-low energy')

ax2 = fig.add_subplot(rows, cols, 2)
ax2.text(0, 0, f'SSIM: {ssim1[0]:.4f}', color='green', fontsize=10, transform=ax2.transAxes,
         verticalalignment='bottom')
plt.imshow(ct_output.reshape(img_shape), cmap='gray')
plt.clim(0,0.05)
plt.axis('off')
plt.title('Obtained CT high energy')

fig.add_subplot(rows, cols, 3)
plt.imshow(ct_tgt.reshape(img_shape), cmap='gray')
plt.clim(0,0.05)
plt.axis('off')
plt.title('CT tgt-high energy')


ct_low = np.squeeze(ct_input,axis=3)
ct_low = np.squeeze(ct_low,axis=0)
ct_obtain = np.squeeze(ct_output,axis=3)
ct_obtain = np.squeeze(ct_obtain,axis=0)
ct_high = np.squeeze(ct_tgt,axis=3)
ct_high = np.squeeze(ct_high,axis=0)

position = 60

plt.figure(2)

plt.plot(ct_low[position,:],color='b',linewidth=1.0,label='low')
plt.plot(ct_obtain[position,:],color='g',linewidth=1.0,label='obtain')
plt.plot(ct_high[position,:],color='r',linewidth=1.0,label='high')

plt.legend()


'''

plt.show()
