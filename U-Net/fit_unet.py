import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from unet import build_unet
import cv2


def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=0.05)




energy_tgt = 80
energy_ref = 40
path_tgt = '/home/zhihan/data/patient1/RME_baguesse_philippe/DICOM/atten_' + str(energy_tgt) + 'KeV.npy'
path_ref = '/home/zhihan/data/patient1/RME_baguesse_philippe/DICOM/atten_' + str(energy_ref) + 'KeV.npy'
img_tgt = np.load(path_tgt)
img_ref = np.load(path_ref)

#mul = np.mean(img_ref)/np.mean(img_tgt)
#img_tgt = img_tgt*mul


imgs_tgt = []
imgs_ref = []


for i_slide in range(100):
    x_tgt = img_tgt[:,:,i_slide]
    #x_tgt = cv2.resize(x_tgt,(128,128), interpolation=cv2.INTER_LINEAR)
    #x_tgt = x_tgt*4
    x_ref = img_ref[:,:,i_slide]
    #x_ref = cv2.resize(x_ref,(128,128), interpolation=cv2.INTER_LINEAR)
    #x_ref = x_ref*4
    s = np.sum(x_ref) / np.sum(x_tgt)
    imgs_tgt.append(x_tgt*s)
    imgs_ref.append(x_ref)



imgs_tgt = np.expand_dims(imgs_tgt, axis=3)
imgs_ref = np.expand_dims(imgs_ref, axis=3)



x_train = imgs_ref
y_train = imgs_tgt

x_val = imgs_ref
y_val = imgs_tgt


# set hyperparameters
learning_rate = 2e-3
epoch = 2500
train_batch_size = 10
val_batch_size = 10

# get model
input_shape = (512, 512, 1)
model = build_unet(input_shape)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='mean_squared_error', metrics=[SSIM])

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") + \
          f"unet_e{epoch}_t{train_batch_size}_v{val_batch_size}_lr{learning_rate}"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(x=x_train, y=y_train, epochs=epoch, batch_size=train_batch_size, validation_data=(x_val, y_val), shuffle=True,
                    callbacks=[tensorboard_callback])

saved_model_dir = 'saved_models_unet'
saved_model_name = f'unet_e{epoch}_t{train_batch_size}_v{val_batch_size}_lr{learning_rate}_512x512_ct_ssim' + \
                   datetime.now().strftime("%Y%m%d-%H%M%S")
saved_model_path = f'{saved_model_dir}/{saved_model_name}'
model.save(saved_model_path)
keras.backend.clear_session()
