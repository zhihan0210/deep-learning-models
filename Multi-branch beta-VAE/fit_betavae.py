from datetime import datetime

import numpy as np
from tensorflow import keras
import tensorflow as tf

from betavae_B import build_dual_betavae


try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
except:
    print('0')


def input_fn(x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, img_size=64, batch_size=32, buffer_size=32, shuffle=True):
    """https://stackoverflow.com/questions/52582275/tf-data-with-multiple-inputs-outputs-in-keras"""
    x_train1 = x_train1.reshape([x_train1.shape[0], img_size, img_size, 1])
    x_train2 = x_train2.reshape([x_train2.shape[0], img_size, img_size, 1])
    x_train3 = x_train3.reshape([x_train3.shape[0], img_size, img_size, 1])
    x_train4 = x_train4.reshape([x_train4.shape[0], img_size, img_size, 1])
    x_train5 = x_train5.reshape([x_train5.shape[0], img_size, img_size, 1])
    x_train6 = x_train6.reshape([x_train6.shape[0], img_size, img_size, 1])

    def generator():
        for i1, i2, i3, i4, i5, i6 in zip(x_train1, x_train2, x_train3, x_train4, x_train5, x_train6):
            yield {"input_1": i1, "input_2": i2, "input_3": i3, "input_4": i4, "input_5": i5, "input_6": i6}

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_types=({"input_1": tf.float32, "input_2": tf.float32, "input_3": tf.float32, "input_4": tf.float32, "input_5": tf.float32, "input_6": tf.float32}))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset


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


def train_betavae(image_size=128, latent_dim=128, inter_dim=16, batch_size=64, learning_rate=1e-4,
                 filters=16, layers_n=3, epoch=500, merge_type='concatenate'):
    # get xcat_data
    imgs40 = np.load('/home2/zwang/betavae/data/data_40keV.npy')
    imgs60 = np.load('/home2/zwang/betavae/data/data_60keV.npy')
    imgs80 = np.load('/home2/zwang/betavae/data/data_80keV.npy')
    imgs100 = np.load('/home2/zwang/betavae/data/data_100keV.npy')
    imgs120 = np.load('/home2/zwang/betavae/data/data_120keV.npy')
    imgs140 = np.load('/home2/zwang/betavae/data/data_140keV.npy')


    images40 = np.expand_dims(imgs40, axis=3)
    images40 = normalize_data(images40, mu_pixel_max[0])
    
    images60 = np.expand_dims(imgs60, axis=3)
    images60 = normalize_data(images60, mu_pixel_max[1])

    images80 = np.expand_dims(imgs80, axis=3)
    images80 = normalize_data(images80, mu_pixel_max[2])

    images100 = np.expand_dims(imgs100, axis=3)
    images100 = normalize_data(images100, mu_pixel_max[3])

    images120 = np.expand_dims(imgs120, axis=3)
    images120 = normalize_data(images120, mu_pixel_max[4])

    images140 = np.expand_dims(imgs140, axis=3)
    images140 = normalize_data(images140, mu_pixel_max[5])

    # build  model
    model_arch = 'beta_vae'
    input_shape = (image_size, image_size, 1)
    build_model = build_dual_betavae


    model = build_model(input_shape=input_shape, inter_dim=inter_dim, latent_dim=latent_dim, filters=filters,
                        layers_n=layers_n, g=1, merge_type=merge_type)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    model_name = f'{model_arch}_gamma1_xcat_lungs{image_size}x{image_size}_p{imgs60.shape[0]}_lr{learning_rate}_' \
                 f'ld{latent_dim}_id{inter_dim}_batch{batch_size}_f{filters}_l{layers_n}_e{epoch}_m{merge_type}'

    # create tensorboard callback
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/fit/{model_name}_{date_time}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # create checkpoint callback
    checkpoint_filepath = f'tmp/{model_name}_{date_time}/checkpoint_{epoch}'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        # verbose=1,
        monitor='total_loss',
        mode='min',
        save_freq=20,
        save_best_only=True)

    # train vae
    model.fit(input_fn(images40, images60, images80, images100, images120, images140, img_size=image_size, batch_size=batch_size), epochs=epoch, verbose=2,
              callbacks=[tensorboard_callback, model_checkpoint_callback])

    # save models encoder and decoder
    model.encoder.save(f'saved_models/betavae/{model_name}_{date_time}/encoder')


def train_singlemode_bvae():
    pass


if __name__ == "__main__":
    image_size = 128
    latent_dim = 128
    inter_dim = 256
    batch_size = 128
    filters = 64
    layers_n = 5
    learning_rate = 2e-4
    merge_type = 'concatenate'
    epoch = 4000

    train_betavae(image_size=image_size, inter_dim=inter_dim, latent_dim=latent_dim,
                    batch_size=batch_size, learning_rate=learning_rate, epoch=epoch, filters=filters, layers_n=layers_n,
                    merge_type=merge_type)
