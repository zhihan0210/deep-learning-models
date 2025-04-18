from keras.layers import Conv3D, BatchNormalization, Activation, MaxPool3D, Conv3DTranspose, \
    Concatenate, Input, AveragePooling3D, UpSampling3D
from keras.models import Model

def conv_block(input, num_filters):
    x = Conv3D(num_filters, (3, 3, 3), padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv3D(num_filters, (3, 3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)  # Only pool in height and width
    return x, p


def decoder_block(input, skip_features, num_filters):
    x = Conv3DTranspose(num_filters, (2, 2, 2), strides=(2, 2, 2), padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_3d_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv3D(1, (1, 1, 1), padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="3DUNet")
    return model

if __name__ == "__main__":
    import tensorflow as tf
    input_shape = (64, 128, 128, 1)  # (Depth, Height, Width, Channels)
    model = build_3d_unet(input_shape)
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True)
