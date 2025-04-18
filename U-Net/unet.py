"""
https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture
"""

from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, \
    Concatenate, Input
from keras.models import Model


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape):
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

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="U-Net")
    return model

def build_unet_with_multiple_heads(input_shape, num_heads):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    # Decoder with multiple heads
    heads = []
    for i in range(num_heads):
        d1 = decoder_block(b1, s4, 512)
        d2 = decoder_block(d1, s3, 256)
        d3 = decoder_block(d2, s2, 128)
        d4 = decoder_block(d3, s1, 64)

        output = Conv2D(1, 1, padding="same", activation="sigmoid", name=f"head_{i + 1}_output")(d4)
        heads.append(output)

    model = Model(inputs, heads, name="U-Net_with_Multiple_Heads")
    return model


if __name__ == "__main__":
    import tensorflow as tf
    input_shape = (128, 128, 1)
    model = build_unet(input_shape)
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True)
