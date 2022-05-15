import tensorflow as tf
from tensorflow.keras import layers


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):

    """

    Helper function to apply convolution + BN.

    source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py

    @param x: input tensor
    @param filters: filters in `Conv2D`
    @param num_row: height of the convolution kernel
    @param num_col: width of the convolution kernel
    @param padding: padding mode in `Conv2D`
    @param strides: strides in `Conv2D`
    @param name: name
    @return: Output tensor after applying `Conv2D` and `BatchNormalization`.
    """

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(
        x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)

    return x


def create_inception(name="model", input_shape=(50, 20, 1), output='sigmoid'):

    """
    This function creates a modified Inception model. Code used here is a changed version from keras GitHub.

    original function: source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py

    @param name: string - model name
    @param input_shape: tuple - input shape
    @param output: string - output activation function
    @return: tf.keras.Model
    """

    input = layers.Input(shape=input_shape)
    channel_axis = 3

    # mixed 0: 50 x 20
    branch1x1 = conv2d_bn(input, 64, 1, 1)

    branch5x5 = conv2d_bn(input, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(input, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(input)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                           axis=channel_axis,
                           name='mixed0')

    x = layers.Dropout(0.25)(x)

    # mixed 1: 50 x 20
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                           axis=channel_axis,
                           name='mixed1')

    x = layers.Dropout(0.25)(x)

    # mixed 2: 50 x 20
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                           axis=channel_axis,
                           name='mixed2')

    x = layers.Dropout(0.25)(x)

    # mixed 3: 25 x 10
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool],
                           axis=channel_axis,
                           name='mixed3')

    x = layers.Dropout(0.25)(x)

    # mixed 4: 25 x 10
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                           axis=channel_axis,
                           name='mixed4')

    x = layers.Dropout(0.25)(x)

    # Classification block
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    if output == 'sigmoid':
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(2, activation='softmax')(x)

    # Create model.
    model = tf.keras.Model(input, outputs, name=name)

    return model
