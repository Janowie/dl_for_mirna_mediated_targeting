import tensorflow as tf
from tensorflow.keras import layers


def create_cnn(name="model", kernel_size=4, cnn_num=6, dropout_rate=0.2, output="sigmoid", input_shape=(50, 20, 1)):

    """
    This function creates a CNN 'optimized' model, designed by Eva Klimentova. This is a copy of her function which can
    be found on her GitHub page under the name 'make_architecture'. We only include it since it was used as a baseline
    in our experiments.

    source: https://github.com/evaklimentova/smallRNA_binding/blob/main/Additional_scripts/training.ipynb

    @param name: string - model name
    @param kernel_size: kernel size
    @param cnn_num: int - number of CNN "blocks" with increasing number of filters (starting at 32)
    @param dropout_rate: int - dropout rate
    @param output: string - output activation function
    @param input_shape: tuple - input shape, defaults to (50,20,1)
    @return: tf.keras.Model
    """

    pool_size = 2
    dense_num = 3

    x = layers.Input(shape=input_shape, dtype='float32', name='main_input')
    main_input = x

    for cnn_i in range(cnn_num):
        x = layers.Conv2D(
            filters=32 * (cnn_i + 1),
            kernel_size=(kernel_size, kernel_size),
            padding="same",
            data_format="channels_last",
            name="conv_" + str(cnn_i + 1))(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(pool_size, pool_size), padding='same', name='Max_' + str(cnn_i + 1))(x)
        x = layers.Dropout(rate=dropout_rate)(x)

    x = layers.Flatten(name='2d_matrix')(x)

    for dense_i in range(dense_num):
        neurons = 32 * (cnn_num - dense_i)
        x = layers.Dense(neurons)(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(rate=dropout_rate)(x)

    if output == "sigmoid":
        main_output = layers.Dense(1, activation='sigmoid', name='main_output')(x)
    else:
        main_output = layers.Dense(2, activation='softmax', name='main_output')(x)

    model = tf.keras.Model(inputs=[main_input], outputs=[main_output], name=name)

    return model
