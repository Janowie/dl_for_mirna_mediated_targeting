from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import layers
from tensorflow.keras.models import Model


@register_keras_serializable()
class ResBlock(layers.Layer):
    """
    Defines a Residual block. For more information refer to the original paper at https://arxiv.org/abs/1512.03385 .
    """

    def __init__(self, downsample=False, filters=16, kernel_size=3):

        super(ResBlock, self).__init__()

        # store parameters
        self.downsample = downsample
        self.filters = filters
        self.kernel_size = kernel_size

        # initialize inner layers
        self.conv1 = layers.Conv2D(kernel_size=self.kernel_size,
                                   strides=(1 if not self.downsample else 2),
                                   filters=self.filters,
                                   padding="same")
        self.activation1 = layers.ReLU()
        self.batch_norm1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(kernel_size=self.kernel_size,
                                   strides=1,
                                   filters=self.filters,
                                   padding="same")
        if self.downsample:
          self.conv3 = layers.Conv2D(kernel_size=1,
                                      strides=2,
                                      filters=self.filters,
                                      padding="same")

        self.activation2 = layers.ReLU()
        self.batch_norm2 = layers.BatchNormalization()

    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.activation1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)

        if self.downsample:
            inputs = self.conv3(inputs)

        x = layers.Add()([inputs, x])

        x = self.activation2(x)
        x = self.batch_norm2(x)

        return x

    def get_config(self):
        return {'filters': self.filters, 'downsample': self.downsample, 'kernel_size': self.kernel_size}


def create_resnet(name="model", size="small", output="sigmoid", input_shape=(50, 20, 1), num_blocks=(4, 6, 4)):
    """
    This function creates and returns a ResNet model.

    @param name: string - model name
    @param size: string - either of 'mini', 'small', 'large'; defines initial number of filters and dropout rate
    @param output: string - output activation function
    @param input_shape: tuple - input shape
    @param num_blocks: int - number of residual blocks before downsampling
    @return: tf.keras.Model
    """

    inputs = layers.Input(shape=input_shape)
    num_filters = 16
    initial_dropout = 0.15

    if size == "mini":
        num_filters = 8
    elif size == "large":
        num_filters = 32
        initial_dropout = 0.20

    x = layers.BatchNormalization()(inputs)
    x = layers.Conv2D(kernel_size=3,
                      strides=1,
                      filters=num_filters,
                      padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(initial_dropout)(x)

    for i in range(len(num_blocks)):
        num_b = num_blocks[i]
        for j in range(num_b):
            x = ResBlock(downsample=(j == 0 and i != 0), filters=num_filters)(x)
            x = layers.Dropout(initial_dropout + (i * 0.05))(x)

        num_filters *= 2

    x = layers.AveragePooling2D(4)(x)
    x = layers.Flatten()(x)

    if output == 'sigmoid':
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name=name)

    return model
