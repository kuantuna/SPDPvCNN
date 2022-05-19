import tensorflow_addons as tfa
import architectures.helpers.constants as constants

from tensorflow.keras import layers, regularizers
from tensorflow import keras

'''
CONVMIXER
Reference: (https://github.com/keras-team/keras-io/blob/master/examples/vision/convmixer.py)
'''
hyperparameters = constants.hyperparameters["convmixer"]


''' ConvMixer Implementation
'''


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size,
                      strides=patch_size)(x)
    # , kernel_regularizer=regularizers.l2(1e-2)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    # ,
    #                   kernel_regularizer=regularizers.l2(1e-2)
    x = activation_block(x)

    return x


def get_conv_mixer_model(
        image_size=hyperparameters["image_size"], filters=hyperparameters["filters"], depth=hyperparameters["depth"],
        kernel_size=hyperparameters["kernel_size"], patch_size=hyperparameters["patch_size"], num_classes=3
):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = keras.Input((image_size, image_size, 1))

    # Extract patch embeddings.
    x = conv_stem(inputs, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


''' Compiling, Training and Evaluating
'''


def compile_model_optimizer(model):
    optimizer = keras.optimizers.Adadelta()

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Used to load a saved model to train and/or evaluate


def load_saved_model(path):
    return keras.models.load_model(path, custom_objects={'MyOptimizer': keras.optimizers.Adadelta})


def get_cm_model():
    print("Getting the ConvMixer model...")
    conv_mixer_model = get_conv_mixer_model()
    return compile_model_optimizer(conv_mixer_model)
