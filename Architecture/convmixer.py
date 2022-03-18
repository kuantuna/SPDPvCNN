import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf

from tensorflow.keras import layers, regularizers
from tensorflow import keras
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

import wandb

# from callbacks import *
import time

wandb.init(project="test-project", entity="spdpvcnn",
           config={
               "model": "ConvMixer",
               "learning_rate": "0.001",
               "epochs": 60,
               "batch_size": 16,
               "weight_decay": 0.0001,
               "image_size": 11,
               "filters": 768,
               "depth": 32,
               "kernel_size": 5,
               "patch_size": 1,
               "threshold": 0.0038
           })
'''
sweep_config = {
    "method": "random",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    }
}

parameters_dict = {
    
}
'''
imageList = np.load("C:\\Users\\Tuna\\Desktop\\2022-spring\\CS402\\SPDPvCNN\\ETF\\Images.npy")
labelList = np.load("C:\\Users\\Tuna\\Desktop\\2022-spring\\CS402\\SPDPvCNN\\ETF\\Labels.npy")

unique, counts = np.unique(labelList, return_counts=True)
print(np.asarray((unique, counts)).T)

'''
IMPLEMENTING THE CONVMIXER
Reference: (https://github.com/keras-team/keras-io/blob/master/examples/vision/convmixer.py)
'''
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 16
num_epochs = 60




'''
TOTAL_STEPS = int((50000 / batch_size) * num_epochs)
WARMUP_STEPS = 10000
INIT_LR = 0.01
WAMRUP_LR = 0.002
'''





x_train, x_test, y_train, y_test = train_test_split(imageList, labelList, test_size=0.1, random_state=41)
val_split = 0.1

val_indices = int(len(x_train) * val_split)
new_x_train, new_y_train = x_train[val_indices:], y_train[val_indices:]
x_val, y_val = x_train[:val_indices], y_train[:val_indices]

print(f"Training data samples: {len(new_x_train)}")
print(f"Validation data samples: {len(x_val)}")
print(f"Test data samples: {len(x_test)}")

image_size = 11
auto = tf.data.AUTOTUNE

data_augmentation = keras.Sequential(
    [layers.RandomCrop(image_size, image_size), layers.RandomFlip("horizontal"), ],
    name="data_augmentation",
)


def make_datasets(images, labels, is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    if is_train:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x), y), num_parallel_calls=auto
        )
    return dataset.prefetch(auto)


train_dataset = make_datasets(new_x_train, new_y_train, is_train=True)
val_dataset = make_datasets(x_val, y_val)
test_dataset = make_datasets(x_test, y_test)


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    #, kernel_regularizer=regularizers.l2(1e-2)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    #, kernel_regularizer=regularizers.l2(1e-2)
    x = activation_block(x)

    return x


def get_conv_mixer_256_8(
        image_size=11, filters=768, depth=32, kernel_size=5, patch_size=1, num_classes=3
):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = keras.Input((image_size, image_size, 1))
    # x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(inputs, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)






'''
class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )






scheduled_lrs = WarmUpCosine(
    learning_rate_base=INIT_LR,
    total_steps=TOTAL_STEPS,
    warmup_learning_rate=WAMRUP_LR,
    warmup_steps=WARMUP_STEPS,
)
'''
'''
lrs = [scheduled_lrs(step) for step in range(TOTAL_STEPS)]
plt.plot(lrs)
plt.xlabel("Step", fontsize=14)
plt.ylabel("LR", fontsize=14)
plt.grid()
plt.show()
'''






def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    # scheduled_lrs

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    # , f1_m,precision_m, recall_m, TP, TN, FP, FN

    # checkpoint_filepath = "C:/Users/Tuna/Desktop/2022-spring/CS402/SPDPvCNN/Results"  # fix here
    '''
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    '''
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[WandbCallback()],
    )

    # model.load_weights(checkpoint_filepath)
    # model.save('../saved_model/wc0317', save_format='tf')
    t = time.time()
    export_path_keras = "./{}.h5".format(int(t))
    model.save(export_path_keras)
    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, model


conv_mixer_model = get_conv_mixer_256_8()
history, conv_mixer_model = run_experiment(conv_mixer_model)

predictions = conv_mixer_model.predict(test_dataset)
classes = np.argmax(predictions, axis=1)
cm = confusion_matrix(y_test, classes)
print(cm)
cr = classification_report(y_test, classes)
print(cr)
f1 = f1_score(y_test, classes, average='micro')
print(f1)
'''
print(history.history.keys())


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''