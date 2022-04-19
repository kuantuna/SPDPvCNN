import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import time
import wandb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, regularizers
from tensorflow import keras
from wandb.keras import WandbCallback

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
CONVMIXER
Reference: (https://github.com/keras-team/keras-io/blob/master/examples/vision/convmixer.py)
'''


''' Setting Hyperparameter Values
'''
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 128
num_epochs = 500
filters_ = 256
depth = 8
kernel_size = 7
patch_size = 5

image_size = 67
etfList = ['XLF', 'XLU', 'QQQ', 'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE']
auto = tf.data.AUTOTUNE

TOTAL_STEPS = int((50000 / batch_size) * num_epochs)
WARMUP_STEPS = 10000
INIT_LR = 0.01
WAMRUP_LR = 0.002

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
lrs = [scheduled_lrs(step) for step in range(TOTAL_STEPS)]
plt.plot(lrs)
plt.xlabel("Step", fontsize=14)
plt.ylabel("LR", fontsize=14)
plt.grid()
plt.show()
'''

t = time.time()
epoch_counter = 1


''' Initializing Weights & Biases
'''
def initialize_wandb():
    wandb.init(project="Convmixer", entity="spdpvcnn",
            config={
                "model": "ConvMixer(w/regularizers)",
                "learning_rate": "WarmUpCosine",
                "epochs": num_epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "filters": filters_,
                "depth": depth,
                "kernel_size": kernel_size,
                "patch_size": patch_size,
                "threshold": 0.01,
                "image_size": image_size
            })


''' Dataset Preperation
'''
def load_dataset():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for etf in etfList:
        x_train.extend(np.load(f"../ETF/01/TrainData/x_train_{etf}.npy"))
        y_train.extend(np.load(f"../ETF/01/TrainData/y_train_{etf}.npy"))
        x_test.extend(np.load(f"../ETF/01/TestData/x_test_{etf}.npy"))
        y_test.extend(np.load(f"../ETF/01/TestData/y_test_{etf}.npy"))
    return x_train, y_train, x_test, y_test

# def print_data_counts(labelList):
#     labelDict = {
#         0: "Buy",
#         1: "Hold",
#         2: "Sell"
#     }
#     unique, counts = np.unique(labelList, return_counts=True)
#     for label, count in np.asarray((unique, counts)).T:
#         print("{} count: {}".format(labelDict[int(label)], int(count)))


def prepare_dataset(x_train, y_train, x_test):
    val_split = 0.1

    val_indices = int(len(x_train) * val_split)
    new_x_train, new_y_train = x_train[val_indices:], y_train[val_indices:]
    x_val, y_val = x_train[:val_indices], y_train[:val_indices]

    print(f"Training data samples: {len(new_x_train)}")
    print(f"Validation data samples: {len(x_val)}")
    print(f"Test data samples: {len(x_test)}")

    return new_x_train, new_y_train, x_val, y_val


def make_datasets(images, labels, is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(auto)

def get_finalized_datasets(new_x_train, new_y_train, x_val, y_val, x_test, y_test):
    train_dataset = make_datasets(new_x_train, new_y_train, is_train=True)
    val_dataset = make_datasets(x_val, y_val)
    test_dataset = make_datasets(x_test, y_test)
    return train_dataset, val_dataset, test_dataset




''' ConvMixer Implementation
'''
def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size, kernel_regularizer=regularizers.l2(1e-2))(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1, kernel_regularizer=regularizers.l2(1e-2))(x)
    x = activation_block(x)

    return x


def get_conv_mixer_model(
        image_size=image_size, filters=filters_, depth=depth, kernel_size=kernel_size, patch_size=patch_size, num_classes=3
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
    optimizer = tfa.optimizers.AdamW(
        learning_rate=scheduled_lrs, weight_decay=weight_decay
    ) 

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model



def run_experiment(model, test_dataset):
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[CmPrinter(test_dataset, epoch_counter), WandbCallback()]
    )

    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, model


# Runs at the end of every epoch and prints the confusion matrix
class CmPrinter(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, epoch_counter) -> None:
        super().__init__()
        self.test_dataset = test_dataset
        self.epoch_counter = epoch_counter


    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(self.test_dataset)
        classes = np.argmax(predictions, axis=1)
        print(confusion_matrix(y_test, classes))

        export_path_keras = f"../SavedModels/01/{int(t)}-{filters_}x{depth}-k{kernel_size}p{patch_size}e{self.epoch_counter}.h5"
        self.model.save_weights(export_path_keras)
        self.epoch_counter += 1


# Used to load a saved model to train and/or evaluate
def load_saved_model(path):
    return keras.models.load_model(path, custom_objects={'MyOptimizer': tfa.optimizers.AdamW})




if __name__ == "__main__":
    initialize_wandb()
    x_train, y_train, x_test, y_test = load_dataset()
    # print_data_counts(labelList)
    new_x_train, new_y_train, x_val, y_val = prepare_dataset(x_train, y_train, x_test)    
    train_dataset, val_dataset, test_dataset = get_finalized_datasets(new_x_train, new_y_train, x_val, y_val, x_test, y_test)


    conv_mixer_model = get_conv_mixer_model()                                                  # If you want to load a saved model and train it
    conv_mixer_model = compile_model_optimizer(conv_mixer_model)                               # Comment these two lines and uncomment the line below
    #conv_mixer_model = load_saved_model('../SavedModels/1647585460.h5')  

    history, conv_mixer_model = run_experiment(conv_mixer_model, test_dataset)

    predictions = conv_mixer_model.predict(test_dataset)
    classes = np.argmax(predictions, axis=1)

    cr = classification_report(y_test, classes)
    print(cr)
    f1 = f1_score(y_test, classes, average='micro')
    print(f1)
    